# Copyright © 2025 Apple Inc.
# Gemma 4 text model for mlx-lm
# Ported from HuggingFace transformers (modeling_gemma4.py)

import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "gemma4_text"
    hidden_size: int = 2304
    num_hidden_layers: int = 30
    intermediate_size: int = 9216
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 256
    global_head_dim: int = 512
    num_global_key_value_heads: Optional[int] = None
    rms_norm_eps: float = 1e-6
    vocab_size: int = 262144
    sliding_window: int = 512
    max_position_embeddings: int = 131072
    hidden_activation: str = "gelu_pytorch_tanh"
    layer_types: Optional[List[str]] = None
    rope_parameters: Optional[Dict] = None
    # Per-layer input embeddings (PLE)
    hidden_size_per_layer_input: int = 256
    vocab_size_per_layer_input: int = 262144
    # KV sharing
    num_kv_shared_layers: int = 0
    attention_k_eq_v: bool = False
    # Mixture of Experts
    enable_moe_block: bool = False
    num_experts: Optional[int] = None
    top_k_experts: Optional[int] = None
    moe_intermediate_size: Optional[int] = None
    # Other
    use_double_wide_mlp: bool = False
    final_logit_softcapping: Optional[float] = None
    attention_bias: bool = False
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.layer_types is None:
            sliding_window_pattern = 6
            self.layer_types = [
                (
                    "sliding_attention"
                    if (i + 1) % sliding_window_pattern != 0
                    else "full_attention"
                )
                for i in range(self.num_hidden_layers)
            ]
            # Ensure the last layer is always full attention
            self.layer_types[-1] = "full_attention"

        if self.num_global_key_value_heads is None:
            self.num_global_key_value_heads = self.num_key_value_heads

        if self.rope_parameters is None:
            self.rope_parameters = {
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": 10_000.0,
                },
                "full_attention": {
                    "rope_type": "proportional",
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1_000_000.0,
                },
            }


# ---------------------------------------------------------------------------
# Norm helpers
# ---------------------------------------------------------------------------


class RMSNoScale(nn.Module):
    """RMSNorm without a learnable scale (used for v_norm, router norm)."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, None, self.eps)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"

        # Head dimensions differ between sliding and full attention layers
        if self.is_sliding:
            self.head_dim = config.head_dim
            self.n_kv_heads = config.num_key_value_heads
            self.use_k_eq_v = False
        else:
            self.head_dim = (
                config.global_head_dim if config.global_head_dim else config.head_dim
            )
            self.use_k_eq_v = config.attention_k_eq_v
            self.n_kv_heads = (
                config.num_global_key_value_heads
                if self.use_k_eq_v
                else config.num_key_value_heads
            )

        self.n_heads = config.num_attention_heads
        self.n_kv_groups = self.n_heads // self.n_kv_heads
        # Gemma 4 uses unit scaling; q/k norms handle normalization
        self.scale = 1.0

        # Projections
        self.q_proj = nn.Linear(
            config.hidden_size, self.n_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.n_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        if not self.use_k_eq_v:
            self.v_proj = nn.Linear(
                config.hidden_size,
                self.n_kv_heads * self.head_dim,
                bias=config.attention_bias,
            )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

        # Norms
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = RMSNoScale(eps=config.rms_norm_eps)

        # RoPE: parameters differ per layer type
        layer_type = config.layer_types[layer_idx]
        rope_params = config.rope_parameters.get(layer_type, {})
        rope_theta = rope_params.get("rope_theta", 10_000.0)
        partial_rotary_factor = rope_params.get("partial_rotary_factor", 1.0)
        rotary_dims = int(partial_rotary_factor * self.head_dim)
        self.rope = nn.RoPE(rotary_dims, traditional=False, base=rope_theta)

        # KV sharing: layers beyond this index reuse cached KV from earlier layers
        first_kv_shared = config.num_hidden_layers - config.num_kv_shared_layers
        self.is_kv_shared_layer = layer_idx >= first_kv_shared > 0

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        queries = self.q_norm(queries)

        offset = 0
        if self.is_kv_shared_layer and cache is not None:
            # Reuse KV from the designated non-shared layer's cache
            keys, values = cache.state
            offset = cache.offset
        else:
            if cache is not None:
                offset = cache.offset

            # Raw key projection (shared with value when k_eq_v is enabled)
            keys_raw = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)

            if self.use_k_eq_v:
                values_raw = keys_raw
            else:
                values_raw = self.v_proj(x).reshape(
                    B, L, self.n_kv_heads, self.head_dim
                )

            # Apply k_norm + RoPE to keys
            keys = self.k_norm(keys_raw)
            keys = keys.transpose(0, 2, 1, 3)
            keys = self.rope(keys, offset=offset)

            # Apply v_norm to values (no RoPE, no learnable scale)
            values = self.v_norm(values_raw)
            values = values.transpose(0, 2, 1, 3)

            if cache is not None:
                keys, values = cache.update_and_fetch(keys, values)

        queries = queries.transpose(0, 2, 1, 3)
        queries = self.rope(queries, offset=offset)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        first_kv_shared = config.num_hidden_layers - config.num_kv_shared_layers
        is_kv_shared = layer_idx >= first_kv_shared > 0
        use_double_wide = config.use_double_wide_mlp and is_kv_shared

        dim = config.hidden_size
        hidden_dim = config.intermediate_size * (2 if use_double_wide else 1)

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# MoE (Mixture of Experts) — optional, used in larger Gemma 4 variants
# ---------------------------------------------------------------------------


class Router(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.scalar_root_size = config.hidden_size**-0.5
        self.top_k = config.top_k_experts

        self.norm = RMSNoScale(eps=config.rms_norm_eps)
        self.proj = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.scale = mx.ones((config.hidden_size,))
        self.per_expert_scale = mx.ones((config.num_experts,))

    def __call__(self, x: mx.array):
        x = self.norm(x)
        x = x * self.scale * self.scalar_root_size
        scores = self.proj(x)
        probs = mx.softmax(scores, axis=-1)

        # Top-k selection
        top_k_indices = mx.argpartition(-probs, kth=self.top_k - 1, axis=-1)[
            ..., : self.top_k
        ]
        top_k_weights = mx.take_along_axis(probs, top_k_indices, axis=-1)

        # Normalize and apply per-expert scale
        top_k_weights = top_k_weights / mx.sum(
            top_k_weights, axis=-1, keepdims=True
        )
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_indices]

        return top_k_weights, top_k_indices


class Experts(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size

        # 3D parameter tensors: [num_experts, out_features, in_features]
        self.gate_up_proj = mx.zeros(
            (self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
        )
        self.down_proj = mx.zeros(
            (self.num_experts, self.hidden_dim, self.intermediate_dim)
        )

    def __call__(
        self,
        x: mx.array,
        top_k_indices: mx.array,
        top_k_weights: mx.array,
    ) -> mx.array:
        # x: [tokens, hidden], top_k_indices/weights: [tokens, k]
        final = mx.zeros_like(x)

        for expert_idx in range(self.num_experts):
            expert_mask = top_k_indices == expert_idx  # [tokens, k]
            token_mask = mx.any(expert_mask, axis=-1)  # [tokens]

            if not mx.any(token_mask):
                continue

            token_indices = mx.argwhere(token_mask).squeeze(-1)
            if token_indices.size == 0:
                continue

            current = x[token_indices]  # [n, hidden]

            # Fused gate + up projection
            gate_up = current @ self.gate_up_proj[expert_idx].T
            gate, up = mx.split(gate_up, 2, axis=-1)
            hidden = nn.gelu_approx(gate) * up
            hidden = hidden @ self.down_proj[expert_idx].T

            # Apply router weight
            weights_for_tokens = mx.where(
                expert_mask[token_indices], top_k_weights[token_indices], 0.0
            )
            weight = mx.sum(weights_for_tokens, axis=-1, keepdims=True)
            hidden = hidden * weight

            final[token_indices] = final[token_indices] + hidden

        return final


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------


@partial(mx.compile, shapeless=True)
def logit_softcap(cap, x):
    return mx.tanh(x / cap) * cap


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.self_attn = Attention(config, layer_idx)
        self.mlp = MLP(config, layer_idx)

        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # Per-layer scalar (loaded from checkpoint, initialized to 1.0)
        self.layer_scalar = mx.ones((1,))

        # Per-layer input gating (PLE)
        self.has_per_layer_input = config.hidden_size_per_layer_input > 0
        if self.has_per_layer_input:
            self.per_layer_input_gate = nn.Linear(
                config.hidden_size, config.hidden_size_per_layer_input, bias=False
            )
            self.per_layer_projection = nn.Linear(
                config.hidden_size_per_layer_input, config.hidden_size, bias=False
            )
            self.post_per_layer_input_norm = nn.RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )

        # MoE (optional, only in larger variants)
        self.enable_moe = config.enable_moe_block
        if self.enable_moe:
            self.router = Router(config)
            self.experts = Experts(config)
            self.post_feedforward_layernorm_1 = nn.RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.post_feedforward_layernorm_2 = nn.RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.pre_feedforward_layernorm_2 = nn.RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        per_layer_input: Optional[mx.array] = None,
    ) -> mx.array:
        # Self-attention block
        residual = x
        h = self.input_layernorm(x)
        h = self.self_attn(h, mask, cache)
        h = self.post_attention_layernorm(h)
        x = residual + h

        # Feed-forward block (dense MLP + optional MoE)
        residual = x
        h = self.pre_feedforward_layernorm(x)
        h = self.mlp(h)

        if self.enable_moe:
            h_mlp = self.post_feedforward_layernorm_1(h)

            # MoE branch operates on the pre-MLP residual
            flat = residual.reshape(-1, residual.shape[-1])
            top_k_weights, top_k_indices = self.router(flat)
            h_moe = self.pre_feedforward_layernorm_2(flat)
            h_moe = self.experts(h_moe, top_k_indices, top_k_weights)
            h_moe = h_moe.reshape(residual.shape)
            h_moe = self.post_feedforward_layernorm_2(h_moe)

            # Combine dense MLP and MoE outputs
            h = h_mlp + h_moe

        h = self.post_feedforward_layernorm(h)
        x = residual + h

        # Per-layer input gating (PLE)
        if self.has_per_layer_input and per_layer_input is not None:
            residual = x
            g = self.per_layer_input_gate(x)
            g = nn.gelu_approx(g)
            g = g * per_layer_input
            g = self.per_layer_projection(g)
            g = self.post_per_layer_input_norm(g)
            x = residual + g

        x = x * self.layer_scalar
        return x


# ---------------------------------------------------------------------------
# Full Text Model
# ---------------------------------------------------------------------------


class Gemma4TextModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.vocab_size = config.vocab_size
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        first_kv_shared = config.num_hidden_layers - config.num_kv_shared_layers
        self.first_kv_shared_layer_idx = first_kv_shared

        self.layers = [
            TransformerBlock(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Per-layer input embeddings (PLE)
        if self.hidden_size_per_layer_input > 0:
            self.embed_tokens_per_layer = nn.Embedding(
                config.vocab_size_per_layer_input,
                config.num_hidden_layers * config.hidden_size_per_layer_input,
            )
            self.per_layer_model_projection = nn.Linear(
                config.hidden_size,
                config.num_hidden_layers * config.hidden_size_per_layer_input,
                bias=False,
            )
            self.per_layer_projection_norm = nn.RMSNorm(
                config.hidden_size_per_layer_input, eps=config.rms_norm_eps
            )

        # Build cache-index mapping for KV sharing
        concrete_layers = config.layer_types[:first_kv_shared]
        self.layer_idx_to_cache_idx = []

        if config.num_kv_shared_layers > 0 and first_kv_shared < config.num_hidden_layers:
            shared_map = {}
            for lt in set(config.layer_types):
                if lt in concrete_layers:
                    shared_map[lt] = (
                        len(concrete_layers) - 1 - concrete_layers[::-1].index(lt)
                    )
            for i, lt in enumerate(config.layer_types):
                if i < first_kv_shared:
                    self.layer_idx_to_cache_idx.append(i)
                else:
                    self.layer_idx_to_cache_idx.append(shared_map.get(lt, i))
        else:
            self.layer_idx_to_cache_idx = list(range(config.num_hidden_layers))

        # Representative layer indices for mask creation
        self.first_sliding_idx = next(
            (
                self.layer_idx_to_cache_idx[i]
                for i, lt in enumerate(config.layer_types)
                if lt == "sliding_attention"
            ),
            0,
        )
        self.first_full_idx = next(
            (
                self.layer_idx_to_cache_idx[i]
                for i, lt in enumerate(config.layer_types)
                if lt == "full_attention"
            ),
            0,
        )

    def get_per_layer_inputs(self, input_ids: mx.array) -> mx.array:
        """Compute per-layer token-identity embeddings, scaled by sqrt(ple_dim)."""
        if self.hidden_size_per_layer_input <= 0:
            return None
        mask = input_ids < self.config.vocab_size_per_layer_input
        tokens = mx.where(mask, input_ids, mx.zeros_like(input_ids))
        result = self.embed_tokens_per_layer(tokens) * (
            self.hidden_size_per_layer_input**0.5
        )
        return result.reshape(
            *input_ids.shape,
            self.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )

    def project_per_layer_inputs(
        self, inputs_embeds: mx.array, per_layer_inputs: mx.array
    ) -> mx.array:
        """Compute context-aware projection and combine with token-identity PLE."""
        per_layer_proj = self.per_layer_model_projection(inputs_embeds) * (
            self.hidden_size**-0.5
        )
        per_layer_proj = per_layer_proj.reshape(
            *inputs_embeds.shape[:-1],
            self.config.num_hidden_layers,
            self.config.hidden_size_per_layer_input,
        )
        per_layer_proj = self.per_layer_projection_norm(per_layer_proj)
        if per_layer_inputs is None:
            return per_layer_proj
        return (per_layer_proj + per_layer_inputs) * (2.0**-0.5)

    def __call__(
        self,
        inputs: mx.array = None,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)

        # Scale embeddings (bfloat16 cast matches HF numerical behavior)
        h = h * mx.array(self.hidden_size**0.5, dtype=mx.bfloat16).astype(h.dtype)

        # Per-layer inputs (PLE)
        per_layer_inputs = None
        if self.hidden_size_per_layer_input > 0 and inputs is not None:
            per_layer_inputs = self.get_per_layer_inputs(inputs)
            per_layer_inputs = self.project_per_layer_inputs(h, per_layer_inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        # Create masks from representative cache entries
        full_cache = (
            cache[self.first_full_idx] if self.first_full_idx < len(cache) else None
        )
        sliding_cache = (
            cache[self.first_sliding_idx]
            if self.first_sliding_idx < len(cache)
            else None
        )
        global_mask = create_attention_mask(h, full_cache)
        sliding_mask = create_attention_mask(
            h, sliding_cache, window_size=self.config.sliding_window
        )

        for i, layer in enumerate(self.layers):
            is_global = self.config.layer_types[i] == "full_attention"
            mask = global_mask if is_global else sliding_mask

            cache_idx = self.layer_idx_to_cache_idx[i]
            c = cache[cache_idx] if cache_idx < len(cache) else None

            per_layer_input = (
                per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
            )
            h = layer(h, mask, c, per_layer_input)

        return self.norm(h)


# ---------------------------------------------------------------------------
# Top-level Model (with LM head)
# ---------------------------------------------------------------------------


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Gemma4TextModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.tie_word_embeddings = args.tie_word_embeddings

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        out = self.model(inputs, cache, input_embeddings)
        if self.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        if self.args.final_logit_softcapping is not None:
            out = logit_softcap(self.args.final_logit_softcapping, out)
        return out

    def sanitize(self, weights):
        if "lm_head.weight" not in weights:
            self.tie_word_embeddings = True
            if hasattr(self, "lm_head"):
                delattr(self, "lm_head")

        # Remove non-persistent buffers and multimodal keys
        keys_to_remove = [
            k
            for k in weights
            if "embed_scale" in k
            or any(
                p in k
                for p in [
                    "vision_tower.",
                    "audio_tower.",
                    "embed_vision.",
                    "embed_audio.",
                ]
            )
        ]
        for k in keys_to_remove:
            del weights[k]
        return weights

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        config = self.args
        first_kv_shared = config.num_hidden_layers - config.num_kv_shared_layers
        caches = []

        if config.num_kv_shared_layers > 0:
            # Only create caches for non-shared layers
            for lt in config.layer_types[:first_kv_shared]:
                if lt == "full_attention":
                    caches.append(KVCache())
                else:
                    caches.append(
                        RotatingKVCache(max_size=config.sliding_window, keep=0)
                    )
        else:
            # No KV sharing: one cache per layer
            for lt in config.layer_types:
                if lt == "full_attention":
                    caches.append(KVCache())
                else:
                    caches.append(
                        RotatingKVCache(max_size=config.sliding_window, keep=0)
                    )
        return caches
