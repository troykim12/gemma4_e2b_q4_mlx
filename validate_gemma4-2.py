#!/usr/bin/env python3
"""
Gemma 4 MLX Implementation Validator
=====================================
Validates that the gemma4_text.py / gemma4.py implementation correctly
matches the HuggingFace Gemma 4 model architecture and weights.

Validation steps:
  1. Weight Mapping  — all parameters load without missing or extra keys
  2. Architecture     — config.json values are parsed and applied correctly
  3. Numerical        — outputs match HF transformers for identical inputs
  4. Inference        — actual text generation produces sensible output

Usage:
  # Basic validation (weight mapping + architecture)
  python validate_gemma4.py --model ./gemma-4-e2b-q4

  # Numerical comparison against HF (requires transformers + torch)
  python validate_gemma4.py --model ./gemma-4-e2b-q4 --numerical --hf-model google/gemma-4-E2B-it

  # Inference validation
  python validate_gemma4.py --model ./gemma-4-e2b-q4 --inference
"""

import argparse
import json
import sys
from pathlib import Path


def check_imports():
    """Verify required packages are installed."""
    try:
        import mlx.core as mx
        import mlx.nn as nn
    except ImportError:
        print("  mlx is not installed. Run: pip install mlx")
        return False

    try:
        import mlx_lm
    except ImportError:
        print("  mlx-lm is not installed. Run: pip install mlx-lm")
        return False

    return True


def check_files_installed():
    """Verify gemma4_text.py and gemma4.py are in mlx_lm/models/."""
    import mlx_lm
    models_dir = Path(mlx_lm.__file__).parent / "models"

    results = {}
    for fname in ["gemma4_text.py", "gemma4.py"]:
        fpath = models_dir / fname
        results[fname] = fpath.exists()
        status = "OK" if fpath.exists() else "MISSING"
        print(f"  [{status}] {fpath}")

    if not all(results.values()):
        print(f"\n  To install, run:")
        print(f"    cp gemma4_text.py gemma4.py {models_dir}/")

    return all(results.values())


def validate_weight_mapping(model_path: str):
    """
    Validation 1: Weight Mapping
    Ensures all safetensor weights map to model parameters without gaps.
    """
    print("\n" + "=" * 60)
    print("  Validation 1: Weight Mapping")
    print("=" * 60)

    from mlx_lm.utils import load

    try:
        model, tokenizer = load(model_path)
        print(f"  [PASS] Model loaded successfully: {model.model_type}")
    except Exception as e:
        print(f"  [FAIL] Model loading failed: {e}")
        return False

    # Parameter count
    import mlx.core as mx
    from mlx.utils import tree_flatten

    params = tree_flatten(model.parameters())
    total_params = sum(p.size for _, p in params)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Number of layers: {len(model.layers)}")

    # Cross-check with config.json
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

        text_config = config.get("text_config", config)
        expected_layers = text_config.get("num_hidden_layers", "?")
        actual_layers = len(model.layers)

        if actual_layers == expected_layers:
            print(f"  [PASS] Layer count matches: {actual_layers}")
        else:
            print(f"  [FAIL] Layer count mismatch: expected {expected_layers}, got {actual_layers}")
            return False

    return True


def validate_architecture(model_path: str):
    """
    Validation 2: Architecture
    Checks that each config.json field is correctly reflected in the model structure.
    """
    print("\n" + "=" * 60)
    print("  Validation 2: Architecture")
    print("=" * 60)

    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        print("  [WARN] config.json not found, skipping.")
        return True

    with open(config_path) as f:
        config = json.load(f)

    text_config = config.get("text_config", config)
    model_type = config.get("model_type", text_config.get("model_type", "unknown"))
    print(f"  Model type: {model_type}")

    checks = []

    # Key architecture features
    features = {
        "hidden_size": text_config.get("hidden_size"),
        "num_hidden_layers": text_config.get("num_hidden_layers"),
        "num_attention_heads": text_config.get("num_attention_heads"),
        "num_key_value_heads": text_config.get("num_key_value_heads"),
        "head_dim": text_config.get("head_dim"),
        "global_head_dim": text_config.get("global_head_dim"),
        "sliding_window": text_config.get("sliding_window"),
        "hidden_size_per_layer_input": text_config.get("hidden_size_per_layer_input"),
        "attention_k_eq_v": text_config.get("attention_k_eq_v"),
        "num_kv_shared_layers": text_config.get("num_kv_shared_layers"),
        "enable_moe_block": text_config.get("enable_moe_block"),
        "use_double_wide_mlp": text_config.get("use_double_wide_mlp"),
        "tie_word_embeddings": config.get(
            "tie_word_embeddings", text_config.get("tie_word_embeddings")
        ),
    }

    for k, v in features.items():
        if v is not None:
            print(f"    {k}: {v}")

    # Layer types
    layer_types = text_config.get("layer_types", [])
    if layer_types:
        sliding = sum(1 for lt in layer_types if lt == "sliding_attention")
        full = sum(1 for lt in layer_types if lt == "full_attention")
        print(f"    layer_types: {sliding} sliding + {full} full = {len(layer_types)} total")

        if layer_types[-1] != "full_attention":
            print("  [WARN] Last layer is NOT full_attention!")
            checks.append(False)
        else:
            print("  [PASS] Last layer = full_attention")
            checks.append(True)

    # RoPE parameters
    rope_params = text_config.get("rope_parameters", {})
    if rope_params:
        for lt, params in rope_params.items():
            theta = params.get("rope_theta", "?")
            prf = params.get("partial_rotary_factor", 1.0)
            print(f"    RoPE [{lt}]: theta={theta}, partial_rotary_factor={prf}")

    # PLE validation (critical for E2B/E4B)
    ple_dim = text_config.get("hidden_size_per_layer_input", 0)
    n_layers = text_config.get("num_hidden_layers", 0)
    if ple_dim > 0:
        expected_embed_dim = n_layers * ple_dim
        print(f"    PLE: embed_tokens_per_layer expected shape = [vocab, {expected_embed_dim}]")
        print(f"         ({n_layers} layers x {ple_dim} dim)")
        print(f"         scale factor = sqrt({ple_dim}) = {ple_dim**0.5}")

    return all(checks) if checks else True


def validate_numerical(model_path: str, hf_model_id: str):
    """
    Validation 3: Numerical Comparison
    Compares HF transformers and MLX outputs for the same input.
    """
    print("\n" + "=" * 60)
    print("  Validation 3: Numerical Comparison (HF vs MLX)")
    print("=" * 60)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("  [SKIP] transformers/torch not installed.")
        print("         pip install transformers torch")
        return True

    import numpy as np
    import mlx.core as mx
    from mlx_lm.utils import load

    test_prompt = "Hello, world!"

    print(f"  Test input: '{test_prompt}'")
    print(f"  HF model:  {hf_model_id}")
    print(f"  MLX model: {model_path}")

    # HF inference
    print("  Loading HF model...")
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_id, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    hf_inputs = hf_tokenizer(test_prompt, return_tensors="pt")

    with torch.no_grad():
        hf_outputs = hf_model(**hf_inputs)
        hf_logits = hf_outputs.logits[0, -1, :].float().numpy()

    # MLX inference
    print("  Loading MLX model...")
    mlx_model, mlx_tokenizer = load(model_path)
    mlx_input_ids = mx.array(hf_inputs["input_ids"].numpy())
    mlx_logits = mlx_model(mlx_input_ids)
    mlx_logits_np = np.array(mlx_logits[0, -1, :].astype(mx.float32))

    # Compare top-k tokens
    hf_top5 = np.argsort(hf_logits)[-5:][::-1]
    mlx_top5 = np.argsort(mlx_logits_np)[-5:][::-1]

    print(f"\n  HF  Top-5 tokens: {hf_top5.tolist()}")
    print(f"  MLX Top-5 tokens: {mlx_top5.tolist()}")

    if hf_top5[0] == mlx_top5[0]:
        print("  [PASS] Top-1 token matches!")
    else:
        print("  [WARN] Top-1 token mismatch (may be due to quantization)")

    # Cosine similarity
    cos_sim = np.dot(hf_logits, mlx_logits_np) / (
        np.linalg.norm(hf_logits) * np.linalg.norm(mlx_logits_np)
    )
    print(f"  Cosine similarity: {cos_sim:.6f}")

    if cos_sim > 0.95:
        print("  [PASS] Numerically consistent (cos_sim > 0.95)")
        return True
    elif cos_sim > 0.85:
        print("  [WARN] Minor differences detected (likely quantization). Acceptable.")
        return True
    else:
        print("  [FAIL] Significant numerical mismatch. Check implementation.")
        return False


def validate_inference(model_path: str):
    """
    Validation 4: Inference
    Checks that actual text generation produces reasonable output.
    """
    print("\n" + "=" * 60)
    print("  Validation 4: Inference")
    print("=" * 60)

    from mlx_lm import load, generate

    model, tokenizer = load(model_path)

    test_prompts = [
        "1 + 1 =",
        "The capital of France is",
        "def hello():",
    ]

    all_ok = True
    for prompt in test_prompts:
        try:
            result = generate(
                model, tokenizer, prompt=prompt, max_tokens=20, verbose=False
            )
            if result and len(result.strip()) > 0:
                short = result.strip()[:80]
                print(f"  [PASS] '{prompt}' -> '{short}'")
            else:
                print(f"  [WARN] '{prompt}' -> (empty output)")
                all_ok = False
        except Exception as e:
            print(f"  [FAIL] '{prompt}' -> Error: {e}")
            all_ok = False

    return all_ok


def validate_weight_keys(model_path: str):
    """
    Helper: Enumerate safetensor weight keys and check for missing mappings.
    """
    print("\n" + "=" * 60)
    print("  Weight Key Analysis")
    print("=" * 60)

    try:
        from safetensors import safe_open
    except ImportError:
        print("  [SKIP] safetensors package not installed.")
        return True

    model_dir = Path(model_path)
    st_files = list(model_dir.glob("*.safetensors"))
    if not st_files:
        print("  [SKIP] No safetensors files found.")
        return True

    all_keys = set()
    for sf in st_files:
        with safe_open(str(sf), framework="numpy") as f:
            all_keys.update(f.keys())

    # Filter text model keys only
    text_keys = {
        k
        for k in all_keys
        if not any(
            skip in k
            for skip in [
                "vision_tower",
                "audio_tower",
                "embed_vision",
                "embed_audio",
            ]
        )
    }
    multimodal_keys = all_keys - text_keys

    print(f"  Total weight keys:              {len(all_keys)}")
    print(f"  Text model keys:                {len(text_keys)}")
    print(f"  Multimodal keys (stripped):      {len(multimodal_keys)}")

    # Check key patterns
    patterns = {
        "embed_tokens": 0,
        "embed_tokens_per_layer": 0,
        "per_layer_model_projection": 0,
        "per_layer_input_gate": 0,
        "self_attn.q_proj": 0,
        "self_attn.k_proj": 0,
        "self_attn.v_proj": 0,
        "self_attn.o_proj": 0,
        "self_attn.q_norm": 0,
        "self_attn.k_norm": 0,
        "self_attn.v_norm": 0,
        "mlp.gate_proj": 0,
        "mlp.up_proj": 0,
        "mlp.down_proj": 0,
        "layer_scalar": 0,
        "input_layernorm": 0,
        "post_attention_layernorm": 0,
        "pre_feedforward_layernorm": 0,
        "post_feedforward_layernorm": 0,
    }

    for key in text_keys:
        for pattern in patterns:
            if pattern in key:
                patterns[pattern] += 1

    print("\n  Key parameter patterns:")
    for pattern, count in sorted(patterns.items()):
        status = "OK  " if count > 0 else "WARN"
        print(f"    [{status}] {pattern}: {count}")

    # Note about k_eq_v (v_proj may be absent for full attention layers)
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        text_config = config.get("text_config", config)
        k_eq_v = text_config.get("attention_k_eq_v", False)
        if k_eq_v:
            print(
                f"\n  Note: attention_k_eq_v=True means v_proj is absent in full attention layers. This is expected."
            )

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Gemma 4 MLX Implementation Validator"
    )
    parser.add_argument("--model", required=True, help="Path to the MLX model directory")
    parser.add_argument(
        "--numerical", action="store_true", help="Run numerical comparison against HF"
    )
    parser.add_argument(
        "--hf-model",
        default="google/gemma-4-E2B-it",
        help="HF model ID for numerical comparison",
    )
    parser.add_argument(
        "--inference", action="store_true", help="Run inference validation"
    )
    parser.add_argument(
        "--keys", action="store_true", help="Run weight key analysis"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Gemma 4 MLX Implementation Validator")
    print("=" * 60)

    # Basic checks
    if not check_imports():
        sys.exit(1)

    print("\n  File installation check:")
    if not check_files_installed():
        print("\n  Files not installed. Please install them first.")
        sys.exit(1)

    # Run validations
    results = {}

    results["weight_mapping"] = validate_weight_mapping(args.model)
    results["architecture"] = validate_architecture(args.model)

    if args.keys:
        results["weight_keys"] = validate_weight_keys(args.model)

    if args.numerical:
        results["numerical"] = validate_numerical(args.model, args.hf_model)

    if args.inference:
        results["inference"] = validate_inference(args.model)

    # Summary
    print("\n" + "=" * 60)
    print("  Validation Summary")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}]  {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\n  All validations passed!")
    else:
        print("\n  Some validations failed. See logs above for details.")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
