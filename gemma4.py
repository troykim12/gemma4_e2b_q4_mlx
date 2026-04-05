# Copyright © 2025 Apple Inc.
# Gemma 4 multimodal wrapper for mlx-lm (text-only inference)
# Strips vision/audio towers and delegates to the text model.

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from . import gemma4_text
from .base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "gemma4"
    text_config: dict = None
    vision_config: dict = None
    audio_config: dict = None
    vocab_size: int = 262144

    def __post_init__(self):
        if self.text_config is None:
            self.text_config = {}
        # Propagate top-level vocab_size into text_config if not already set
        if "vocab_size" not in self.text_config:
            self.text_config["vocab_size"] = self.vocab_size


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.language_model = gemma4_text.Model(
            gemma4_text.ModelArgs.from_dict(args.text_config)
        )

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        return self.language_model(
            inputs, cache=cache, input_embeddings=input_embeddings
        )

    def sanitize(self, weights):
        weights = tree_unflatten(list(weights.items()))

        # Remove multimodal components
        for k in ["vision_tower", "audio_tower", "embed_vision", "embed_audio"]:
            weights.pop(k, None)
            if "model" in weights and isinstance(weights["model"], dict):
                weights["model"].pop(k, None)

        flat = dict(tree_flatten(weights))

        # Promote "model.language_model.*" -> "language_model.*"
        promoted = {}
        for k, v in flat.items():
            if k.startswith("model.language_model."):
                promoted[k.replace("model.language_model.", "language_model.", 1)] = v
            else:
                promoted[k] = v

        # Delegate to the text model's sanitize for tied weights, etc.
        lm_weights = {
            k.replace("language_model.", "", 1): v
            for k, v in promoted.items()
            if k.startswith("language_model.")
        }
        if lm_weights:
            lm_weights = self.language_model.sanitize(lm_weights)
            promoted = {
                k: v
                for k, v in promoted.items()
                if not k.startswith("language_model.")
            }
            for k, v in lm_weights.items():
                promoted[f"language_model.{k}"] = v

        return promoted

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()
