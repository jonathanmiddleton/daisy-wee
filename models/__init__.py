from __future__ import annotations

from typing import Dict, Type, Any
from torch import nn
from model_specs import load_model_spec

# Prefer the path under models.gpt2
from .gpt2.gpt_core import GPT2Core

# Registry mapping model_type keys in configs/checkpoints to concrete model classes
MODEL_REGISTRY: Dict[str, Type[GPT2Core]] = {
    "gpt2": GPT2Core,
}

def get_model_class(model_type: str):
    """Return the model class for a given model_type key.
    Defaults to GPT2Core when the key is unknown or empty for backward compatibility.
    """
    if not model_type:
        return GPT2Core
    return MODEL_REGISTRY.get(model_type.lower(), GPT2Core)


def model_from_spec(spec_name_or_path: str | dict, device: str = "cuda") -> nn.Module:
    spec = {}
    if isinstance(spec_name_or_path, dict):
        spec = spec_name_or_path
    else:
        spec: Dict[str, Any] = load_model_spec(spec_name_or_path)

    # Pull required fields (ensure your spec contains these)
    vocab_size = int(spec["vocab_size"])               # required
    num_layers = int(spec["num_layers"])               # required
    num_heads = int(spec["num_heads"])                 # required
    model_dim = int(spec["model_dim"])                 # required
    head_dim = int(spec["head_dim"])                   # required
    window_block_size = int(spec.get("window_block_size", 128))

    # Determine max sequence length using the same logic as train/sample
    if "training_sequence_length" in spec and "val_seq_len" in spec:
        max_seq_len = int(max(int(spec["training_sequence_length"]), int(spec["val_seq_len"])))
    else:
        # Fallback if your spec has an older `max_seq_len` key
        max_seq_len = int(spec.get("max_seq_len", 16 * 1024))

    model_type = str(spec.get("model_type", "gpt2")).lower()
    ModelClass = get_model_class(model_type)

    model: nn.Module = ModelClass(
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_heads=num_heads,
        model_dim=model_dim,
        max_seq_len=max_seq_len,
        head_dim=head_dim,
        window_block_size=window_block_size,
    ).to(device)

    return model


__all__ = [
    "get_model_class",
    "model_from_spec",
]
