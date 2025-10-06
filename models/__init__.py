from __future__ import annotations

from typing import Any, Dict, Type
from importlib import import_module
from torch import nn
from model_specs import load_model_spec


def get_model_class(model_class: str) -> Type[nn.Module]:
    """Import and return a model class given its fully-qualified class name.
    Example: 'models.gpt2.gpt_core.GPT2Core'. No defaults or fallbacks.
    """
    if not model_class or not isinstance(model_class, str):
        raise ValueError("model_class must be a non-empty fully-qualified class name string")
    if "." not in model_class:
        raise ValueError("model_class must be a fully-qualified class name like 'models.gpt2.gpt_core.GPT2Core'")
    module_path, cls_name = model_class.rsplit(".", 1)
    mod = import_module(module_path)
    cls = getattr(mod, cls_name, None)
    if cls is None:
        raise ImportError(f"Class '{cls_name}' not found in module '{module_path}' for model_class '{model_class}'")
    return cls


def model_from_spec(spec_name_or_path: str | dict, device: str = "cuda") -> nn.Module:
    spec: Dict[str, Any]
    if isinstance(spec_name_or_path, dict):
        spec = spec_name_or_path
    else:
        spec = load_model_spec(spec_name_or_path)

    # Required fields
    model_class = str(spec["model_class"])  # must be present
    vocab_size = int(spec["vocab_size"])    # required
    eos_token_id = int(spec["eos_token_id"])  # required
    num_layers = int(spec["num_layers"])    # required
    num_heads = int(spec["num_heads"])      # required
    model_dim = int(spec["model_dim"])      # required
    head_dim = int(spec["head_dim"])        # required
    window_block_size = int(spec["window_block_size"])  # required


    # If a single explicit max_seq_len is provided, allow it; otherwise require both new fields.
    if "max_seq_len" in spec:
        max_seq_len = int(spec["max_seq_len"])
        # Determine max sequence length using the same logic as train/sample (must be derivable, no legacy fallbacks)
        if "training_sequence_length" in spec and "val_seq_len" in spec:
            assert max_seq_len >= int(max(int(spec["training_sequence_length"]), int(spec["val_seq_len"])))

    else:
        raise ValueError("model spec must include training_sequence_length and val_seq_len or an explicit max_seq_len")

    ModelClass = get_model_class(model_class)
    model: nn.Module = ModelClass(
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_heads=num_heads,
        model_dim=model_dim,
        max_seq_len=max_seq_len,
        head_dim=head_dim,
        window_block_size=window_block_size,
        eos_token_id=eos_token_id,
    ).to(device)
    return model


__all__ = [
    "get_model_class",
    "model_from_spec",
]
