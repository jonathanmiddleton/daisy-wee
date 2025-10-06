from __future__ import annotations

from typing import Any, Dict, Type
from importlib import import_module
from dataclasses import is_dataclass, asdict as dc_asdict, fields as dc_fields
from torch import nn
from model_specs import load_model_spec, ModelSpec


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


def model_from_spec(spec_or_cfg: str | dict | ModelSpec | Any, device: str = "cuda") -> nn.Module:
    # Normalize to ModelSpec for validation of architecture fields
    spec: ModelSpec
    aux_cfg: Dict[str, Any] = {}

    if isinstance(spec_or_cfg, str):
        # YAML -> ModelSpec (load_model_spec already validates)
        spec = load_model_spec(spec_or_cfg)
    elif isinstance(spec_or_cfg, ModelSpec):
        spec = spec_or_cfg
    elif isinstance(spec_or_cfg, dict):
        cfg = dict(spec_or_cfg)
        # Keep extras for aux, but filter to ModelSpec fields for construction
        aux_cfg = cfg
        allowed = {f.name for f in dc_fields(ModelSpec)}
        spec_data = {k: v for k, v in cfg.items() if k in allowed}
        spec = ModelSpec(**spec_data)
    elif is_dataclass(spec_or_cfg):
        cfg = dc_asdict(spec_or_cfg)
        aux_cfg = cfg
        allowed = {f.name for f in dc_fields(ModelSpec)}
        spec_data = {k: v for k, v in cfg.items() if k in allowed}
        spec = ModelSpec(**spec_data)
    else:
        # Fallback: read attributes best-effort
        keys = {f.name for f in dc_fields(ModelSpec)} | {"training_sequence_length", "val_seq_len", "window_block_size", "max_seq_len"}
        cfg = {k: getattr(spec_or_cfg, k) for k in keys if hasattr(spec_or_cfg, k)}
        aux_cfg = cfg
        allowed = {f.name for f in dc_fields(ModelSpec)}
        spec_data = {k: v for k, v in cfg.items() if k in allowed}
        spec = ModelSpec(**spec_data)

    # Pull fields from validated ModelSpec
    model_class = str(spec.model_class)
    vocab_size = int(spec.vocab_size)
    eos_token_id = int(spec.eos_token_id)
    num_layers = int(spec.num_layers)
    num_heads = int(spec.num_heads)
    model_dim = int(spec.model_dim)
    head_dim = int(spec.head_dim)

    # window_block_size is a training/runtime setting; prefer from aux cfg, else default to 128
    window_block_size = int((aux_cfg.get("window_block_size", 128) if isinstance(aux_cfg, dict) else 128) or 128)

    # Determine max_seq_len from available training settings
    if isinstance(aux_cfg, dict) and "max_seq_len" in aux_cfg:
        max_seq_len = int(aux_cfg["max_seq_len"])  # explicit
    else:
        tsl = int((aux_cfg.get("training_sequence_length", 0) if isinstance(aux_cfg, dict) else 0) or 0)
        # prefer val_seq_len from ModelSpec if provided; else aux
        vsl = spec.val_seq_len if spec.val_seq_len is not None else int((aux_cfg.get("val_seq_len", 0) if isinstance(aux_cfg, dict) else 0) or 0)
        if tsl <= 0 and (not vsl or int(vsl) <= 0):
            raise ValueError("Cannot determine max_seq_len: provide max_seq_len, or training_sequence_length/val_seq_len in cfg or spec")
        max_seq_len = max(int(tsl), int(vsl or 0))

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
