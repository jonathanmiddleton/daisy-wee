import dataclasses
from dataclasses import dataclass, fields as dataclass_fields
from pathlib import Path
from typing import Any

import yaml

@dataclass(frozen=True)
class ModelSpec:
    """
    Strict schema for model_specs/*.yml files.
    Only architecture-relevant fields live here. Training config belongs in the training YAML.
    Note: window_block_size is part of the model/runtime contract and is specified in the ModelSpec.
    """
    model_class: str
    vocab_size: int
    num_layers: int
    num_heads: int
    model_dim: int
    head_dim: int
    eos_token_id: int
    window_block_size: int  # block granularity for sliding window/masks
    attention_window_len: int  # largest sliding attention window supported by the model
    max_seq_len: int  # maximum context size supported by the model


def _strict_keys(obj: dict[str, Any], allowed: set[str], ctx: str) -> None:
    unknown = set(obj) - allowed
    if unknown:
        raise ValueError(f"Unknown key(s) in {ctx}: {sorted(unknown)}")

def _require_keys(obj: dict[str, Any], required: set[str], ctx: str) -> None:
    missing = [k for k in required if k not in obj]
    if missing:
        raise ValueError(f"Missing required key(s) in {ctx}: {sorted(missing)}")

def _as_int(name: str, value: Any, ctx: str, min_value: int | None = 1) -> int:
    try:
        iv = int(value)
    except Exception:
        raise ValueError(f"{ctx}.{name} must be an integer; got {value!r}")
    if min_value is not None and iv < min_value:
        raise ValueError(f"{ctx}.{name} must be >= {min_value}; got {iv}")
    return iv

def _as_bool(name: str, value: Any, ctx: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, str)):
        v = str(value).lower()
        if v in ("1", "true", "yes", "y"):
            return True
        if v in ("0", "false", "no", "n"):
            return False
    raise ValueError(f"{ctx}.{name} must be a boolean; got {value!r}")

def _as_float(name: str, value: Any, ctx: str, min_value: float | None = None, max_value: float | None = None) -> float:
    try:
        fv = float(value)
    except Exception:
        raise ValueError(f"{ctx}.{name} must be a float; got {value!r}")
    if min_value is not None and fv < min_value:
        raise ValueError(f"{ctx}.{name} must be >= {min_value}; got {fv}")
    if max_value is not None and fv > max_value:
        raise ValueError(f"{ctx}.{name} must be <= {max_value}; got {fv}")
    return fv

def load_model_spec(name_or_path: str) -> dict[str, Any]:
    """
    Load a model spec YAML by name (model_specs/<name>.yml) or explicit file path.
    Enforces a strict schema (no extra keys, required keys present, type/value checks).
    Returns a dict (to be merged into training config where relevant).
    """
    # Resolve path
    p = Path(name_or_path)
    if not p.exists():
        p = Path("") / (name_or_path if name_or_path.endswith(".yml") else f"{name_or_path}.yml")
    if not p.exists():
        raise FileNotFoundError(f"Model spec not found: {name_or_path!r} (resolved: {p})")

    with open(p, "r") as f:
        spec = yaml.safe_load(f) or {}

    if not isinstance(spec, dict):
        raise ValueError(f"Model spec must be a mapping/dict. File: {p}")

    allowed = {f.name for f in dataclass_fields(ModelSpec)}
    required = {f.name for f in dataclass_fields(ModelSpec) if f.default is dataclasses.MISSING}
    _strict_keys(spec, allowed, f"model_spec:{p.name}")
    _require_keys(spec, required, f"model_spec:{p.name}")

    # Type and value checks
    ctx = f"model_spec:{p.name}"
    # ints
    for k in ("vocab_size", "num_layers", "num_heads", "model_dim", "head_dim", "eos_token_id", "window_block_size", "attention_window_len", "max_seq_len"):
        spec[k] = _as_int(k, spec[k], ctx, min_value=1)
    # strings
    if not isinstance(spec["model_class"], str) or not spec["model_class"]:
        raise ValueError(f"{ctx}.model_class must be a non-empty string")

    # Cross-field constraints
    if spec["attention_window_len"] % spec["window_block_size"] != 0:
        raise ValueError(f"{ctx}: attention_window_len ({spec['attention_window_len']}) must be divisible by window_block_size ({spec['window_block_size']})")
    if spec["max_seq_len"] < spec["attention_window_len"]:
        raise ValueError(f"{ctx}: max_seq_len ({spec['max_seq_len']}) must be >= attention_window_len ({spec['attention_window_len']})")

    return spec

def _validate_optimizers_schema(cfg_list: Any, ctx: str = "optimizers") -> list[dict[str, Any]]:
    if not isinstance(cfg_list, list) or not cfg_list:
        raise ValueError(f"{ctx} must be a non-empty list")

    out: list[dict[str, Any]] = []
    for i, item in enumerate(cfg_list):
        ictx = f"{ctx}[{i}]"
        if not isinstance(item, dict):
            raise ValueError(f"{ictx} must be a mapping/dict")
        allowed_top = {"type", "betas", "eps", "weight_decay", "fused", "momentum", "params"}
        required_top = {"type", "params"}
        _strict_keys(item, allowed_top, ictx)
        _require_keys(item, required_top, ictx)

        # Basic fields
        if not isinstance(item["type"], str) or not item["type"]:
            raise ValueError(f"{ictx}.type must be a non-empty string")

        if "betas" in item:
            betas = item["betas"]
            if (not isinstance(betas, (list, tuple))) or len(betas) != 2:
                raise ValueError(f"{ictx}.betas must be a 2-length list")
            b0 = _as_float("betas[0]", betas[0], ictx, min_value=0.0, max_value=0.9999999)
            b1 = _as_float("betas[1]", betas[1], ictx, min_value=0.0, max_value=0.9999999)
            item["betas"] = [b0, b1]

        if "eps" in item:
            item["eps"] = _as_float("eps", item["eps"], ictx, min_value=0.0)

        if "weight_decay" in item:
            item["weight_decay"] = _as_float("weight_decay", item["weight_decay"], ictx, min_value=0.0)

        if "momentum" in item:
            item["momentum"] = _as_float("momentum", item["momentum"], ictx, min_value=0.0, max_value=0.9999999)

        if "fused" in item:
            item["fused"] = _as_bool("fused", item["fused"], ictx)

        # params groups
        params = item["params"]
        if not isinstance(params, list) or not params:
            raise ValueError(f"{ictx}.params must be a non-empty list")
        validated_groups: list[dict[str, Any]] = []
        for j, g in enumerate(params):
            gctx = f"{ictx}.params[{j}]"
            if not isinstance(g, dict):
                raise ValueError(f"{gctx} must be a mapping/dict")
            allowed_g = {"group", "lr"}
            required_g = {"group", "lr"}
            _strict_keys(g, allowed_g, gctx)
            _require_keys(g, required_g, gctx)
            if not isinstance(g["group"], str) or not g["group"]:
                raise ValueError(f"{gctx}.group must be a non-empty string")
            g["lr"] = _as_float("lr", g["lr"], gctx, min_value=0.0)
            validated_groups.append({"group": g["group"], "lr": g["lr"]})
        item["params"] = validated_groups
        out.append(item)

    return out

