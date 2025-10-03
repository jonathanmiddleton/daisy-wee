from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn


def human_int(n: int) -> str:
    s = str(n)
    groups = []
    while s and s[-3:]:
        groups.append(s[-3:])
        s = s[:-3]
    return ",".join(reversed(groups))


def sizeof_params(module: nn.Module) -> int:
    total = 0
    for p in module.parameters():
        if p is None:
            continue
        total += p.numel() * (torch.finfo(p.dtype).bits // 8 if p.is_floating_point() else p.element_size())
    return total


def dtype_breakdown(module: nn.Module) -> Dict[str, int]:
    d: Dict[str, int] = {}
    for p in module.parameters():
        key = str(p.dtype).replace("torch.", "")
        d[key] = d.get(key, 0) + p.numel()
    return d


def analyze_scalars(model: nn.Module, hparams: Dict[str, Any], zero_threshold: float) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "present": False,
        "length": 0,
        "num_layers": None,
        "zero_threshold": zero_threshold,
        "groups": {},
        "per_layer": [],
    }
    scalars: torch.Tensor | None = None
    try:
        scalars = getattr(model, "scalars", None)
    except Exception:
        scalars = None
    if scalars is None:
        # Try via state_dict in case model class differs
        try:
            for name, t in model.state_dict().items():
                if name.endswith("scalars") and t.ndim == 1:
                    scalars = t
                    break
        except Exception:
            scalars = None

    if scalars is None:
        return out

    out["present"] = True
    s = scalars.detach().float().cpu()
    S = s.numel()
    out["length"] = int(S)
    L_from_hp = int(hparams.get("num_layers", 0) or 0)
    L = L_from_hp if (L_from_hp and S % 5 == 0 and S // 5 == L_from_hp) else (S // 5)
    out["num_layers"] = int(L)

    if L <= 0 or S < 5 * L:
        out["error"] = f"Unexpected scalars shape: length={S}, num_layers={L}"
        return out

    skip_w = s[:L]
    lambdas = s[1 * L:3 * L].view(-1, 2)
    sa_lambdas = s[3 * L:5 * L].view(-1, 2)

    def nz_mask(x: torch.Tensor):
        return (x.abs() <= zero_threshold)

    groups = {
        "skip_weights": {
            "tensor": skip_w,
            "near_zero_mask": nz_mask(skip_w),
        },
        "lambdas": {
            "tensor": lambdas,
            "near_zero_mask": nz_mask(lambdas),
        },
        "sa_lambdas": {
            "tensor": sa_lambdas,
            "near_zero_mask": nz_mask(sa_lambdas),
        },
    }

    # Summaries
    for k, g in groups.items():
        t = g["tensor"]
        mask = g["near_zero_mask"]
        g["shape"] = list(t.shape)
        g["num_near_zero"] = int(mask.sum().item())
        g["frac_near_zero"] = float((mask.float().mean().item()))
        g["min"] = float(t.min().item())
        g["max"] = float(t.max().item())
        g["mean"] = float(t.mean().item())
        g["std"] = float(t.std(unbiased=False).item())
        # Flag layers fully off for skip weights or per‑element for pairs
    fully_off_layers = []
    per_layer = []
    for i in range(L):
        layer_info = {
            "layer": i,
            "skip_w": float(skip_w[i].item()),
            "skip_w_near_zero": bool(abs(skip_w[i].item()) <= zero_threshold),
            "lambda": [float(lambdas[i, 0].item()), float(lambdas[i, 1].item())],
            "lambda_near_zero": [bool(abs(lambdas[i, 0].item()) <= zero_threshold), bool(abs(lambdas[i, 1].item()) <= zero_threshold)],
            "sa_lambda": [float(sa_lambdas[i, 0].item()), float(sa_lambdas[i, 1].item())],
            "sa_lambda_near_zero": [bool(abs(sa_lambdas[i, 0].item()) <= zero_threshold), bool(abs(sa_lambdas[i, 1].item()) <= zero_threshold)],
        }
        if layer_info["skip_w_near_zero"]:
            fully_off_layers.append(i)
        per_layer.append(layer_info)

    out["groups"] = {k: {kk: vv for kk, vv in g.items() if kk != "tensor" and kk != "near_zero_mask"} for k, g in groups.items()}
    out["per_layer"] = per_layer
    out["layers_with_skip_near_zero"] = fully_off_layers
    out["any_near_zero"] = any(
        g["num_near_zero"] > 0 for g in out["groups"].values()
    )
    return out


def build_report(model: nn.Module, hparams: Optional[Dict[str, Any]] = None, zero_threshold: float = 1e-3) -> Dict[str, Any]:
    """
    Build a model report for an instantiated nn.Module without requiring a checkpoint.

    Args:
        model: The PyTorch module to inspect.
        hparams: Optional hyperparameters dict that can enrich the report (e.g., num_layers, vocab_size).
        zero_threshold: Threshold for classifying learned scalars as near-zero.

    Returns:
        A dictionary with summary statistics about the model and (if present) learned scalars.
    """
    hparams = dict(hparams or {})

    report: Dict[str, Any] = {}
    # Hyperparameters (optional)
    if hparams:
        report["hparams"] = hparams

    # Model stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    report["params_total"] = int(total_params)
    report["params_trainable"] = int(trainable_params)
    report["params_total_h"] = human_int(int(total_params))
    report["params_trainable_h"] = human_int(int(trainable_params))

    # Dtypes
    dtypes = dtype_breakdown(model)
    report["dtypes"] = {k: {"count": int(v), "count_h": human_int(int(v))} for k, v in dtypes.items()}

    # Size estimate
    try:
        bytes_ = sizeof_params(model)
    except Exception:
        bytes_ = 0
    report["param_bytes"] = int(bytes_)
    report["param_megabytes"] = float(bytes_ / (1024 ** 2)) if bytes_ else None

    # GPT2Core-specific info if available
    try:
        from models.gpt_core import GPT2Core  # local import
        if isinstance(model, GPT2Core):
            L = len(model.blocks)
            report.setdefault("model", {})
            report["model"].update({
                "type": "GPT2Core",
                "num_layers": L,
                "has_attn_every_layer": all(getattr(b, "attn", None) is not None for b in model.blocks),
                "attn_off_layers": [i for i, b in enumerate(model.blocks) if getattr(b, "attn", None) is None],
                "lm_head_rows": int(model.lm_head_w.shape[0]) if hasattr(model, "lm_head_w") else None,
                "lm_head_cols": int(model.lm_head_w.shape[1]) if hasattr(model, "lm_head_w") else None,
            })
    except Exception:
        pass

    # Scalars analysis
    scalars_info = analyze_scalars(model, hparams, zero_threshold)
    report["scalars"] = scalars_info

    return report


__all__ = [
    "build_report",
    "analyze_scalars",
    "dtype_breakdown",
    "sizeof_params",
    "human_int",
]
