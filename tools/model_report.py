

from dataclasses import asdict
from typing import Any, Dict, Optional

import torch
from torch import nn

from training.hparams import Hyperparameters


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
    scalars = getattr(model, "scalars", None)
    if scalars is None:
        for name, t in model.state_dict().items():
            if name.endswith("scalars") and t.ndim == 1:
                scalars = t
                break

    if scalars is None:
        return out

    out["present"] = True
    s = scalars.detach().float().cpu()
    S = s.numel()
    out["length"] = int(S)
    L_from_hp = int(hparams.get("num_layers", 0) or 0)
    # assume scalars are 1 skip weight, 2 lambdas, and 2 sa_lambdas per layer
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


def build_report(model: nn.Module, hparams: Optional[Dict[str, Any] | Hyperparameters] = None, zero_threshold: float = 1e-3) -> Dict[str, Any]:
    """
    Build a model report for an instantiated nn.Module without requiring a checkpoint.

    Args:
        model: The PyTorch module to inspect.
        hparams: Optional hyperparameters dict that can enrich the report (e.g., num_layers, vocab_size).
        zero_threshold: Threshold for classifying learned scalars as near-zero.

    Returns:
        A dictionary with summary statistics about the model and (if present) learned scalars.
    """
    if hparams is not None and isinstance(hparams, Hyperparameters):
        hparams = asdict(hparams)

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

    # DaisyCore-specific info if available
    from models.daisy.daisy_core import DaisyCore  # local import
    if isinstance(model, DaisyCore):
        L = len(model.blocks)
        report.setdefault("model", {})
        report["model"].update({
            "type": "DaisyCore",
            "num_layers": L,
            "has_attn_every_layer": all(getattr(b, "attn", None) is not None for b in model.blocks),
            "attn_off_layers": [i for i, b in enumerate(model.blocks) if getattr(b, "attn", None) is None],
            "lm_head_rows": int(model.lm_head_w.shape[0]) if hasattr(model, "lm_head_w") else None,
            "lm_head_cols": int(model.lm_head_w.shape[1]) if hasattr(model, "lm_head_w") else None,
        })

    # Scalars analysis
    scalars_info = analyze_scalars(model, hparams, zero_threshold)
    report["scalars"] = scalars_info

    return report


def format_report_text(report: Dict[str, Any]) -> str:
    lines = ["=== Checkpoint ==="]

    # Path and step if available
    if report.get("path"):
        lines.append(f"path: {report['path']}")

    hparams = report.get("hparams", {}) or {}
    step = hparams.get("step") or None
    if step is not None:
        lines.append(f"step: {step}")

    # Hyperparameters
    if hparams:
        lines.append("\n=== Hyperparameters ===")
        for k in sorted(hparams.keys()):
            lines.append(f"{k}: {hparams[k]}")

    # Model stats
    lines.append("\n=== Model stats ===")
    lines.append(f"parameters (total): {report.get('params_total_h')} ({report.get('params_total')})")
    lines.append(f"parameters (trainable): {report.get('params_trainable_h')} ({report.get('params_trainable')})")
    if report.get("param_megabytes") is not None:
        lines.append(f"parameter size: {report['param_megabytes']:.2f} MiB")

    if "model" in report:
        mi = report["model"] or {}
        if mi:
            lines.append(f"model type: {mi.get('type')}")
            if mi.get("num_layers") is not None:
                lines.append(f"layers: {mi['num_layers']}")
            if mi.get("attn_off_layers"):
                lines.append(f"attention skipped at layers: {mi['attn_off_layers']}")
            if mi.get("lm_head_rows") is not None and hparams.get("vocab_size"):
                vocab_size = int(hparams["vocab_size"]) or 0
                pad = int(mi["lm_head_rows"]) - vocab_size
                if pad > 0:
                    lines.append(f"lm_head rows: {mi['lm_head_rows']} (padded by {pad} beyond vocab_size={vocab_size})")

    # Dtypes breakdown
    lines.append("\nparameter dtypes:")
    for k, v in (report.get("dtypes") or {}).items():
        lines.append(f"  {k}: {v['count_h']} ({v['count']})")

    # Scalars section
    lines.append("\n=== Learned scalars (DaisyCore) ===")
    sc = report.get("scalars", {}) or {}
    if not sc.get("present"):
        lines.append("No 'scalars' parameter found in model.")
    else:
        L = sc.get("num_layers")
        lines.append(f"num_layers (inferred): {L}")
        lines.append(f"threshold for near-zero: {sc['zero_threshold']}")
        gsum = sc.get("groups", {})
        for name in ("skip_weights", "lambdas", "sa_lambdas"):
            g = gsum.get(name)
            if not g:
                continue
            lines.append(f"- {name}: shape={g['shape']}, min={g['min']:.4g}, max={g['max']:.4g}, mean={g['mean']:.4g}, std={g['std']:.4g}")
            lines.append(f"  near-zero: {g['num_near_zero']} elements ({100.0*g['frac_near_zero']:.2f}%)")
        if sc.get("layers_with_skip_near_zero"):
            lines.append(f"layers with near-zero skip weight: {sc['layers_with_skip_near_zero']}")
        # Per-layer compact print
        lines.append("\nPer-layer (i: skip | lambda -> sa_lambda):")
        for li in sc.get("per_layer", []):
            i = li["layer"]
            def mark(val, is_nz):
                return f"{val:.4f}" + ("*" if is_nz else "")
            skip_s = mark(li["skip_w"], li["skip_w_near_zero"])
            lam_s = ", ".join(mark(v, nz) for v, nz in zip(li["lambda"], li["lambda_near_zero"]))
            sal_s = ", ".join(mark(v, nz) for v, nz in zip(li["sa_lambda"], li["sa_lambda_near_zero"]))
            lines.append(f"  {i:02d}: {skip_s} | [{lam_s}] -> [{sal_s}]")
        if sc.get("any_near_zero"):
            lines.append("\nNote: values marked with * are near zero and may indicate unused pathways.")

    return "\n".join(lines)

def report_from_training_yml(path: str, device: str = 'cpu') -> str:
    from training.hparams import load_hparams_from_yaml
    from models import model_from_spec
    hparams = load_hparams_from_yaml(path)
    model = model_from_spec(hparams.model_spec, device)
    return format_report_text(build_report(model, hparams))

__all__ = [
    "build_report",
    "analyze_scalars",
    "dtype_breakdown",
    "sizeof_params",
    "human_int",
    "format_report_text",
]

if __name__ == "__main__":
    import sys
    print(report_from_training_yml(sys.argv[1]))