from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn


STANDARD_KEYS = {
    "model",         # state_dict of the model
    "optimizers",    # list of optimizer.state_dict()
    "hparams",       # dict of hyperparameters used to build the model
    "step",          # int: global step when saved
    "best_val",      # float: best validation loss so far
}

UNWANTED_PREFIX = "_orig_mod."


@dataclass
class LoadedCheckpoint:
    model: Dict[str, torch.Tensor]
    hparams: Dict[str, Any]
    step: Optional[int]
    best_val: Optional[float]
    optimizers: Optional[List[Dict[str, Any]]]


def _strip_prefix(state_dict: Dict[str, Any], prefix: str = UNWANTED_PREFIX) -> Dict[str, Any]:
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_sd[k[len(prefix):]] = v
        else:
            new_sd[k] = v
    return new_sd


def _normalize(obj: Any) -> LoadedCheckpoint:
    # Accept either a bare state_dict or a dict with our standard layout (or a variant).
    if isinstance(obj, dict) and "model" in obj:
        model_sd = obj.get("model")
        hparams = obj.get("hparams") or {}
        step = obj.get("step")
        best_val = obj.get("best_val")
        optimizers = obj.get("optimizers")
        if not isinstance(hparams, dict):
            hparams = {}
        if not isinstance(optimizers, list):
            optimizers = None
        if not isinstance(model_sd, dict):
            # Some checkpoints might accidentally store the whole module
            try:
                model_sd = model_sd.state_dict()  # type: ignore[attr-defined]
            except Exception:
                raise TypeError("Unsupported checkpoint format: 'model' field is not a state_dict or module")
        return LoadedCheckpoint(model=model_sd, hparams=hparams, step=step, best_val=best_val, optimizers=optimizers)

    # Fallback: treat as a raw state_dict
    if isinstance(obj, dict):
        return LoadedCheckpoint(model=obj, hparams={}, step=None, best_val=None, optimizers=None)

    # Last resort: try to call state_dict on it
    try:
        sd = obj.state_dict()  # type: ignore[attr-defined]
        return LoadedCheckpoint(model=sd, hparams={}, step=None, best_val=None, optimizers=None)
    except Exception as e:
        raise TypeError(f"Unsupported checkpoint object: {type(obj)}") from e


def load_checkpoint(path: str, map_location: Any | None = None, strip_prefix: bool = True) -> LoadedCheckpoint:
    obj = torch.load(path, map_location=map_location)
    ckpt = _normalize(obj)
    if strip_prefix:
        ckpt.model = _strip_prefix(ckpt.model)
    return ckpt


def save_checkpoint(
    path: str,
    model: nn.Module | Dict[str, Any],
    optimizers: Optional[List[torch.optim.Optimizer | Dict[str, Any]]] = None,
    step: Optional[int] = None,
    best_val: Optional[float] = None,
    hparams: Optional[Dict[str, Any]] = None,
) -> None:
    # Accept either a module or a state_dict for model
    if isinstance(model, nn.Module):
        # If torch.compile was used, original module may be under _orig_mod
        model_to_state = getattr(model, "_orig_mod", model)
        model_sd = model_to_state.state_dict()
    elif isinstance(model, dict):
        model_sd = model
    else:
        raise TypeError("model must be an nn.Module or a state_dict dict")

    opt_states: Optional[List[Dict[str, Any]]] = None
    if optimizers is not None:
        opt_states = []
        for opt in optimizers:
            if isinstance(opt, dict):
                opt_states.append(opt)
            else:
                opt_states.append(opt.state_dict())

    payload = dict(
        step=step,
        model=model_sd,
        optimizers=opt_states,
        best_val=best_val,
        hparams=hparams or {},
    )
    torch.save(payload, path)


def apply_model_state(model: nn.Module, state_dict: Dict[str, Any], strict: bool = False) -> Tuple[List[str], List[str]]:
    # Convenience to load with common prefix stripping already handled by load_checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    return list(missing), list(unexpected)


def peek_hparams(path: str, map_location: Any | None = None) -> Dict[str, Any]:
    obj = torch.load(path, map_location=map_location)
    ckpt = _normalize(obj)
    return ckpt.hparams or {}
