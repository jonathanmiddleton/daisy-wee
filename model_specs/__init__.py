from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def resolve_model_spec_path(name_or_path: str) -> Path:
    """
    Resolve a model spec identifier to a concrete file path.
    Accepts:
      - Absolute/relative file paths to a YAML file.
      - Bare names referencing files inside this package directory (with or without .yml/.yaml).
    Returns a Path that exists or raises FileNotFoundError.
    """
    # Direct path first
    p = Path(name_or_path)
    if p.exists():
        return p

    # Try relative to this package directory
    base = Path(__file__).resolve().parent

    # If user provided a name with extension, check directly under package
    named = base / name_or_path
    if named.exists():
        return named

    # Try adding common YAML extensions under package dir
    cand1 = base / f"{name_or_path}.yml"
    cand2 = base / f"{name_or_path}.yaml"
    if cand1.exists():
        return cand1
    if cand2.exists():
        return cand2

    # As a last attempt, allow bare files in current working directory with YAML extensions
    cwd1 = Path(f"{name_or_path}.yml")
    cwd2 = Path(f"{name_or_path}.yaml")
    if cwd1.exists():
        return cwd1
    if cwd2.exists():
        return cwd2

    raise FileNotFoundError(
        f"Model spec '{name_or_path}' not found under {base} or as a file path"
    )


def load_model_spec(name_or_path: str) -> Dict[str, Any]:
    """
    Load and parse a model spec YAML into a dict. Returns an empty dict if the YAML is empty.
    Raises FileNotFoundError if the spec cannot be resolved.
    """
    spec_path = resolve_model_spec_path(name_or_path)
    with open(spec_path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        # Be tolerant; only dict specs are useful for merging
        return {}
    return data


__all__ = [
    "resolve_model_spec_path",
    "load_model_spec",
]
