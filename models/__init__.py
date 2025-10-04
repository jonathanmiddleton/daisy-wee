from __future__ import annotations

from typing import Dict, Type

# Prefer the migrated path under models.gpt2
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
