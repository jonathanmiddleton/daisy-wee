from typing import get_origin, get_args, Any, Tuple, Dict
import yaml
from contextlib import contextmanager

import logging

logger = logging.getLogger(__name__)

def _coerce_value(val_str: str, typ):
    # Support Optional[...] and unions with None
    origin = get_origin(typ)
    args_ = get_args(typ)
    is_optional = False
    if origin is None:
        target_types = (typ,)
    elif origin is list or origin is tuple or origin is dict:
        # Simple YAML-like JSON parsing for collections
        # Fall back to YAML safe_load for flexible parsing
        return yaml.safe_load(val_str)
    elif origin is type(None):
        # Only NoneType
        is_optional = True
        target_types = (type(None),)
    else:
        # Assume Union
        target_types = args_
        if type(None) in target_types:
            is_optional = True
    # None coercion
    if val_str.lower() in ("none", "null"):
        if is_optional:
            return None
        # If not optional but asked for None, keep as string 'None'
        return val_str
    # Try booleans explicitly
    if bool in target_types or typ is bool:
        if val_str.lower() in ("1", "true", "t", "yes", "y", "on"):
            return True
        if val_str.lower() in ("0", "false", "f", "no", "n", "off"):
            return False
        # Fall through to attempt other conversions
    # Numeric conversions
    if int in target_types or typ is int:
        try:
            return int(val_str)
        except ValueError:
            pass
    if float in target_types or typ is float:
        try:
            return float(val_str)
        except ValueError:
            pass
    # Fallback: try YAML to parse literals, else string
    #noinspection PyBroadException
    try:
        parsed = yaml.safe_load(val_str)
        return parsed
    except Exception:
        return val_str

def _as_bool(value: Any, default: bool = False) -> bool:
    """Convert environment variable value to boolean."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, str)):
        v = str(value).lower()
        if v in ("1", "true", "yes", "y"):
            return True
        if v in ("0", "false", "no", "n"):
            return False
    return default


def _default_sanitizer(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    # Replace large/sensitive objects with summaries
    def summarize(x):
        #noinspection PyBroadException
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return {"type": "tensor", "shape": tuple(x.shape), "dtype": str(x.dtype), "device": str(x.device), "requires_grad": bool(x.requires_grad)}
        except Exception:
            pass
        if isinstance(x, (bytes, bytearray)):
            return f"<bytes:{len(x)}>"
        if isinstance(x, (str, int, float, bool, type(None))):
            return x
        # Avoid retaining references: return type name only
        return f"<{type(x).__name__}>"
    return tuple(summarize(a) for a in args), {k: summarize(v) for k, v in kwargs.items()}


@contextmanager
def measure_time():
    """
    Measure wall-clock time for a code block.

    Usage:
        with measure_time() as elapsed:
            do_work()
        print(elapsed())  # seconds as float

    Yields:
        Callable[[], float]: A zero-arg function that returns the elapsed
        seconds since entering the context.
    """
    import time

    t0 = time.perf_counter()
    yield lambda: time.perf_counter() - t0

def is_mac_os():
    import sys as _sys, platform
    return _sys.platform == "darwin" or platform.system() == "Darwin"

__all__ = ["_coerce_value", "measure_time"]
