from __future__ import annotations

import json
import os
import platform
from dataclasses import asdict
from typing import Any, Dict

import torch


def collect_env() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    info["python"] = {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
    }
    info["os"] = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    info["torch"] = {
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False,
        "backends": {
            "cudnn": getattr(torch.backends, "cudnn", None) is not None,
            "mps": getattr(torch.backends, "mps", None) is not None,
            "mkldnn": getattr(torch.backends, "mkldnn", None) is not None,
        },
        "deterministic_algos": torch.are_deterministic_algorithms_enabled(),
    }
    info["env"] = {
        "float32_matmul_precision": torch.get_float32_matmul_precision() if hasattr(torch, "get_float32_matmul_precision") else None,
        "PYTORCH_CUDA_ALLOC_CONF": os.getenv("PYTORCH_CUDA_ALLOC_CONF"),
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO": os.getenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO"),
    }
    # Device properties
    devices = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            cap = torch.cuda.get_device_capability(i)
            devices.append({
                "type": "cuda",
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "capability": cap,
                "total_memory": int(torch.cuda.get_device_properties(i).total_memory),
            })
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append({"type": "mps", "index": 0})
    devices.append({"type": "cpu"})
    info["devices"] = devices
    return info


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def record_env(root: str) -> Dict[str, Any]:
    info = collect_env()
    write_json(os.path.join(root, "logs", "env.json"), info)
    return info


def record_unsupported(root: str, payload: Any) -> None:
    write_json(os.path.join(root, "logs", "unsupported.json"), payload)
