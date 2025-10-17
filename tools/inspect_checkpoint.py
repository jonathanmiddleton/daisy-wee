#!/usr/bin/env python3


import argparse
import json
import math
import os
from typing import Any, Dict

import torch
from torch import nn

from tools.checkpoint import load_checkpoint, peek_hparams, model_from_checkpoint
from tools.model_report import build_report, format_report_text


def main():
    parser = argparse.ArgumentParser(description="Inspect a daisy-wee checkpoint and print model details.")
    parser.add_argument("path", help="Path to checkpoint .pt file")
    parser.add_argument("--device", default="cpu", help="Device to load the model on (default: cpu)")
    parser.add_argument("--zero-threshold", type=float, default=1e-3, help="Absolute value threshold to consider a scalar 'near zero'")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of pretty text")
    args = parser.parse_args()

    path = args.path
    if not os.path.exists(path):
        print(f"Checkpoint not found: {path}")
        raise SystemExit(2)

    # Peek hyperparameters without building the model
    try:
        hparams = peek_hparams(path, map_location="cpu")
    except Exception as e:
        print(f"Failed to read hyperparameters: {e}")
        hparams = {}

    # Build model and load state
    try:
        model = model_from_checkpoint(path, device=args.device)
    except Exception as e:
        print(f"Failed to load model from checkpoint: {e}")
        # Try partial: just state_dict for scalars analysis
        ckpt = load_checkpoint(path, map_location="cpu")
        dummy_model = nn.Module()
        dummy_model.load_state_dict(ckpt.model, strict=False)
        model = dummy_model

    report = build_report(model, hparams, args.__dict__["zero_threshold"])  # handle dash name
    report["path"] = path

    if args.json:
        print(json.dumps(report, indent=2))
        return

    # Pretty print
    print(format_report_text(report))


if __name__ == "main__" or __name__ == "__main__":
    main()


