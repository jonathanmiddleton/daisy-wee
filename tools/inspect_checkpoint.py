#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict

import torch
from torch import nn

from tools.checkpoint import load_checkpoint, peek_hparams, model_from_checkpoint
from tools.model_report import build_report


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
        model = model_from_checkpoint(path, device=args.device, map_location=args.device)
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
    print("=== Checkpoint ===")
    print(f"path: {path}")
    step = hparams.get("step") or None
    if step is not None:
        print(f"step: {step}")

    print("\n=== Hyperparameters ===")
    for k in sorted(hparams.keys()):
        print(f"{k}: {hparams[k]}")

    print("\n=== Model stats ===")
    print(f"parameters (total): {report['params_total_h']} ({report['params_total']})")
    print(f"parameters (trainable): {report['params_trainable_h']} ({report['params_trainable']})")
    if report.get("param_megabytes") is not None:
        print(f"parameter size: {report['param_megabytes']:.2f} MiB")
    if "model" in report:
        mi = report["model"]
        print(f"model type: {mi.get('type')}")
        if mi.get("num_layers") is not None:
            print(f"layers: {mi['num_layers']}")
        if mi.get("attn_off_layers"):
            print(f"attention skipped at layers: {mi['attn_off_layers']}")
        if mi.get("lm_head_rows") is not None and hparams.get("vocab_size"):
            vocab_size = int(hparams["vocab_size"]) or 0
            pad = mi["lm_head_rows"] - vocab_size
            if pad > 0:
                print(f"lm_head rows: {mi['lm_head_rows']} (padded by {pad} beyond vocab_size={vocab_size})")

    print("\nparameter dtypes:")
    for k, v in report["dtypes"].items():
        print(f"  {k}: {v['count_h']} ({v['count']})")

    print("\n=== Learned scalars (GPT2Core) ===")
    sc = report["scalars"]
    if not sc.get("present"):
        print("No 'scalars' parameter found in model.")
    else:
        L = sc.get("num_layers")
        print(f"num_layers (inferred): {L}")
        print(f"threshold for near-zero: {sc['zero_threshold']}")
        gsum = sc["groups"]
        for name in ("skip_weights", "lambdas", "sa_lambdas"):
            g = gsum.get(name)
            if not g:
                continue
            print(f"- {name}: shape={g['shape']}, min={g['min']:.4g}, max={g['max']:.4g}, mean={g['mean']:.4g}, std={g['std']:.4g}")
            print(f"  near-zero: {g['num_near_zero']} elements ({100.0*g['frac_near_zero']:.2f}%)")
        if sc.get("layers_with_skip_near_zero"):
            print(f"layers with near-zero skip weight: {sc['layers_with_skip_near_zero']}")
        # Per-layer compact print
        print("\nPer-layer (i: skip | lambda -> sa_lambda):")
        for li in sc["per_layer"]:
            i = li["layer"]
            def mark(val, is_nz):
                return f"{val:.4f}" + ("*" if is_nz else "")
            skip_s = mark(li["skip_w"], li["skip_w_near_zero"])
            lam_s = ", ".join(mark(v, nz) for v, nz in zip(li["lambda"], li["lambda_near_zero"]))
            sal_s = ", ".join(mark(v, nz) for v, nz in zip(li["sa_lambda"], li["sa_lambda_near_zero"]))
            print(f"  {i:02d}: {skip_s} | [{lam_s}] -> [{sal_s}]")
        if sc.get("any_near_zero"):
            print("\nNote: values marked with * are near zero and may indicate unused pathways.")


if __name__ == "main__" or __name__ == "__main__":
    main()


