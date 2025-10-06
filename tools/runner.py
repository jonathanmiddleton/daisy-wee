#!/usr/bin/env python3
"""
Runner for use in conjunction with run.sh, supporting Cartesian product overrides

Examples:
  python -m tools.runner config/pretrain.yml head_params_lr=0.7,0.8,0.9
  python -m tools.runner config/pretrain.yml head_params_lr=0.7,0.8,0.9 cooldown_frac=0.9,0.8,0.7

This will execute torchrun multiple times, once for each combination of overrides.
"""
from __future__ import annotations

import argparse
import itertools
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _setup_log_file() -> Tuple[Path, "io.TextIOBase"]:
    logs_dir = Path(__file__).resolve().parent.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{_timestamp()}.log"
    f = open(log_path, "a", buffering=1, encoding="utf-8")
    print(f"Logging to {log_path}")
    return log_path, f


def _split_override(arg: str) -> Tuple[str, List[str]]:
    """
    Parse a single override token like 'key=1,2,3' or '--key=1,2'.
    Returns (key, [values...]) with values split on commas (commas within quotes are not supported).
    If the token has no '=', it's treated as a boolean flag set to 'true'.
    """
    if arg.startswith("--"):
        arg = arg[2:]
    if "=" not in arg:
        # treat bare flag as true
        k = arg.strip().replace("-", "_")
        return k, ["true"]
    k, v = arg.split("=", 1)
    k = k.strip().replace("-", "_")
    # split by commas, ignore empty strings
    parts = [p for p in (v.split(",") if "," in v else [v])]
    parts = [p.strip() for p in parts if p is not None and p != ""]
    return k, parts if parts else [v]


def _cartesian_product(overrides: List[Tuple[str, List[str]]]) -> List[List[Tuple[str, str]]]:
    if not overrides:
        return [[]]
    keys = [k for k, _ in overrides]
    values_lists = [vals for _, vals in overrides]
    combos = []
    for prod in itertools.product(*values_lists):
        combos.append(list(zip(keys, prod)))
    return combos


def build_torchrun_cmd(
    *,
    nproc: int,
    config: str,
    checkpoint: str | None,
    extra_long_opts: List[str],
    overrides: List[Tuple[str, str]],
) -> List[str]:
    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={nproc}",
        "train.py",
        config,
    ]
    if checkpoint:
        cmd.append(f"--init_checkpoint={checkpoint}")
    # include any extra pre-parsed long opts (already prefixed with --)
    cmd.extend(extra_long_opts)
    # add overrides as --key=value
    for k, v in overrides:
        # preserve original key style expected by train.py (underscores); it'll accept --key=value
        cmd.append(f"--{k}={v}")
    return cmd


def _stream_subprocess(cmd: List[str], log_fp) -> int:
    """Run subprocess, streaming stdout/stderr to both console and log file."""
    # flush to avoid out-of-order
    sys.stdout.flush()
    sys.stderr.flush()
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
        env=os.environ.copy(),
    ) as p:
        assert p.stdout is not None
        for line in p.stdout:
            # write to console
            print(line, end="", flush=True)
            # write to log
            try:
                log_fp.write(line)
            except Exception:
                pass
        returncode = p.wait()
    return returncode


def main(argv: List[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # We accept a mix of short opts and trailing overrides. Use argparse for known ones and leave the rest.
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("-n", dest="nproc", type=int, default=1, help="nproc per node")
    parser.add_argument("-p", dest="checkpoint", default="", help="init checkpoint path")
    parser.add_argument("-s", dest="begin_shard", default="", help="BEGIN_SHARD env value")
    parser.add_argument("-r", dest="run_id", default="1", help="RUN_ID env value for the run")
    # Accept passthrough flags like --full_windows and arbitrary long options; we'll collect them

    # Parse known args and capture the rest for override processing
    args, extras = parser.parse_known_args(argv)

    config = args.config
    nproc = int(args.nproc)
    checkpoint = args.checkpoint or ""
    begin_shard = args.begin_shard or ""
    run_id = str(args.run_id)

    # Split the leftover overrides/long options
    raw_tail = list(extras or [])

    passthrough_long_opts: List[str] = []
    override_pairs: List[Tuple[str, List[str]]] = []

    i = 0
    while i < len(raw_tail):
        tok = raw_tail[i]
        if tok == "--":
            i += 1
            continue
        # Handle common flag aliases
        if tok in ("--full_windows", "--full-windows"):
            passthrough_long_opts.append("--full_windows=true")
            i += 1
            continue
        # --run_id style should map to env var RUN_ID but maintain compatibility
        if tok.startswith("--run_id="):
            run_id = tok.split("=", 1)[1]
            i += 1
            continue
        if tok.startswith("--run-id="):
            run_id = tok.split("=", 1)[1]
            i += 1
            continue
        # Any other token that starts with -- and either has = or not
        if tok.startswith("--") and ("=" not in tok):
            # preserve as-is long option without value
            passthrough_long_opts.append(tok)
            i += 1
            continue
        # For everything else, treat as an override key=value or --key=value
        k, vals = _split_override(tok)
        # If this looks like a long option not present in train Hyperparameters (can't easily know here),
        # but still in --foo=bar style, we'll forward as override; train.py will ignore unknown keys.
        override_pairs.append((k, vals))
        i += 1

    # Compute Cartesian product of overrides
    combos = _cartesian_product(override_pairs)

    # Environment setup
    os.environ.setdefault("TORCH_COMPILE_OFF", "0")
    os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "8")
    if begin_shard:
        os.environ["BEGIN_SHARD"] = str(begin_shard)
    # RUN_ID may be overridden per combo; we will increment optionally, but here set base
    base_run_id = int(run_id) if run_id.isdigit() else run_id

    log_path, log_fp = _setup_log_file()
    try:
        print(f"Config: {config}")
        print(f"nproc: {nproc}")
        if checkpoint:
            print(f"checkpoint: {checkpoint}")
        if begin_shard:
            print(f"BEGIN_SHARD: {begin_shard}")
        print(f"RUN_ID base: {base_run_id}")
        print(f"Extra opts: {' '.join(passthrough_long_opts) if passthrough_long_opts else '(none)'}")
        print(f"Override dimensions: {len(combos[0]) if combos and len(combos[0])>0 else 0}; total runs: {len(combos)}")
        # Execute each combination
        for idx, combo in enumerate(combos, start=1):
            # Update RUN_ID for each run: if numeric base, use base + (idx-1)
            if isinstance(base_run_id, int):
                os.environ["RUN_ID"] = str(base_run_id + (idx - 1))
            else:
                os.environ["RUN_ID"] = str(base_run_id)
            # Build command
            cmd = build_torchrun_cmd(
                nproc=nproc,
                config=config,
                checkpoint=checkpoint or None,
                extra_long_opts=passthrough_long_opts,
                overrides=combo,
            )
            print("\n=== Running (" + str(idx) + f"/{len(combos)}): " + shlex.join(cmd))
            rc = _stream_subprocess(cmd, log_fp)
            if rc != 0:
                print(f"Run {idx} failed with exit code {rc}. Aborting remaining runs.")
                return rc
        return 0
    finally:
        try:
            log_fp.flush()
            log_fp.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
