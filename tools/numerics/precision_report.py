from __future__ import annotations

import argparse
import os
from typing import Optional

from .config import RunConfig
from .runner import PrecisionRunner


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Numerical Precision Characterization Report for GPT decoding")
    p.add_argument("-cfg", "--config", required=False, help="Path to YAML config per spec. If omitted, defaults are used.")
    p.add_argument("-pt", "--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    p.add_argument("-o", "--out", required=False, default=None, help="Artifacts root directory (overrides outputs.root)")
    args = p.parse_args(argv)

    if args.config:
        cfg = RunConfig.from_yaml(args.config)
    else:
        cfg = RunConfig()
    if args.out:
        cfg.outputs.root = args.out
    if not args.checkpoint or not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    runner = PrecisionRunner(cfg)
    runner.run(args.checkpoint)

    # Minimal markdown report (summary only)
    summ_path = os.path.join(cfg.outputs.root, "summaries", "case_summaries.json")
    rep_md = os.path.join(cfg.outputs.root, "reports", "precision_report.md")
    os.makedirs(os.path.dirname(rep_md), exist_ok=True)
    with open(summ_path, "r", encoding="utf-8") as f:
        import json
        summary = json.load(f)
    with open(rep_md, "w", encoding="utf-8") as f:
        f.write("# Numerical Precision Report\n\n")
        for cid, stats in summary.items():
            f.write(f"## {cid}\n")
            for k, v in stats.items():
                if isinstance(v, dict):
                    mean = v.get("mean")
                    ci = v.get("ci95")
                    med = v.get("median")
                    if ci is not None:
                        f.write(f"- {k}: mean={mean:.6f}, median={med:.6f}, 95% CI=({ci[0]:.6f}, {ci[1]:.6f})\n")
                    else:
                        f.write(f"- {k}: mean={mean:.6f}\n")
            f.write("\n")
    # Simple HTML as a copy of MD for now
    rep_html = os.path.join(cfg.outputs.root, "reports", "precision_report.html")
    with open(rep_md, "r", encoding="utf-8") as f:
        md_text = f.read()
    with open(rep_html, "w", encoding="utf-8") as f:
        f.write("<html><body><pre>\n")
        f.write(md_text)
        f.write("\n</pre></body></html>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
