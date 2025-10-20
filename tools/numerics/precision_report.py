import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from .config import RunConfig
from .runner import PrecisionRunner


def _ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _load_table(path_parquet: str):
    # Try pandas parquet, fall back to CSV if parquet missing or engine unavailable
    import pandas as pd  # type: ignore
    # Prefer parquet
    return pd.read_parquet(path_parquet)


def _format_mean_pm_ci(entry: Dict[str, Any], digits: int = 4) -> str:
    if not isinstance(entry, dict):
        return ""
    mean = entry.get("mean")
    ci = entry.get("ci95")
    if mean is None:
        return ""
    if ci is None or not isinstance(ci, (list, tuple)) or len(ci) != 2:
        return f"{mean:.{digits}f}"
    hw = (float(ci[1]) - float(ci[0])) / 2.0
    return f"{float(mean):.{digits}f} ± {hw:.{digits}f}"


def _format_rate_pm_wilson(successes: int, trials: int, digits: int = 4) -> str:
    if trials <= 0:
        return "n/a"
    p_hat = successes / trials
    # Wilson interval
    z = 1.96
    denom = 1 + z * z / trials
    center = (p_hat + z * z / (2 * trials)) / denom
    hw = z * math.sqrt(p_hat * (1 - p_hat) / trials + z * z / (4 * trials * trials)) / denom
    return f"{center:.{digits}f} ± {hw:.{digits}f}"


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

    # Paths
    root = cfg.outputs.root
    summ_path = os.path.join(root, "summaries", "case_summaries.json")
    comp_path = os.path.join(root, "summaries", "comparisons.json")
    div_summ_path = os.path.join(root, "summaries", "divergence_summary.json")
    tokens_path = os.path.join(root, "open_loop", "tokens.parquet")
    div_path = os.path.join(root, "closed_loop", "divergence.parquet")
    rep_md = os.path.join(root, "reports", "precision_report.md")
    figs_dir = os.path.join(root, "reports", "figures")
    _ensure_parent(rep_md)
    os.makedirs(figs_dir, exist_ok=True)

    # Load summaries
    with open(summ_path, "r", encoding="utf-8") as f:
        summary: Dict[str, Any] = json.load(f)
    comparisons: Dict[str, Any] = {}
    if os.path.exists(comp_path):
        with open(comp_path, "r", encoding="utf-8") as f:
            comparisons = json.load(f)
    div_summ: Dict[str, Any] = {}
    if os.path.exists(div_summ_path):
        with open(div_summ_path, "r", encoding="utf-8") as f:
            div_summ = json.load(f)

    # Create tables by case: ΔNLL, JS, flip rate, top-k overlaps (mean ± 95% CI)
    import pandas as pd
    tokens_df = _load_table(tokens_path)
    div_df = _load_table(div_path)

    # Compute flip rate with Wilson CI from tokens_df if available
    flip_rows: Dict[str, str] = {}
    if tokens_df is not None:
        # Normalize column names possibly coming from CSV with case_id
        if "case_id" in tokens_df.columns and "flip_top1" in tokens_df.columns:
            gb = tokens_df.groupby("case_id")
            for cid, g in gb:
                # Treat truthy / boolean flips
                flips = g["flip_top1"].astype(int)
                s = int(flips.sum())
                n = int(flips.count())
                flip_rows[str(cid)] = _format_rate_pm_wilson(s, n)

    # Build summary table
    case_rows: List[Dict[str, Any]] = []
    # Determine topk columns present in summary
    topk_cols: List[str] = []
    for v in summary.values():
        for k in list(v.keys()):
            if k.startswith("topk_overlap@"):  # collect
                if k not in topk_cols:
                    topk_cols.append(k)
    topk_cols = sorted(topk_cols, key=lambda s: int(s.split("@")[1]))

    for cid, stats in summary.items():
        row: Dict[str, Any] = {"case": cid}
        row["ΔNLL (mean±95%CI)"] = _format_mean_pm_ci(stats.get("delta_nll", {}))
        row["JS (mean±95%CI)"] = _format_mean_pm_ci(stats.get("js", {}))
        # Flip rate
        fr = flip_rows.get(cid)
        if fr is None and isinstance(stats.get("flip_top1"), dict):
            # no CI in summary; show mean only
            m = stats["flip_top1"].get("mean")
            fr = f"{float(m):.4f}" if m is not None else ""
        row["Flip rate (Wilson 95% CI)"] = fr or ""
        for tk in topk_cols:
            row[f"{tk} (mean±95%CI)"] = _format_mean_pm_ci(stats.get(tk, {}))
        case_rows.append(row)

    if pd is not None:
        table_df = pd.DataFrame(case_rows)
        table_md = table_df.to_markdown(index=False)
    else:
        # Minimal ASCII without pandas
        headers = list(case_rows[0].keys()) if case_rows else []
        lines = []
        if headers:
            lines.append(" | ".join(headers))
            lines.append(" | ".join(["---" for _ in headers]))
            for r in case_rows:
                lines.append(" | ".join(str(r.get(h, "")) for h in headers))
        table_md = "\n".join(lines)

    # Generate figures per §15
    figs = []
    # 1) Boxplots of per-token L2 and JS by case
    import matplotlib.pyplot as plt  # type: ignore
    import seaborn as sns  # type: ignore

    if plt is not None and sns is not None and tokens_df is not None and "case_id" in tokens_df.columns:
        # Cap extreme values for nicer plots (optional)
        def _clip_series(s, q=0.999):
            try:
                import numpy as np  # type: ignore
                hi = float(s.quantile(q))
                return s.clip(upper=hi)
            except Exception:
                return s
        # L2 boxplot
        if "l2" in tokens_df.columns:
            fig_path = os.path.join(figs_dir, "boxplot_l2_by_case.png")
            plt.figure(figsize=(max(6, len(tokens_df["case_id"].unique()) * 0.9), 4))
            sns.boxplot(data=tokens_df, x="case_id", y=_clip_series(tokens_df["l2"]))
            plt.xticks(rotation=45, ha="right")
            plt.title("Per-token L2 drift by case")
            plt.tight_layout()
            plt.savefig(fig_path, dpi=160)
            plt.close()
            figs.append(("Boxplot: L2 by case", os.path.relpath(fig_path, os.path.dirname(rep_md))))
        # JS boxplot
        if "js" in tokens_df.columns:
            fig_path = os.path.join(figs_dir, "boxplot_js_by_case.png")
            plt.figure(figsize=(max(6, len(tokens_df["case_id"].unique()) * 0.9), 4))
            sns.boxplot(data=tokens_df, x="case_id", y=_clip_series(tokens_df["js"]))
            plt.xticks(rotation=45, ha="right")
            plt.title("Per-token JS divergence by case")
            plt.tight_layout()
            plt.savefig(fig_path, dpi=160)
            plt.close()
            figs.append(("Boxplot: JS by case", os.path.relpath(fig_path, os.path.dirname(rep_md))))
    
    # 2) Heatmap: mean layerwise drift per block (if available)
    # Not collected in current pipeline; generate placeholder if unavailable
    heatmap_created = False
    if plt is not None and sns is not None:
        # Look for columns like layer, block, l2_layer etc.
        if tokens_df is not None and "layer" in getattr(tokens_df, "columns", []):
            # If there were a per-layer metric like l2_layer, compute mean per (case, layer)
            metric_col = "l2"
            pivot = tokens_df.pivot_table(index="layer", columns="case_id", values=metric_col, aggfunc="mean")
            fig_path = os.path.join(figs_dir, "heatmap_mean_layerwise_drift.png")
            plt.figure(figsize=(8, 4))
            sns.heatmap(pivot, cmap="viridis")
            plt.title("Mean layerwise drift per block")
            plt.tight_layout()
            plt.savefig(fig_path, dpi=160)
            plt.close()
            figs.append(("Heatmap: mean layerwise drift per block", os.path.relpath(fig_path, os.path.dirname(rep_md))))
            heatmap_created = True
    if not heatmap_created:
        # Create a tiny placeholder image stating not available
        if plt is not None:
            fig_path = os.path.join(figs_dir, "heatmap_mean_layerwise_drift_na.png")
            plt.figure(figsize=(4, 2))
            plt.text(0.5, 0.5, "Layerwise drift not collected\n(heatmap unavailable)", ha="center", va="center")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(fig_path, dpi=160)
            plt.close()
            figs.append(("Heatmap: mean layerwise drift per block (N/A)", os.path.relpath(fig_path, os.path.dirname(rep_md))))

    # 3) Distribution of first divergence index per case
    if plt is not None and sns is not None and div_df is not None and "case_id" in div_df.columns and "first_div_idx" in div_df.columns:
        fig_path = os.path.join(figs_dir, "first_divergence_distribution.png")
        plt.figure(figsize=(max(6, len(div_df["case_id"].unique()) * 0.9), 4))
        sns.histplot(data=div_df, x="first_div_idx", hue="case_id", element="step", stat="density", common_norm=False)
        plt.title("Distribution of first divergence index by case")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=160)
        plt.close()
        figs.append(("Distribution: first divergence index by case", os.path.relpath(fig_path, os.path.dirname(rep_md))))

    # 4) Flip-given-margin curves per case
    # Prefer using bins and rates from summary if present
    if plt is not None and summary:
        fig_path = os.path.join(figs_dir, "flip_given_margin_curves.png")
        plt.figure(figsize=(6, 4))
        # Build curves
        for cid, stats in summary.items():
            fgm = stats.get("flip_given_margin")
            if isinstance(fgm, dict) and fgm:
                # Convert categorical bins to x numeric positions
                xs: List[float] = []
                ys: List[float] = []
                for name, d in fgm.items():
                    rate = d.get("rate")
                    if rate is None or not (rate == rate):  # NaN check
                        continue
                    # parse name like "(lo,hi]" or ">x"
                    if name.startswith("(") and "," in name:
                        try:
                            lo_str, hi_str = name[1:-1].split(",")
                            lo = float(lo_str)
                            hi = float(hi_str) if hi_str != "None" and hi_str != "" else None
                            x = (lo + (hi if hi is not None else lo + 1.0)) / 2.0
                        except Exception:
                            x = float(len(xs))
                    elif name.startswith(">"):
                        try:
                            x = float(name[1:])
                        except Exception:
                            x = float(len(xs))
                    else:
                        x = float(len(xs))
                    xs.append(x)
                    ys.append(float(rate))
                if xs:
                    xs2, ys2 = zip(*sorted(zip(xs, ys)))
                    plt.plot(xs2, ys2, marker="o", label=cid)
        plt.xlabel("Reference margin")
        plt.ylabel("Flip rate")
        plt.title("Flip-given-margin")
        if summary and len(summary) <= 12:
            plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=160)
        plt.close()
        figs.append(("Flip-given-margin curves", os.path.relpath(fig_path, os.path.dirname(rep_md))))

    # Write markdown report
    with open(rep_md, "w", encoding="utf-8") as f:
        f.write("# Numerical Precision Report\n\n")
        f.write("### Tables by case: mean ± 95% CI for ΔNLL, JS, flip rate, top-k overlaps\n\n")
        if table_md:
            f.write(table_md + "\n\n")
        else:
            f.write("(No data)\n\n")
        # Figures
        for title, relpath in figs:
            f.write(f"### {title}\n\n")
            f.write(f"![]({relpath})\n\n")
        # Optional: brief divergence summary table
        if div_summ:
            try:
                if pd is not None:
                    ds_df = pd.DataFrame([{**{"case": k}, **v} for k, v in div_summ.items()])
                    f.write("### Divergence summary\n\n")
                    f.write(ds_df.to_markdown(index=False) + "\n\n")
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    start_wall = datetime.now(timezone.utc)
    start_cpu = time.perf_counter()
    print(f"[START] {start_wall.isoformat()}")
    try:
        main()
    finally:
        end_wall = datetime.now(timezone.utc)
        elapsed_sec = time.perf_counter() - start_cpu
        print(f"[END]   {end_wall.isoformat()}  (elapsed: {elapsed_sec:.3f}s)")


