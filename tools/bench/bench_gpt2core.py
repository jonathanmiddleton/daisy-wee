import argparse
import math
import time
import statistics as stats
from pathlib import Path
from typing import Optional

import torch
from torch import nn

from models import model_from_spec
from model_specs.model_spec import load_model_spec
from tools.checkpoint import model_from_checkpoint
from inference.kv_cache import KVCache


def _devtype(device: str) -> str:
    if str(device).startswith("cuda"):
        return "cuda"
    if str(device).startswith("mps"):
        return "mps"
    return "cpu"


class _NullCtx:
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc, tb):
        return False


def cuda_sync(device: str):
    class _Sync:
        def __enter__(self):
            if torch.cuda.is_available() and str(device).startswith("cuda"):
                torch.cuda.synchronize()
        def __exit__(self, exc_type, exc, tb):
            if torch.cuda.is_available() and str(device).startswith("cuda"):
                torch.cuda.synchronize()
    return _Sync()


def mean_ci(x: list[float], confidence: float = 0.95) -> tuple[float, float]:
    if not x:
        return float("nan"), float("nan")
    n = len(x)
    if n == 1:
        return x[0], 0.0
    m = stats.mean(x)
    sd = stats.pstdev(x) if n < 2 else stats.stdev(x)
    # Normal approx 95% CI: 1.96 * sd / sqrt(n)
    z = 1.96 if confidence == 0.95 else 1.96
    half = z * (sd / math.sqrt(n))
    return m, half


def build_cache(model: nn.Module, window: int, device: str, dtype: torch.dtype) -> KVCache:
    h = None; d = None
    for b in model.blocks:  # type: ignore[attr-defined]
        a = getattr(b, "attn", None)
        if a is not None:
            h = a.num_heads; d = a.head_dim; break
    assert h is not None and d is not None
    L = len(model.blocks)  # type: ignore[attr-defined]
    cache = KVCache(L=L, B=1, H=h, W=window, D=d, device=device, dtype=dtype)
    return cache


def run_benchmark(
    model: nn.Module,
    window: int,
    device: str,
    prompt_len: int,
    new_tokens: int,
    prefill_reps: int,
    step_reps: int,
    amp_dtype: Optional[torch.dtype] = torch.bfloat16,
    compile_model: bool = True,
    seed: int = 1234,
):
    torch.manual_seed(seed)
    vocab_size = int(model.embed.num_embeddings)  # type: ignore[attr-defined]

    # Prepare prompt ids (synthetic for speed/repeatability)
    prompt_ids = torch.randint(0, vocab_size, (prompt_len,), dtype=torch.long, device=device)

    devtype = _devtype(device)
    amp_ctx = torch.autocast(device_type=devtype, dtype=amp_dtype) if amp_dtype is not None else _NullCtx()

    if compile_model and device != 'cpu':
        model = torch.compile(model, dynamic=True)  # type: ignore[assignment]

    # Warmup: one prefill + a few steps
    cache = build_cache(model, window, device, dtype=torch.bfloat16)
    with amp_ctx:
        _logits, kv = model.prefill_batch(prompt_ids[None, :], window=window)  # type: ignore[attr-defined]
    cache.bulk_write_packed(kv.bfloat16(), pos=prompt_len, window=window)
    logits = _logits
    for _ in range(min(8, new_tokens)):
        # views
        k_ctxs, v_ctxs = [], []
        for i in range(len(model.blocks)):  # type: ignore[attr-defined]
            kc, vc = cache.view(i)
            k_ctxs.append(kc); v_ctxs.append(vc)
        with amp_ctx:
            token = torch.argmax(logits[0, :vocab_size]).to(dtype=torch.long)
            logits, k_new, v_new = model.step(token, k_ctxs, v_ctxs, cache.t, window)  # type: ignore[attr-defined]
        for i in range(len(model.blocks)):  # type: ignore[attr-defined]
            if k_new[i] is not None:
                cache.write(i, k_new[i], v_new[i])
        cache.advance()

    # Prefill timing
    prefill_times = []
    for _ in range(prefill_reps):
        cache.reset()
        with cuda_sync(device):
            t0 = time.perf_counter()
            with amp_ctx:
                _logits, kv = model.prefill_batch(prompt_ids[None, :], window=window)  # type: ignore[attr-defined]
            cache.bulk_write_packed(kv.bfloat16(), pos=prompt_len, window=window)
            t1 = time.perf_counter()
        prefill_times.append(t1 - t0)

    # Step timing (each rep includes fresh cache from prefill to isolate step speed)
    step_times = []
    for _ in range(step_reps):
        cache.reset()
        with amp_ctx:
            _logits, kv = model.prefill_batch(prompt_ids[None, :], window=window)  # type: ignore[attr-defined]
        cache.bulk_write_packed(kv.bfloat16(), pos=prompt_len, window=window)
        logits = _logits
        with cuda_sync(device):
            t0 = time.perf_counter()
            for _ in range(new_tokens):
                k_ctxs, v_ctxs = [], []
                for i in range(len(model.blocks)):  # type: ignore[attr-defined]
                    kc, vc = cache.view(i)
                    k_ctxs.append(kc); v_ctxs.append(vc)
                with amp_ctx:
                    token = torch.argmax(logits[0, :vocab_size]).to(dtype=torch.long)
                    logits, k_new, v_new = model.step(token, k_ctxs, v_ctxs, cache.t, window)  # type: ignore[attr-defined]
                for i in range(len(model.blocks)):  # type: ignore[attr-defined]
                    if k_new[i] is not None:
                        cache.write(i, k_new[i], v_new[i])
                cache.advance()
            t1 = time.perf_counter()
        step_times.append(t1 - t0)

    # Metrics
    prefill_tps = [prompt_len / t for t in prefill_times]
    step_tps = [new_tokens / t for t in step_times]

    prefill_mean, prefill_ci = mean_ci(prefill_tps)
    step_mean, step_ci = mean_ci(step_tps)

    return {
        "prefill_times": prefill_times,
        "step_times": step_times,
        "prefill_tps": prefill_tps,
        "step_tps": step_tps,
        "prefill_mean_tps": prefill_mean,
        "prefill_ci_tps": prefill_ci,
        "step_mean_tps": step_mean,
        "step_ci_tps": step_ci,
    }


def main():
    p = argparse.ArgumentParser(description="Benchmark GPT2Core prefill_batch and step performance.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", type=str, help="Path to checkpoint .pt file")
    src.add_argument("--model-spec", type=str, help="Model spec name or YAML path")
    p.add_argument("--prompt-length", type=int, default=1024, help="Prompt length (tokens)")
    p.add_argument("--new-tokens", type=int, default=256, help="Number of tokens to generate via step")
    p.add_argument("--prefill-reps", type=int, default=5, help="Number of prefill repetitions")
    p.add_argument("--step-reps", type=int, default=5, help="Number of step repetitions")
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"), help="Device, e.g. cpu, cuda, cuda:0")
    p.add_argument("--window", type=int, default=None, help="Attention window override (defaults from checkpoint/spec)")
    p.add_argument("--no-compile", action="store_true", help="Disable torch.compile even if available")
    p.add_argument("--seed", type=int, default=1234, help="Random seed for synthetic prompt")

    args = p.parse_args()

    device = args.device

    # Load model
    model: nn.Module
    window: int
    src_name: str
    if args.checkpoint:
        model, hparams = model_from_checkpoint(args.checkpoint, device=device)
        window = int(hparams.get('train_attention_window_len') or hparams.get('attention_window_len') or hparams.get('max_seq_len'))
        src_name = Path(args.checkpoint).name
    else:
        spec = load_model_spec(args.model_spec)
        model = model_from_spec(spec, device=device)
        window = int(spec["attention_window_len"])  # type: ignore[index]
        src_name = Path(args.model_spec).name

    if args.window is not None:
        window = int(args.window)

    model.eval()
    # Use bfloat16 embeddings to match runtime elsewhere
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            m.bfloat16()

    results = run_benchmark(
        model=model,
        window=window,
        device=device,
        prompt_len=int(args.prompt_length),
        new_tokens=int(args.new_tokens),
        prefill_reps=int(args.prefill_reps),
        step_reps=int(args.step_reps),
        amp_dtype=torch.bfloat16 if _devtype(device) != 'cpu' else None,
        compile_model=not args.no_compile,
        seed=int(args.seed),
    )

    # Render table
    # Columns: Source, Device, Window, Prompt, New, Prefill TPS (mean±CI), Step TPS (mean±CI)
    hdr = [
        ("source", 24),
        ("device", 8),
        ("win", 6),
        ("prompt", 8),
        ("new", 6),
        ("prefill tps", 18),
        ("step tps", 18),
    ]
    line = " ".join(name.ljust(w) for name, w in hdr)
    print(line)
    print("-" * len(line))

    def fmt_pair(mean: float, ci: float) -> str:
        if math.isnan(mean):
            return "n/a"
        return f"{mean:,.1f} b {ci:,.1f}"

    row = [
        str(src_name)[:24].ljust(24),
        str(device)[:8].ljust(8),
        str(window).rjust(6),
        str(int(args.prompt_length)).rjust(8),
        str(int(args.new_tokens)).rjust(6),
        f"{results['prefill_mean_tps']:.1f} +/- {results['prefill_ci_tps']:.1f}".rjust(18),
        f"{results['step_mean_tps']:.1f} +/- {results['step_ci_tps']:.1f}".rjust(18),
    ]
    print(" ".join(row))

    print()
    print("Notes: mean b 95% CI across repetitions. Synthetic random prompt tokens used.")


if __name__ == "__main__":
    main()
