import argparse
import math
import time
import os
import statistics as stats
from pathlib import Path
from typing import Optional
import contextlib

import torch
from torch import nn

from models import model_from_spec
from model_specs import load_model_spec
from tools.checkpoint import model_from_checkpoint
from inference.kv_cache import KVCache
from inference.generate import Generator

@contextlib.contextmanager
def dev_sync(device: str):
    is_cuda = torch.cuda.is_available() and str(device).startswith("cuda")
    is_mps = hasattr(torch, "mps") and torch.mps.is_available() and str(device).startswith("mps")
    if is_cuda:
        torch.cuda.synchronize()
    elif is_mps:
        torch.mps.synchronize()
    try:
        yield
    finally:
        if is_cuda:
            torch.cuda.synchronize()
        elif is_mps:
            torch.mps.synchronize()

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

@torch.inference_mode()
def run_benchmark(
    model: nn.Module,
    window: int,
    device: str,
    prompt_len: int,
    new_tokens: int,
    prefill_reps: int,
    step_reps: int,
    dtype: torch.dtype,
    compile_model: bool = True,
    seed: int = 1234,
):
    print("Running benchmarks, please wait... ", flush=True)
    torch.manual_seed(seed)
    vocab_size = int(model.embed.num_embeddings)  # type: ignore[attr-defined]

    # Prepare prompt ids (synthetic for speed/repeatability)
    prompt_ids = torch.randint(0, vocab_size, (prompt_len,), dtype=torch.long, device=device)

    devtype = _devtype(device)

    if compile_model and device != 'cpu':
        model = torch.compile(model, dynamic=False)  # type: ignore[assignment]

    # Warmup: one prefill + a few steps
    cache = build_cache(model, window, device, dtype=dtype)
    _logits, kv = model.prefill(prompt_ids[None, :], window=window)  # type: ignore[attr-defined]
    cache.bulk_write_packed(kv.to(dtype), pos=prompt_len, window=window)
    logits = _logits
    for _ in range(min(8, new_tokens)):
        # views
        k_ctxs, v_ctxs = [], []
        for i in range(len(model.blocks)):  # type: ignore[attr-defined]
            kc, vc = cache.view(i)
            k_ctxs.append(kc); v_ctxs.append(vc)
        token = torch.argmax(logits[0, :vocab_size]).to(dtype=torch.long)
        logits, k_new, v_new = model.step(token, k_ctxs, v_ctxs, cache.t, window)  # type: ignore[attr-defined]
        for i in range(len(model.blocks)):  # type: ignore[attr-defined]
            if k_new[i] is not None:
                cache.write(i, k_new[i], v_new[i])
        cache.advance()

    # Prefill timing
    prefill_times = []
    for _ in range(prefill_reps):
        cache.reset_history()
        with dev_sync(device):
            t0 = time.perf_counter()
            _logits, kv = model.prefill(prompt_ids[None, :], window=window)  # type: ignore[attr-defined]
            cache.bulk_write_packed(kv.to(dtype), pos=prompt_len, window=window)
        t1 = time.perf_counter()
        prefill_times.append(t1 - t0)

    # Step timing (each rep includes fresh cache from prefill to isolate step speed)
    step_times = []
    for _ in range(step_reps):
        cache.reset_history()
        _logits, kv = model.prefill(prompt_ids[None, :], window=window)  # type: ignore[attr-defined]
        cache.bulk_write_packed(kv.to(dtype), pos=prompt_len, window=window)
        logits = _logits
        with dev_sync(device):
            t0 = time.perf_counter()
            for _ in range(new_tokens):
                k_ctxs, v_ctxs = [], []
                for i in range(len(model.blocks)):  # type: ignore[attr-defined]
                    kc, vc = cache.view(i)
                    k_ctxs.append(kc); v_ctxs.append(vc)
                token = torch.argmax(logits[0, :vocab_size]).to(dtype=torch.long)
                logits, k_new, v_new = model.step(token, k_ctxs, v_ctxs, cache.t, window)  # type: ignore[attr-defined]
                for i in range(len(model.blocks)):  # type: ignore[attr-defined]
                    if k_new[i] is not None:
                        cache.write(i, k_new[i], v_new[i])
                cache.advance()
        t1 = time.perf_counter()
        step_times.append(t1 - t0)

    # Generator-based timings using Generator.generate()
    gen_prefill_times: list[float] = []
    gen_step_times: list[float] = []
    gen = Generator(model, window=window, seed=1337, device=device, dtype=dtype, temperature=0.0)
    for _ in range(step_reps):
        gen.reset_history()
        it = gen.generate(prompt_ids, max_new_tokens=new_tokens)
        while True:
            try:
                next(it)
            except StopIteration as e:
                _out, gen_prefill_dur, gen_step_dur = e.value
                gen_prefill_times.append(float(gen_prefill_dur))
                gen_step_times.append(float(gen_step_dur))
                break

    # Metrics
    prefill_tps = [prompt_len / t for t in prefill_times]
    step_tps = [new_tokens / t for t in step_times]
    gen_prefill_tps = [prompt_len / t for t in gen_prefill_times]
    gen_step_tps = [new_tokens / t for t in gen_step_times]

    prefill_mean, prefill_ci = mean_ci(prefill_tps)
    step_mean, step_ci = mean_ci(step_tps)
    gen_prefill_mean, gen_prefill_ci = mean_ci(gen_prefill_tps)
    gen_step_mean, gen_step_ci = mean_ci(gen_step_tps)

    return {
        "prefill_times": prefill_times,
        "step_times": step_times,
        "gen_prefill_times": gen_prefill_times,
        "gen_step_times": gen_step_times,
        "prefill_tps": prefill_tps,
        "step_tps": step_tps,
        "gen_prefill_tps": gen_prefill_tps,
        "gen_step_tps": gen_step_tps,
        "prefill_mean_tps": prefill_mean,
        "prefill_ci_tps": prefill_ci,
        "step_mean_tps": step_mean,
        "step_ci_tps": step_ci,
        "gen_prefill_mean_tps": gen_prefill_mean,
        "gen_prefill_ci_tps": gen_prefill_ci,
        "gen_step_mean_tps": gen_step_mean,
        "gen_step_ci_tps": gen_step_ci,
    }


def main():
    p = argparse.ArgumentParser(description="Benchmark DaisyCore prefill and step performance.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", type=str, help="Path to checkpoint .pt file")
    src.add_argument("--model-spec", type=str, help="Model spec name or YAML path")
    p.add_argument("-dtp", "--dtype", type=str, default="float16", help="Data type [float16, bfloat16, float32]")
    p.add_argument("-pg", "--prompt-length", type=int, default=1024, help="Prompt length (tokens)")
    p.add_argument("-tg", "--new-tokens", type=int, default=256, help="Number of tokens to generate")
    p.add_argument("-rp", "--reps-prefill", type=int, default=10, help="Number of prefill repetitions")
    p.add_argument("-rs", "--reps-steps", type=int, default=10, help="Number of token generation repetitions")
    p.add_argument("-d", "--device", type=str, default=("cuda" if torch.cuda.is_available() else ("mps" if torch.mps.is_available() else "cpu")), help="Device, e.g. cpu, mps, cuda, cuda:0")
    p.add_argument("--window", type=int, default=None, help="Attention window override (defaults from checkpoint/spec)")
    p.add_argument("--no-compile", action="store_true", help="Disable torch.compile even if available")
    p.add_argument("--seed", type=int, default=1234, help="Random seed for synthetic prompt")

    args = p.parse_args()

    device = args.device

    dtype = torch.float16 if args.dtype == "float16" else (torch.bfloat16 if args.dtype == "bfloat16" else torch.float32)

    # Load model
    model: nn.Module
    window: int
    src_name: str
    if args.checkpoint:
        model, hparams = model_from_checkpoint(args.checkpoint, device=device)
        window = int(hparams.get('attention_window_len') or hparams.get('train_attention_window_len'))
        src_name = Path(args.checkpoint).name
    else:
        spec = load_model_spec(args.model_spec)
        model = model_from_spec(spec, device=device)
        window = int(spec["attention_window_len"])  # type: ignore[index]
        src_name = Path(args.model_spec).name

    if args.window is not None:
        window = int(args.window)

    model.eval()

    results = run_benchmark(
        model=model,
        window=window,
        device=device,
        prompt_len=int(args.prompt_length),
        new_tokens=int(args.new_tokens),
        prefill_reps=int(args.reps_prefill),
        step_reps=int(args.reps_steps),
        dtype=dtype,
        compile_model=not args.no_compile,
        seed=int(args.seed),
    )

    def human_params(n: int) -> str:
        if n >= 1_000_000_000:
            return f"{n/1e9:.2f} B"
        if n >= 1_000_000:
            return f"{n/1e6:.2f} M"
        if n >= 1_000:
            return f"{n/1e3:.2f} K"
        return str(n)

    def human_gib(nbytes: int) -> str:
        return f"{nbytes / (1024**3):.2f} GiB"

    def _detect_cpu_blas() -> str:
        try:
            cfg = torch.__config__.show()
        except Exception:
            return "BLAS"
        lower = cfg.lower()
        # Prefer explicit BLAS info lines when present
        for line in cfg.splitlines():
            ll = line.lower()
            if ("blas_info" in ll) or ("blas=" in ll) or ("lapack_info" in ll):
                if ("accelerate" in ll) or ("veclib" in ll):
                    return "Accelerate"
                if ("openblas" in ll) or ("open" in ll):
                    return "OpenBLAS"
                if "mkl" in ll:
                    return "MKL"
                if "atlas" in ll:
                    return "ATLAS"
        # Fallback: search the whole config text
        if ("accelerate" in lower) or ("veclib" in lower):
            return "Accelerate"
        if "openblas" in lower:
            return "OpenBLAS"
        if "mkl" in lower:
            return "MKL"
        if "atlas" in lower:
            return "ATLAS"
        return "BLAS"

    def detect_backend(devtype: str) -> str:
        cpu_blas = _detect_cpu_blas()
        if devtype == "cuda":
            return f"CUDA,{cpu_blas}"
        if devtype == "mps":
            return f"Metal,{cpu_blas}"
        return cpu_blas

    devtype = _devtype(device)
    try:
        param_count = sum(p.numel() for p in model.parameters())
    except Exception:
        param_count = int(getattr(model, 'num_parameters', 0)) or 0

    if args.checkpoint and os.path.exists(args.checkpoint):
        try:
            size_bytes = os.path.getsize(args.checkpoint)
        except Exception:
            size_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    else:
        size_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())

    backend = detect_backend(devtype)
    threads = torch.get_num_threads()

    # Table layout
    cols = [
        ("model", 30, 'l'),
        ("size", 11, 'r'),
        ("params", 11, 'r'),
        ("backend", 18, 'l'),
        ("threads", 8, 'r'),
        ("test", 12, 'r'),
        ("reps", 5, 'r'),
        ("t/s", 21, 'r'),
    ]

    def pad(val: str, width: int, align: str) -> str:
        if len(val) > width:
            val = val[:width]
        if align == 'r':
            return val.rjust(width)
        return val.ljust(width)

    header = "| " + " | ".join(pad(name, w, a) for name, w, a in cols) + " |"
    sep = "| " + " | ".join(("-"*(w-1) + ":") if a == 'r' else ("-"*w) for _, w, a in cols) + " |"
    print(header)
    print(sep)

    # Rows: prefill and token generation
    pp_name = f"pp{int(args.prompt_length)}"
    tg_name = f"tg{int(args.new_tokens)}"
    pp_reps = f"{int(args.reps_prefill)}"
    tg_reps = f"{int(args.reps_steps)}"
    prefill_str = f"{results['prefill_mean_tps']:.2f} ± {results['prefill_ci_tps']:.2f}"
    step_str = f"{results['step_mean_tps']:.2f} ± {results['step_ci_tps']:.2f}"
    gen_prefill_str = f"{results['gen_prefill_mean_tps']:.2f} ± {results['gen_prefill_ci_tps']:.2f}"
    gen_step_str = f"{results['gen_step_mean_tps']:.2f} ± {results['gen_step_ci_tps']:.2f}"

    static_cells = [
        pad(str(src_name), cols[0][1], cols[0][2]),
        pad(human_gib(int(size_bytes)), cols[1][1], cols[1][2]),
        pad(human_params(int(param_count)), cols[2][1], cols[2][2]),
        pad(backend, cols[3][1], cols[3][2]),
        pad(str(threads if threads > 0 else '-'), cols[4][1], cols[4][2]),
    ]

    row_pp = "| " + " | ".join(static_cells + [
        pad(pp_name, cols[5][1], cols[5][2]),
        pad(pp_reps, cols[6][1], cols[6][2]),
        pad(prefill_str, cols[7][1], cols[7][2]),
    ]) + " |"

    row_tg = "| " + " | ".join(static_cells + [
        pad(tg_name, cols[5][1], cols[5][2]),
        pad(tg_reps, cols[6][1], cols[6][2]),
        pad(step_str, cols[7][1], cols[7][2]),
    ]) + " |"

    # Generator-based rows
    ppG_name = f"ppG={int(args.prompt_length)}"
    tgG_name = f"tgG={int(args.new_tokens)}"

    row_ppG = "| " + " | ".join(static_cells + [
        pad(ppG_name, cols[5][1], cols[5][2]),
        pad(pp_reps, cols[6][1], cols[6][2]),
        pad(gen_prefill_str, cols[7][1], cols[7][2]),
    ]) + " |"

    row_tgG = "| " + " | ".join(static_cells + [
        pad(tgG_name, cols[5][1], cols[5][2]),
        pad(tg_reps, cols[6][1], cols[6][2]),
        pad(gen_step_str, cols[7][1], cols[7][2]),
    ]) + " |"

    print(row_pp)
    print(row_tg)
    print(row_ppG)
    print(row_tgG)

    print()
    print("Notes: t/s is mean ± 95% CI across repetitions. Synthetic random prompt tokens used.")
    print("pp=prompt processing, tg=token generation, ppG=prompt processing via Generator, tgG=token generation via Generator")


if __name__ == "__main__":
    main()
