import os, time, torch, torch.nn as nn
from itertools import product

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

assert torch.backends.mps.is_available()

B, T, D, H = 1, 1024, 1024, 8
MPS_ITERS_MULTIPLIER = 100

def bench(fn, dev, iters=50, warmup=10):
    for _ in range(warmup): fn()
    if dev == "mps":
        iters *= MPS_ITERS_MULTIPLIER
        torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    if dev == "mps":
        torch.mps.synchronize()
    return (time.perf_counter() - t0) / iters

def run_linear(dtype, dev, other_dtype=None):
    x = torch.randn(B*T, D, device=dev, dtype=dtype)
    lin = nn.Linear(D, D, bias=False, device=dev, dtype=dtype if other_dtype is None else other_dtype)
    return bench(lambda: lin(x), dev)

def run_mixed_linear(dtype, dev):
    other_dtype = torch.float16 if dtype == torch.bfloat16 else torch.bfloat16
    with torch.autocast(dev, dtype=dtype):
        return run_linear(dtype, dev, other_dtype)

def run_matmul(dtype, dev, other_dtype=None):
    a = torch.randn(D, D, device=dev, dtype=dtype)
    b = torch.randn(D, D, device=dev, dtype=dtype if other_dtype is None else other_dtype)
    return bench(lambda: a @ b, dev)

def run_mixed_matmul(dtype, dev):
    other_dtype = torch.float16 if dtype == torch.bfloat16 else torch.bfloat16
    with torch.autocast(dev, dtype=dtype):
        return run_matmul(dtype, dev, other_dtype)

def run_sdpa(dtype, dev, other_dtype=None):
    q = torch.randn(B, H, T, D//H, device=dev, dtype=dtype if other_dtype is None else other_dtype)
    k = torch.randn(B, H, T, D//H, device=dev, dtype=dtype)
    v = torch.randn(B, H, T, D//H, device=dev, dtype=dtype)
    return bench(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False), dev)

def run_mixed_sdpa(dtype, dev):
    other_dtype = torch.float16 if dtype == torch.bfloat16 else torch.bfloat16
    with torch.autocast(dev, dtype=dtype):
        return run_sdpa(dtype, dev, other_dtype=other_dtype)

cases = [("linear", run_linear),
         ("mixed_linear", run_mixed_linear),
         ("matmul", run_matmul),
         ("mixed_matmul", run_mixed_matmul),
         ("sdpa", run_sdpa),
         ("mixed_sdpa", run_mixed_sdpa),]
devs = ["mps"] #["mps", "cpu"]
dtypes = [torch.float16, torch.bfloat16]

print("Relatively meaningful rates:")
for name, runner in cases:
    for dev, dt in product(devs, dtypes):
            try:
                t = runner(dt, dev)
                print(f"[{dev}] {name:6s} {str(dt):>12s}: {t*1e3:.2f} ms/iter")
            except Exception as e:
                print(f"[{dev}] {name:6s} {str(dt):>12s}: ERROR -> {e}")




print("\n[Exponent-range underflow test (nonzero implies native BF16 ops)]")
dev="mps"
# Values far below FP16 subnormal range but well within BF16 range
x32 = torch.full((8192,), 1e-20, device=dev, dtype=torch.float32)
a16 = x32.to(torch.float16)
aBF = x32.to(torch.bfloat16)
y16 = (a16 * a16).to(torch.float32)
yBF = (aBF * aBF).to(torch.float32)
print("FP16 any nonzero:", (y16!=0).any().item(), "min:", y16[y16!=0].min().item() if (y16!=0).any() else 0.0)
print("BF16 any nonzero:", (yBF!=0).any().item(), "min:", yBF[yBF!=0].min().item() if (yBF!=0).any() else 0.0)

print("\n[Bitwise equality test]")
D=2048
A = torch.randn(D, D, device=dev, dtype=torch.float32)
B = torch.randn(D, D, device=dev, dtype=torch.float32)
C16 = (A.to(torch.float16) @ B.to(torch.float16)).to(torch.float32)
CBF = (A.to(torch.bfloat16) @ B.to(torch.bfloat16)).to(torch.float32)
print("bitwise_equal:", torch.equal(C16, CBF))
print("max_abs_diff:", (C16 - CBF).abs().max().item())