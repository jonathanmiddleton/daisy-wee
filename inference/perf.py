import argparse
import sys
import time
import itertools
from dataclasses import dataclass

import tiktoken
import torch
from torch import nn, tensor

from inference.generate import Generator
from tools.checkpoint import model_from_checkpoint
from tools.helpers import _coerce_value

VOCAB_SIZE = 50257
MAX_SEQ_LEN = 16*1024

# Command line interface
parser = argparse.ArgumentParser(description="Generation performance for a checkpoint.")
parser.add_argument("checkpoint", type=str, help="Path to model checkpoint (.pt)")
# Accept comma-separated values by parsing as strings and coercing later
parser.add_argument("-pl", "--prompt_length", type=str, default="1024", help="Prompt length in tokens, inclusive of chat template (allow comma-separated)")
parser.add_argument("-t", "--temperature", type=str, default="0.7", help="Sampling temperature (allow comma-separated)")
parser.add_argument("--top_k", type=str, default="100", help="Top-k sampling (allow comma-separated)")
parser.add_argument("--top_p", type=str, default="0.95", help="Top-p sampling (allow comma-separated)")
parser.add_argument("-rp", "--repetition_penalty", type=str, default="1.25", help="Repetition penalty (allow comma-separated)")
parser.add_argument("-s", "--seed", type=str, default="1337", help="Random seed for deterministic sampling (allow comma-separated)")
parser.add_argument("--max_tokens", type=str, default="256", help="Number of new tokens to generate (allow comma-separated)")

parser.add_argument(
    "-d",
    "--device",
    type=str,
    default="cpu",
    help="Device to run on: e.g., 'cpu', 'cuda', 'cuda:0' (allow comma-separated)"
)

if len(sys.argv) == 1:
    parser.print_usage()
    print("\nExample: python perf.py /path/to/checkpoint.pt --device cuda --max_tokens 100 --temperature 0.7 --top_k 50 --repetition_penalty 1.15")
    sys.exit(1)

cli = parser.parse_args()

# Helper: parse comma-separated values into a list of proper types
TYPES = {
    'device': str,
    'prompt_length': int,
    'temperature': float,
    'top_k': int,
    'top_p': float,
    'repetition_penalty': float,
    'seed': int,
    'max_tokens': int,
}

def parse_values(val, typ):
    if isinstance(val, (int, float)):
        return [val]
    if isinstance(val, str):
        parts = [p.strip() for p in val.split(',')] if (',' in val) else [val.strip()]
        out = []
        for p in parts:
            if p == '':
                continue
            out.append(_coerce_value(p, typ))
        return out if len(out) > 0 else [None]
    # Fallback
    return [val]

arg_values = {name: parse_values(getattr(cli, name), typ) for name, typ in TYPES.items()}

multi_mode = any(len(v) > 1 for v in arg_values.values())

enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

prompt_prefix = "### Instruction:\n Complete the following story which begins:\n"
prompt_suffix = "\n\n### Response:\n"

with open("../data/the_time_machine.txt") as f:
    TEXT = f.read()

import contextlib

@contextlib.contextmanager
def cuda_sync(dev):
    if torch.cuda.is_available() and str(dev).startswith("cuda"):
        torch.cuda.synchronize()
    yield
    if torch.cuda.is_available() and str(dev).startswith("cuda"):
        torch.cuda.synchronize()

@dataclass
class Result:
    device: str
    prompt_length: int
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float
    seed: int
    max_tokens: int
    input_len: int
    output_len: int
    prefill_tps: float
    step_tps: float
    total_time: float

# Prepare all combinations (Cartesian product)
keys = list(TYPES.keys())
vals_lists = [arg_values[k] for k in keys]
combinations = [dict(zip(keys, prod)) for prod in itertools.product(*vals_lists)]

# Execute runs, reusing models per device
last_device = None
model = None
hparams = None
results: list[Result] = []

for cfg in combinations:
    device = cfg['device']
    prompt_len = int(cfg['prompt_length'])
    temperature = float(cfg['temperature'])
    top_k = int(cfg['top_k']) if cfg['top_k'] is not None else None
    top_p = float(cfg['top_p']) if cfg['top_p'] is not None else None
    repetition_penalty = float(cfg['repetition_penalty'])
    seed = int(cfg['seed']) if cfg['seed'] is not None else None
    max_tokens = int(cfg['max_tokens'])

    if last_device != device or model is None:
        # Load model for this device
        model, hparams = model_from_checkpoint(cli.checkpoint, device=device)
        model.eval()
        for m in model.modules():
            if isinstance(m, nn.Embedding):
                m.bfloat16()
        if device != 'cpu':
            model = torch.compile(model, dynamic=True)
        last_device = device

    # Build prompt for this prompt length and device
    approx_chars = prompt_len * 5  # ~4 for gpt2
    raw_user = TEXT[:approx_chars]
    enc_prefix = encode(prompt_prefix)
    enc_suffix = encode(prompt_suffix)
    enc_user = encode(raw_user)[:prompt_len - len(enc_prefix) - len(enc_suffix)]
    prompt = enc_prefix + enc_user + enc_suffix
    X = tensor(prompt, dtype=torch.long).to(device)

    gen = Generator(
        model=model,
        window=int(hparams['train_attention_window_len']),
        eos_token_id=hparams['eos_token_id'],
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        seed=seed,
        device=device,
    )

    # Warmup to trigger compilation / kernels
    warmup_iter = gen.generate(X, max_new_tokens=16)
    try:
        while True:
            next(warmup_iter)
    except StopIteration:
        pass

    gen.reset()
    it = gen.generate(X, max_new_tokens=max_tokens)
    with cuda_sync(device):
        t0 = time.time()
        try:
            while True:
                _ = next(it)
        except StopIteration as e:
            out_ids, pre_time, step_time = e.value
        t1 = time.time()

    input_len = X.size(0)
    output_len = len(out_ids)
    new_tokens = output_len - input_len
    pre_tps = (input_len / pre_time) if pre_time > 0 else float('inf')
    step_tps = (new_tokens / step_time) if step_time > 0 else float('inf')

    results.append(Result(
        device=str(device),
        prompt_length=prompt_len,
        temperature=temperature,
        top_k=top_k if top_k is not None else 0,
        top_p=top_p if top_p is not None else 0.0,
        repetition_penalty=repetition_penalty,
        seed=seed if seed is not None else 0,
        max_tokens=max_tokens,
        input_len=input_len,
        output_len=output_len,
        prefill_tps=pre_tps,
        step_tps=step_tps,
        total_time=(t1 - t0),
    ))

# Output
if not multi_mode and len(results) == 1:
    r = results[0]
    # Mirror the original detailed output for single run
    print(f"Input token length: {r.input_len}")
    print(f"Output token length: {r.output_len}")
    print(f"{r.prefill_tps:.1f} prefill tokens/s")
    print(f"{r.step_tps:.1f} tokens/s\n")

    # Rebuild the same prompt to print input/output
    prompt_len = results[0].prompt_length
    approx_chars = prompt_len * 5
    raw_user = TEXT[:approx_chars]
    enc_prefix = encode(prompt_prefix)
    enc_suffix = encode(prompt_suffix)
    enc_user = encode(raw_user)[:prompt_len - len(enc_prefix) - len(enc_suffix)]
    prompt = enc_prefix + enc_user + enc_suffix
    new_ids = out_ids[len(prompt):]  # from last run scope
    if len(prompt) > 50:
        print("Input prompt (trimmed):")
        print(f"{decode(prompt[:25])}...{decode(prompt[-25:])}")
    else:
        print("Input prompt:")
        print(f"{decode(prompt)}")
    print(f"Output: {decode(new_ids)}")
else:
    # Tabular summary for multiple runs
    cols = [
        ("device", 10, lambda r: r.device),
        ("pl", 6, lambda r: str(r.prompt_length)),
        ("t", 6, lambda r: f"{r.temperature:g}"),
        ("top_k", 7, lambda r: str(r.top_k)),
        ("top_p", 8, lambda r: f"{r.top_p:.2f}"),
        ("rp", 6, lambda r: f"{r.repetition_penalty:.2f}"),
        ("seed", 6, lambda r: str(r.seed)),
        ("max_new", 8, lambda r: str(r.max_tokens)),
        ("in_len", 7, lambda r: str(r.input_len)),
        ("out_len", 8, lambda r: str(r.output_len)),
        ("prefill_tps", 13, lambda r: f"{r.prefill_tps:8.1f}"),
        ("step_tps", 10, lambda r: f"{r.step_tps:8.1f}"),
        ("time_s", 8, lambda r: f"{r.total_time:6.2f}"),
    ]
    header = " ".join(name.ljust(w) for name, w, _ in cols)
    print(header)
    print("-" * len(header))
    for r in results:
        line = " ".join(str(fn(r)).ljust(w)[:w] for name, w, fn in cols)
        print(line)

