import argparse
import sys
import time

import tiktoken
import torch
from torch import nn, tensor

from inference.generate import Generator
from tools.checkpoint import model_from_checkpoint

VOCAB_SIZE = 50257
MAX_SEQ_LEN = 16*1024

# Command line interface
parser = argparse.ArgumentParser(description="Generation performance for a checkpoint.")
parser.add_argument("checkpoint", type=str, help="Path to model checkpoint (.pt)")
parser.add_argument("-pl", "--prompt_length", type=int, default=1024, help="Prompt length in tokens, inclusive of chat template")
parser.add_argument("-t", "--temperature", type=float, default=0.7, help="Sampling temperature")
parser.add_argument("--top_k", type=int, default=100, help="Top-k sampling")
parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
parser.add_argument("-rp", "--repetition_penalty", type=float, default=1.25, help="Repetition penalty")
parser.add_argument("-s", "--seed", type=int, default=1337, help="Random seed for deterministic sampling")
parser.add_argument("--max_tokens", type=int, default=256, help="Number of new tokens to generate")

parser.add_argument(
    "-d",
    "--device",
    type=str,
    default="cpu",
    help="Device to run on: e.g., 'cpu', 'cuda', 'cuda:0'"
)

if len(sys.argv) == 1:
    parser.print_usage()
    print("\nExample: python perf.py /path/to/checkpoint.pt --device cuda --max_tokens 100 --temperature 0.7 --top_k 50 --repetition_penalty 1.15")
    sys.exit(1)

cli = parser.parse_args()

device = cli.device
model, hparams = model_from_checkpoint(cli.checkpoint, device=device)
model.eval()
for m in model.modules():
    if isinstance(m, nn.Embedding):
        m.bfloat16()
if device != 'cpu':
    model = torch.compile(model, dynamic=True)
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

prompt_prefix = "### Instruction:\n Complete the following story which begins:\n"
prompt_suffix = "\n\n### Response:\n"

with open("../data/the_time_machine.txt") as f:
    text = f.read()
    approx_chars = cli.prompt_length*5 # ~4 for gpt2
    raw_user = text[:approx_chars]
    enc_prefix = encode(prompt_prefix)
    enc_suffix = encode(prompt_suffix)
    enc_user = encode(raw_user)[:cli.prompt_length - len(enc_prefix) - len(enc_suffix)]
    prompt = enc_prefix + enc_user + enc_suffix
    X = tensor(prompt).to(device)

gen = Generator(
    model=model,
    window=int(hparams['train_attention_window_len']),
    eos_token_id=hparams['eos_token_id'],
    temperature=cli.temperature,
    top_k=cli.top_k,
    top_p=cli.top_p,
    repetition_penalty=cli.repetition_penalty,
    seed=cli.seed,
    device=device,
)


import contextlib

@contextlib.contextmanager
def cuda_sync():
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()
    yield
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()

# Warmup to trigger compilation / kernels
warmup_iter = gen.generate(X, max_new_tokens=16)
try:
    while True:
        next(warmup_iter)
except StopIteration:
    pass

N = cli.max_tokens
gen.reset()
it = gen.generate(X, max_new_tokens=N)
with cuda_sync():
    t0 = time.time()
    try:
        while True:
            _ = next(it)
    except StopIteration as e:
        out_ids, pre_time, step_time = e.value
    t1 = time.time()
new_ids = out_ids[len(prompt):]
pre_tps = X.size(0) / pre_time
step_tps = len(out_ids) - X.size(0) / step_time
print(f"Input token length: {X.size(0)}")
print(f"Output token length: {len(out_ids)}")
print(f"{pre_tps:.1f} prefill tokens/s")
print(f"{step_tps:.1f} tokens/s\n")

if len(prompt) > 50:
    print("Input prompt (trimmed):")
    print(f"{decode(prompt[:25])}...{decode(prompt[-25:])}")
else:
    print("Input prompt:")
    print(f"{decode(prompt)}")

print(f"Output: {decode(new_ids)}")

