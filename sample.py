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
parser = argparse.ArgumentParser(description="Generate text with a GPT model from a checkpoint.")
parser.add_argument("checkpoint", type=str, help="Path to model checkpoint (.pt)")
parser.add_argument("--max_tokens", type=int, default=256, help="Number of new tokens to generate")
parser.add_argument("-rp", "--repetition_penalty", type=float, default=1.25, help="Repetition penalty")
parser.add_argument("-t", "--temperature", type=float, default=0.7, help="Sampling temperature")
parser.add_argument("--top_k", type=int, default=100, help="Top-k sampling")
parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
parser.add_argument("-s", "--seed", type=int, default=1337, help="Random seed for deterministic sampling")
parser.add_argument("--base", type=bool, default=False, help="Flag for base sampling")
parser.add_argument("-p", "--prompt", type=str, default="Write a short story about a child playing with a ball.", help="Optional one-shot prompt")
parser.add_argument(
    "-d",
    "--device",
    type=str,
    default="cpu",
    help="Device to run on: e.g., 'cpu', 'cuda', 'cuda:0'"
)
parser.add_argument("--chat", action="store_true", help="Start an interactive turn-based CLI chat")

if len(sys.argv) == 1:
    parser.print_usage()
    print("\nExample: python sample.py /path/to/checkpoint.pt --device cuda --max_tokens 100 --temperature 0.7 --top_k 50 --repetition_penalty 1.15")
    sys.exit(1)

cli = parser.parse_args()

device = cli.device
model, hparams = model_from_checkpoint(cli.checkpoint, device=device)
model.eval()

if device != 'cpu':
    model = torch.compile(model, dynamic=True)
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

use_instruct = not cli.base

# for the instruction tuned checkpoint the prompt should follow this format
'''
### Instruction:
{prompt}

### Response:
'''
template = "### Instruction:\n{prompt}\n\n### Response:\n" if use_instruct else "{prompt}"

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

def _parse_leading_params(s: str):
    s = s.lstrip()
    if not s:
        return {}, ""
    tokens = s.split()
    i = 0
    updates = {}
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith('/t='):
            try:
                updates['temperature'] = float(tok[3:])
            except ValueError:
                pass
            i += 1
            continue
        if tok.startswith('/rp='):
            try:
                updates['repetition_penalty'] = float(tok[4:])
            except ValueError:
                pass
            i += 1
            continue
        if tok.startswith('/new'):
            try:
                updates['new'] = True
            except ValueError:
                pass
            i += 1
            continue
        break
    remaining = " ".join(tokens[i:])
    return updates, remaining

print("Hyperparameters: temperature =", cli.temperature, ", repetition_penalty =", cli.repetition_penalty, ", top_k =", cli.top_k, ", top_p =", cli.top_p,
      ", max_seq_len =", hparams['max_seq_len'], ", seed =", cli.seed,)
def print_token(t):
    print(enc.decode([t]), end="", flush=True)

if cli.chat:
    print("Starting turn-based chat. Type 'exit', 'quit', or press Ctrl-D/Ctrl-C to end.")
    print("Tip: Adjust settings inline, e.g., '/t=0.4', '/rp=1.2', or '/t=0.4 /rp=1.2 write something'. Type '/new' to start a new conversation.\n")
    transcript = ""

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in {"exit", "quit", "q"}:
            break

        updates, remaining = _parse_leading_params(user)
        if 'temperature' in updates:
            gen.set_temperature(updates['temperature'])
        if 'repetition_penalty' in updates:
            gen.set_repetition_penalty(updates['repetition_penalty'])
        if 'new' in updates:
            transcript = ""
            gen.reset()
            print("\033[2J\033[H", end="", flush=True)  # CSI 2J = clear, CSI H = home
            continue
        # If the user only passed settings, acknowledge and reprompt
        if remaining.strip() == "" and len(updates) > 0:
            parts = []
            if 'temperature' in updates:
                parts.append(f"temperature={updates['temperature']}")
            if 'repetition_penalty' in updates:
                parts.append(f"repetition_penalty={updates['repetition_penalty']}")
            print("Updated settings: " + ", ".join(parts) + "\n")
            continue

        effective_user = remaining if len(updates) > 0 else user
        prompt_text = transcript + template.format(prompt=effective_user)
        start_ids = encode(prompt_text)
        X = tensor(start_ids, dtype=torch.long, device=device)
        gen_iter = gen.generate(X, max_new_tokens=cli.max_tokens)
        print(f"Assistant: ", end="", flush=True)
        sys.stdout.flush()

        try:
            while True:
                t = next(gen_iter)
                print_token(t)
        except StopIteration as e:
            out_ids, pre_time, step_time = e.value
        new_ids = out_ids[len(start_ids):]
        pre_tps = len(start_ids) / pre_time
        step_tps = len(new_ids) / step_time
        reply = decode(new_ids).strip()
        transcript = prompt_text + reply + "\n\n" #anything added to transcript becomes the prefix of the next prompt
        print(f"\n({step_tps:.1f} step tokens/s, {pre_tps:.1f} prefill tokens/s)\n\n")
    sys.exit(0)
else:
    # Single-shot sample
    prompt = template.format(prompt=cli.prompt)
    start_ids = encode(prompt)
    x = tensor(start_ids, dtype=torch.long, device=device)
    gen_iter = gen.generate(x, max_new_tokens=cli.max_tokens)
    try:
        while True:
            t = next(gen_iter)
            print_token(t)
    except StopIteration as e:
        (out_ids, pre_time, step_time) = e.value
    new_ids = out_ids[len(start_ids):]
    pre_tps = len(start_ids) / pre_time
    step_tps = len(new_ids) / step_time
    print(f"\n({step_tps:.1f} step tokens/s, {pre_tps:.1f} prefill tokens/s)\n\n")