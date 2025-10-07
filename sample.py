import argparse
import sys
import torch
from torch import nn, tensor
import tiktoken
from models import get_model_class
from inference.generate import Generator
from tools.checkpoint import load_checkpoint, apply_model_state

VOCAB_SIZE = 50257
MAX_SEQ_LEN = 16*1024

# Command line interface
parser = argparse.ArgumentParser(description="Generate text with a GPT model from a checkpoint.")
parser.add_argument("checkpoint", type=str, help="Path to model checkpoint (.pt)")
parser.add_argument("--max_tokens", type=int, default=256, help="Number of new tokens to generate")
parser.add_argument("--repetition_penalty", type=float, default=1.10, help="Repetition penalty")
parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
parser.add_argument("--top_k", type=int, default=100, help="Top-k sampling")
parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN, help="Maximum sequence length")
parser.add_argument("--seed", type=int, default=None, help="Random seed for deterministic sampling")
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="Device to run on: e.g., 'cpu', 'cuda', 'cuda:0'"
)
parser.add_argument("--chat", action="store_true", help="Start an interactive turn-based CLI chat")

if len(sys.argv) == 1:
    parser.print_usage()
    print("\nExample: python sample.py /path/to/checkpoint.pt --device cuda --max_tokens 100 --temperature 0.7 --top_k 50 --repetition_penalty 1.15")
    sys.exit(1)

cli = parser.parse_args()

device = cli.device
ckpt = load_checkpoint(cli.checkpoint, map_location=device)
state_dict = ckpt.model
hparams = ckpt.hparams or {}

vocab_size = int(hparams.get('vocab_size'))
num_layers = int(hparams.get('num_layers'))
num_heads = int(hparams.get('num_heads'))
model_dim = int(hparams.get('model_dim'))
head_dim = int(hparams.get('head_dim'))
max_seq_len = int(hparams.get('max_seq_len', cli.max_seq_len))
window_block_size = int(hparams.get('window_block_size', 128))
model_class = str(hparams.get('model_class'))
if not model_class:
    raise ValueError("Checkpoint hparams must include 'model_class' (fully-qualified class name)")
if 'eos_token_id' not in hparams:
    raise ValueError("Checkpoint hparams must include 'eos_token_id' (no default)")
eos_token_id = int(hparams.get('eos_token_id'))

ModelClass = get_model_class(model_class)
model: nn.Module = ModelClass(
    vocab_size=vocab_size,
    num_layers=num_layers,
    num_heads=num_heads,
    model_dim=model_dim,
    max_seq_len=max_seq_len,
    head_dim=head_dim,
    window_block_size=window_block_size,
    eos_token_id=eos_token_id,
).to(device)

apply_model_state(model, state_dict, strict=False)

model.eval()
for m in model.modules():
    if isinstance(m, nn.Embedding):
        m.bfloat16()
if device != 'cpu':
    model = torch.compile(model, dynamic=True)
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

# for the instruction tuned checkpoint the prompt should follow this format
'''
### Instruction:
{your prompt}

### Response:
'''
template = "### Instruction:\n{prompt}\n\n### Response:\n"

devtype = "cuda" if str(device).startswith("cuda") else ("mps" if str(device).startswith("mps") else "cpu")
ctx = torch.amp.autocast(device_type=devtype, dtype=torch.bfloat16)

gen = Generator(
    model=model,
    window=int(hparams['train_attention_window_len']),
    eos_token_id=eos_token_id,
    temperature=cli.temperature,
    top_k=cli.top_k,
    top_p=cli.top_p,
    repetition_penalty=cli.repetition_penalty,
    seed=cli.seed,
)

if cli.chat:
    print("Starting turn-based chat. Type 'exit', 'quit', or press Ctrl-D/Ctrl-C to end.\n")
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

        prompt_text = transcript + template.format(prompt=user)
        start_ids = encode(prompt_text)
        x = tensor(start_ids, dtype=torch.long, device=device)
        with torch.no_grad():
            with ctx:
                gen.reset()
                out_ids = gen.generate(x, max_new_tokens=cli.max_tokens)
        new_ids = out_ids[len(start_ids):]
        reply = decode(new_ids).strip()
        print(f"Assistant: {reply}\n")

        transcript = prompt_text + reply + "\n\n"
    sys.exit(0)
else:
    # Single-shot sample
    prompt = template.format(prompt="Write a short story about a child who is playing with a ball.")
    start_ids = encode(prompt)
    x = tensor(start_ids, dtype=torch.long, device=device)
    with torch.no_grad():
        with ctx:
            out_ids = gen.generate(x, max_new_tokens=cli.max_tokens)
    print(decode(out_ids))
