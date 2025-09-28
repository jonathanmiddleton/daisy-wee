import argparse
import sys
import torch
from torch import nn, load, tensor
import tiktoken
from models.gpt_core import GPTCore
from inference.generate import Generator

VOCAB_SIZE = 50257
TRAIN_SEQ_LEN = 64*1024

# Command line interface
parser = argparse.ArgumentParser(description="Generate text with a GPT model from a checkpoint.")
parser.add_argument("checkpoint", type=str, help="Path to model checkpoint (.pt)")
parser.add_argument("--max_tokens", type=int, default=100, help="Number of new tokens to generate")
parser.add_argument("--repetition_penalty", type=float, default=1.05, help="Repetition penalty")
parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
parser.add_argument("--top_k", type=int, default=100, help="Top-k sampling")
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="Device to run on: e.g., 'cpu', 'cuda', 'cuda:0'"
)

if len(sys.argv) == 1:
    parser.print_usage()
    print("\nExample: python sample.py /path/to/checkpoint.pt --device cuda --max_tokens 100 --temperature 0.7 --top_k 50 --repetition_penalty 1.15")
    sys.exit(1)

cli = parser.parse_args()

device = cli.device
checkpoint = load(cli.checkpoint, map_location=device)
model: nn.Module = GPTCore(vocab_size=VOCAB_SIZE, num_layers=16, num_heads=8, model_dim=1024,
                       max_seq_len=TRAIN_SEQ_LEN).to(device)
state_dict = checkpoint['model']

unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
if device != 'cpu':
    model.to(device)
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
prompt = template.format(prompt="Write a short story about a child who is playing with a ball.")

start_ids = encode(prompt)
x = tensor(start_ids, dtype=torch.long, device=device)[None, ...]

devtype = "cuda" if str(device).startswith("cuda") else ("mps" if str(device).startswith("mps") else "cpu")
ctx = torch.amp.autocast(device_type=devtype, dtype=torch.bfloat16)
with torch.no_grad():
    with ctx:
        gen = Generator(model=model, window=4096, eos_token_id=50256, temperature=0.7, top_p=0.9,
                        repetition_penalty=1.1)
        tokens = gen.generate(x, max_new_tokens=256)

        print(decode(tokens[0].tolist()))