from pathlib import Path
import numpy as np
import argparse

def trim_shard(shard_in: Path, shard_out: Path, num_tokens: int):
    header = np.memmap(shard_in, mode="r", dtype=np.int32, shape=(256,), offset=0)
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    count = int(header[2])
    assert count >= num_tokens, f"Shard {shard_in} is smaller than {num_tokens} tokens"

    new_header = np.zeros(256, dtype=np.int32)
    new_header[0] = header[0]  # magic
    new_header[1] = header[1]  # version
    new_header[2] = num_tokens  # number of tokens after the 256*4 bytes of header (each 2 bytes as uint16)
    with shard_out.open("wb", buffering=0) as f_out:
        f_out.write(new_header.tobytes())
        tok_in = np.memmap(shard_in, mode="r", dtype=np.uint16, shape=(num_tokens,), offset=4*256)
        f_out.write(tok_in.tobytes())
        del tok_in


parser = argparse.ArgumentParser(description="Trim shard to a given number of tokens.")
parser.add_argument("in_shard", type=str, help="Path to input shard")
parser.add_argument("out_shard", type=str, help="Path to output shard")
parser.add_argument("-t", "--tokens", type=int, help="Target number of tokens in the output shard")
args = parser.parse_args()

trim_shard(Path(args.in_shard), Path(args.out_shard), args.tokens)