import torch
from pathlib import Path
import os, sys

from huggingface_hub import hf_hub_download


def create_overfit_sample( output_file: str, num_tokens: int = 1_000_000):
    """
    Create a smaller bin file with the specified number of tokens from an existing bin file with the Karpathy quasi-standard file format.

    Args:
        output_file: Path to the output bin file (e.g., "data/overfit/edu_fineweb_overfit_1M.bin")
        num_tokens: Number of tokens to extract (default: 1M)
    """
    file_name = "edu_fineweb_train_000001.bin"
    local_dir = os.path.join(os.path.dirname(__file__), 'fineweb-edu-overfit')
    def get(fname):
        if not os.path.exists(os.path.join(local_dir, fname)):
            hf_hub_download(repo_id="karpathy/fineweb-edu-100B-gpt2-token-shards", filename=fname,
                            repo_type="dataset", local_dir=local_dir)

    get(file_name)

    input_path = Path(local_dir) / file_name
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read the header from input file
    header = torch.from_file(str(input_path), False, 256, dtype=torch.int32, device='cpu')
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    original_num_tokens = int(header[2])

    print(f"Original file has {original_num_tokens:,} tokens")
    tokens_to_read = min(num_tokens, original_num_tokens)
    print(f"Extracting {tokens_to_read:,} tokens...")

    with input_path.open("rb", buffering=0) as f:
        tokens = torch.empty(tokens_to_read, dtype=torch.uint16)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * tokens_to_read, "number of tokens read does not match expected"

    new_header = torch.zeros(256, dtype=torch.int32)
    new_header[0] = 20240520  # magic number
    new_header[1] = 1  # version
    new_header[2] = tokens_to_read

    print(f"Writing to {output_path}...")
    with output_path.open("wb") as f:
        f.write(new_header.numpy().tobytes())
        f.write(tokens.numpy().tobytes())

    print(f"Done! Created {output_path} with {tokens_to_read:,} tokens")
    print(f"File size: {output_path.stat().st_size:,} bytes")


if __name__ == "__main__":
    create_overfit_sample(
        output_file="data/overfit/edu_fineweb_overfit_1M.bin",
        num_tokens=1_000_000
    )