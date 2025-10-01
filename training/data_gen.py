import itertools

import torch
from pathlib import Path

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, rank: int, world_size: int, start_shard: int | None = None):
    if filename_pattern.startswith("/"):
        p = Path(filename_pattern)
        files = sorted(list(p.parent.glob(p.name)))
    else:
        files = sorted(Path.cwd().glob(filename_pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {filename_pattern}")
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size

    # Determine starting file index using a 1-based shard number with wrap-around safety
    if start_shard is not None:
        # Convert to zero-based index; allow any integer and wrap within [0, len(files))
        start_idx = (int(start_shard) - 1) % len(files)
    else:
        start_idx = 0

    # Create a cycling iterator starting from the chosen file
    files_ordered = files[start_idx:] + files[:start_idx]
    file_iter = itertools.cycle(files_ordered)
    current_file = next(file_iter)
    tokens, pos = _load_data_shard(current_file), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            print(f"Current file: {current_file}")
            current_file = next(file_iter)
            tokens, pos = _load_data_shard(current_file), 0
        buf = tokens[pos + rank * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)  # no sync on host side;
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)  # H2D in another stream isn't helpful.
        pos += batch_size
        yield inputs, targets