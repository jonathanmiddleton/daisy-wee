import itertools

import torch
from pathlib import Path
import numpy as np

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

class DistributedDataGenerator:
    def __init__(self, filename_pattern: str, batch_size: int, rank: int, world_size: int, start_shard: int | None = None, device: str = "cuda"):
        # Resolve files matching the pattern; support absolute and relative paths
        if filename_pattern.startswith("/"):
            p = Path(filename_pattern)
            files = sorted(list(p.parent.glob(p.name)))
        else:
            files = sorted(Path.cwd().glob(filename_pattern))
        if not files:
            raise FileNotFoundError(f"No files matched pattern: {filename_pattern}")
        assert batch_size % world_size == 0

        self.files: list[Path] = files
        self.batch_size = int(batch_size)
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.local_batch_size = self.batch_size // self.world_size
        self.device = device

        # Determine starting file index using a 1-based shard number with wrap-around safety
        if start_shard is not None:
            # Convert to zero-based index; allow any integer and wrap within [0, len(files))
            start_idx = (int(start_shard) - 1) % len(self.files)
        else:
            start_idx = 0

        # Create a cycling iterator starting from the chosen file
        files_ordered = self.files[start_idx:] + self.files[:start_idx]
        self._files_ordered = files_ordered
        self._file_iter = itertools.cycle(files_ordered)
        self._current_file: Path = next(self._file_iter)
        self._tokens: torch.Tensor = _load_data_shard(self._current_file)
        self._pos: int = 0

    def reset(self):
        """Reset the generator to the beginning of the current file ordering."""
        self._file_iter = itertools.cycle(self._files_ordered)
        self._current_file = next(self._file_iter)
        self._tokens = _load_data_shard(self._current_file)
        self._pos = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._pos + self.batch_size + 1 >= len(self._tokens):
            print(f"Current file: {self._current_file}")
            self._current_file = next(self._file_iter)
            self._tokens = _load_data_shard(self._current_file)
            self._pos = 0
        start = self._pos + self.rank * self.local_batch_size
        buf = self._tokens[start:][: self.local_batch_size + 1]
        inputs = buf[:-1].to(device=self.device, dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device=self.device, dtype=torch.int64, non_blocking=True)
        self._pos += self.batch_size
        return inputs, targets


def distributed_data_generator(filename_pattern: str, batch_size: int, rank: int, world_size: int, start_shard: int | None = None):
    """Backward-compatible generator that delegates to DistributedDataGenerator."""
    return iter(DistributedDataGenerator(filename_pattern, batch_size, rank, world_size, start_shard=start_shard))




