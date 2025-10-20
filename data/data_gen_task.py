import os, math, numpy as np, torch, itertools, json
from pathlib import Path

class _Shard:
    def __init__(self, d: Path):
        self.dir = Path(d)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        assert self.meta["magic"] == 20240520
        self.tokens = np.load(self.dir / "tokens.npy", mmap_mode="r")
        self.labels = np.load(self.dir / "labels.npy", mmap_mode="r")
        self.offsets = np.load(self.dir / "offsets.npy", mmap_mode="r")
        assert self.tokens.shape[0] == self.labels.shape[0] == int(self.offsets[-1])
        assert self.meta["version"] >= 2

    def __len__(self): return len(self.offsets) - 1

    def get(self, i: int):
        s, e = int(self.offsets[i]), int(self.offsets[i+1])
        return self.tokens[s:e], self.labels[s:e]

def _pad(batch):
    L = [len(x[0]) for x in batch]
    T = max(L)
    B = len(batch)
    x = torch.full((B, T), 0, dtype=torch.int32)
    y = torch.full((B, T), -100, dtype=torch.int64)
    m = torch.zeros((B, T), dtype=torch.bool)
    for i, (inp, lab) in enumerate(batch):
        t = len(inp)
        x[i, :t] = torch.from_numpy(inp.astype(np.int32, copy=False))
        y[i, :t] = torch.from_numpy(lab.astype(np.int64, copy=False))
        m[i, :t] = True
    return x, y, m

class TaskDataGenerator:
    def __init__(self, root: str, split: str, batch_size: int, world_size: int = 1, rank: int = 0, seed: int = 1337, device: str = "cpu", start_shard: int | None = None, drop_remainder: bool = False, infinite: bool = True):
        p = Path(root) / split
        self.files = sorted([d for d in p.iterdir() if d.is_dir() and (d / "meta.json").exists()])
        if not self.files: raise FileNotFoundError(f"no shards in {p}")
        assert batch_size % world_size == 0
        self.local_bsz = batch_size // world_size
        self.rank = int(rank)
        self.world = int(world_size)
        self.seed = int(seed)
        self.device = torch.device(device)
        self.drop_remainder = drop_remainder
        self.infinite = infinite
        i0 = (start_shard or 0) % len(self.files)
        self._file_iter = itertools.cycle(self.files[i0:] + self.files[:i0])
        self._rng = np.random.default_rng(self.seed)
        self._shard = None
        self._order = None
        self._pos = 0
        # assert meta["eos_id"] == model.config.eos_token_id
        # assert meta["tokenizer_len"] == model.get_input_embeddings().weight.shape[0]

    def _load_next(self):
        d = next(self._file_iter)
        self._shard = _Shard(d)
        n = len(self._shard)
        self._rng = np.random.default_rng(self.seed ^ hash(d.name) & 0xFFFFFFFF)
        self._order = self._rng.permutation(n).tolist()
        self._pos = 0

    def __iter__(self): return self

    def __next__(self):
        if self._shard is None: self._load_next()
        b = []
        need = self.local_bsz
        while need > 0:
            if self._pos >= len(self._order):
                if self.drop_remainder and b: break
                self._load_next()
                if not self.infinite and not b: raise StopIteration
            if self._pos >= len(self._order): continue
            idx = self._order[self._pos]; self._pos += 1
            if (self._pos - 1) % self.world != self.rank: continue
            b.append(self._shard.get(idx))
            need -= 1
        if len(b) < self.local_bsz and self.drop_remainder: raise StopIteration
        x, y, m = _pad(b)
        non_blocking = self.device.type == "cuda" and torch.cuda.is_available()
        # TODO mask before returning train/targets
        return x.to(self.device, non_blocking=non_blocking), y.to(self.device, non_blocking=non_blocking), m.to(self.device, non_blocking=non_blocking)