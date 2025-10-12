import time
from dataclasses import dataclass
from typing import Optional, Dict

import torch
import torch.distributed as dist
from torch import nn

from training.data_gen import DistributedDataGenerator
from training.optim import get_num_window_blocks


@dataclass
class EvalResult:
    val_loss: float
    val_acc: Optional[float]
    epoch: Optional[int]
    ema_dloss_per_token: Optional[float]


class Evaluator:
    def __init__(
        self,
        *,
        data_generator: DistributedDataGenerator,
        distributed_enabled: bool | None = None,
        world_size: int | None = None,
        rank: int | None = None,
        train_attention_window_len: int,
        window_block_size: int,
    ) -> None:
        self._ddg = data_generator
        self._use_dist = bool(distributed_enabled) if distributed_enabled is not None else dist.is_available() and dist.is_initialized()
        self._rank = int(rank or 0)
        self._world_size = int(world_size or 1)
        self._tawt = int(train_attention_window_len)
        self._wbs = int(window_block_size)
        # Track EMA of dloss/token between eval calls
        self._last_val_loss: Optional[float] = None
        self._last_tokens_seen: int = 0
        self._ema_dloss_per_token: Optional[float] = None


    def reset_generator(self) -> None:
        """
        Resets the internal DistributedDataGenerator so that it generates data from the beginning.
        """
        # If the generator exposes a reset() method, call it; otherwise, reinitialize internal state.
        if hasattr(self._ddg, "reset"):
            self._ddg.reset()
            return
        # Fallback: best-effort reset to start of current file order
        import itertools
        self._ddg._file_iter = itertools.cycle(self._ddg.files)  # type: ignore[attr-defined]
        self._ddg._current_file = next(self._ddg._file_iter)  # type: ignore[attr-defined]
        from training.data_gen import _load_data_shard  # type: ignore
        self._ddg._tokens = _load_data_shard(self._ddg._current_file)  # type: ignore[attr-defined]
        self._ddg._pos = 0  # type: ignore[attr-defined]
        # Also reset internal counters used for ema calc
        self._last_tokens_seen = 0

    def eval(self,*, model: nn.Module, total_tokens: int) -> Dict[str, Optional[float]]:
        """
        Evaluate the model on validation data.

        Args:
            model: the model to evaluate. Must be in eval() mode, otherwise raises an error.
            total_tokens: the total number of tokens globally over which to evaluate.
        Returns:
            dict with keys: val_loss, val_acc, epoch, ema_dloss_per_token
        """
        with torch.no_grad():
            if model.training:
                raise RuntimeError("Evaluator.eval() requires model.eval() mode; got training mode.")

            # Determine steps per rank
            world_batch_size = int(self._ddg.batch_size)
            if total_tokens % world_batch_size != 0:
                raise ValueError(f"total_tokens ({total_tokens}) must be divisible by world_batch_size ({world_batch_size})")
            steps = total_tokens // world_batch_size

            if self._use_dist:
                dist.barrier()

            t0 = time.perf_counter()
            device = next(model.parameters()).device
            loss_acc = torch.zeros((), device=device, dtype=torch.float32)
            interval = max(1, steps // 10)
            for i in range(steps):
                inputs, targets = next(self._ddg)
                # bugfix: drop any partial tail across both tensors
                n_in = len(inputs)
                n_tg = len(targets)
                n = min(n_in, n_tg)
                cut = n - (n % self._wbs)
                if cut == 0:
                    # still advance silently; optionally emit rare heartbeat
                    if self._rank == 0 and (i % interval == 0 or i == steps - 1):
                        print(f"[eval] step {i+1}/{steps}: skipped (insufficient tokens in shard)" )
                    continue
                if n_in != cut:
                    inputs = inputs[:cut]
                if n_tg != cut:
                    targets = targets[:cut]
                # Match training eval: use window schedule with s=1.0 (full windows) for stability
                loss_acc = loss_acc + model(inputs, get_num_window_blocks(1.0, attention_window_len=self._tawt, window_block_size=self._wbs), targets)
                if self._rank == 0 and (i % interval == 0 or i == steps - 1):
                    print(f"[eval] step {i+1}/{steps} done")
            loss_acc = loss_acc / steps

            if self._use_dist:
                dist.all_reduce(loss_acc, op=dist.ReduceOp.AVG)

            cur_val = float(loss_acc.item())
            tokens_since_last = total_tokens
            if self._last_val_loss is not None and tokens_since_last > 0:
                dpt = (cur_val - self._last_val_loss) / tokens_since_last
                self._ema_dloss_per_token = dpt if self._ema_dloss_per_token is None else 0.7 * self._ema_dloss_per_token + 0.3 * dpt
            self._last_val_loss = cur_val
            self._last_tokens_seen += tokens_since_last

            # We don't track real epoch progression here; return None
            return {
                "val_loss": cur_val,
                "val_acc": None,
                "epoch": None,
                "ema_dloss_per_token": self._ema_dloss_per_token if self._ema_dloss_per_token is not None else float("nan"),
            }
