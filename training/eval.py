import math
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
        wandb_enabled: bool,
        data_generator: DistributedDataGenerator,
        distributed_enabled: bool | None = None,
        rank: int | None = None,
        attention_window_len: int,
        window_block_size: int,
    ) -> None:
        self._wandb_enabled = bool(wandb_enabled)
        self._ddg = data_generator
        self._use_dist = bool(distributed_enabled) if distributed_enabled is not None else dist.is_available() and dist.is_initialized()
        self._rank = int(rank or 0)
        self._awt = int(attention_window_len)
        self._wbs = int(window_block_size)
        # Track EMA of dloss/token between eval calls
        self._last_val_loss: Optional[float] = None
        self._last_tokens_seen: int = 0
        self._ema_dloss_per_token: Optional[float] = None
        # attempt to import wandb only if enabled
        self._wandb = None
        if self._wandb_enabled:
            try:
                import wandb  # type: ignore
                self._wandb = wandb
            except Exception:
                # silently disable if import fails
                self._wandb_enabled = False
                self._wandb = None

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

    @torch.no_grad()
    def eval(self, model: nn.Module, num_tokens_per_rank: int, tokens: Optional[int] = None) -> Dict[str, Optional[float]]:
        """
        Evaluate the model on validation data.

        Args:
            model: the model to evaluate. Must be in eval() mode, otherwise raises an error.
            num_tokens_per_rank: number of tokens per rank to evaluate.
            tokens: Optional global tokens counter to log to wandb (e.g., training tokens processed).
        Returns:
            dict with keys: val_loss, val_acc, epoch, ema_dloss_per_token
        """
        if model.training:
            raise RuntimeError("Evaluator.eval() requires model.eval() mode; got training mode.")

        # Determine steps per rank
        local_seq_len = int(self._ddg.local_batch_size)
        if num_tokens_per_rank % local_seq_len != 0:
            raise ValueError(
                f"num_tokens_per_rank ({num_tokens_per_rank}) must be divisible by local sequence length ({local_seq_len})"
            )
        steps = num_tokens_per_rank // local_seq_len

        # Reduce barriers if distributed
        if self._use_dist:
            dist.barrier()

        t0 = time.perf_counter()
        loss_acc = torch.zeros((), device="cuda", dtype=torch.float32)
        for _ in range(steps):
            inputs, targets = next(self._ddg)
            # Match training eval: use window schedule with s=1.0 (full windows) for stability
            loss_acc = loss_acc + model(inputs, get_num_window_blocks(1.0, attention_window_len=self._awt, window_block_size=self._wbs), targets)
        loss_acc = loss_acc / steps

        if self._use_dist:
            dist.all_reduce(loss_acc, op=dist.ReduceOp.AVG)

        cur_val = float(loss_acc.item())
        tokens_since_last = num_tokens_per_rank  # best available proxy in this isolated evaluator
        if self._last_val_loss is not None and tokens_since_last > 0:
            dpt = (cur_val - self._last_val_loss) / tokens_since_last
            self._ema_dloss_per_token = dpt if self._ema_dloss_per_token is None else 0.7 * self._ema_dloss_per_token + 0.3 * dpt
        self._last_val_loss = cur_val
        self._last_tokens_seen += num_tokens_per_rank

        # Logging to wandb (parity with train.py: log val/loss and val/ppl)
        if self._wandb_enabled and self._wandb is not None and self._rank == 0:
            try:
                # Prefer externally provided global tokens; otherwise, fallback to cumulative eval tokens
                if tokens is not None:
                    _tokens_to_log = int(tokens)
                else:
                    _ws = dist.get_world_size() if (self._use_dist and dist.is_initialized()) else 1
                    _tokens_to_log = int(self._last_tokens_seen * _ws)
                self._wandb.log({
                    "val/loss": cur_val,
                    "val/ppl": math.exp(cur_val) if cur_val < 20 else float("inf"),
                    "tokens": _tokens_to_log,
                })
            except Exception:
                pass

        # We don't track real epoch progression here; return None
        return {
            "val_loss": cur_val,
            "val_acc": None,
            "epoch": None,
            "ema_dloss_per_token": self._ema_dloss_per_token,
        }
