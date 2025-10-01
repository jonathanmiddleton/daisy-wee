import dataclasses
import os
import sys
import time
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dataclasses import dataclass, fields as dataclass_fields, asdict

from models import get_model_class
from training.data_gen import distributed_data_generator
from training.optim import Muon
from training.optim import get_lr, get_window_size_blocks
from tools.checkpoint import load_checkpoint, save_checkpoint, apply_model_state

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import torch

torch.empty(1, device="cuda", requires_grad=True).backward()  # prevents a bug on some systems
from torch import nn
import torch.distributed as dist

#torch._inductor.config.coordinate_descent_tuning = True
torch._dynamo.config.compiled_autograd = True
torch._dynamo.config.error_on_nested_fx_trace = False # temp workaround/diagnostic for dynamo error

@dataclass
class Hyperparameters:
    # Required scenario-specific fields
    train_files: str
    val_files: str
    max_seq_len: int
    val_seq_len: int
    num_iterations: int
    cooldown_frac: float
    # Common fields with defaults
    vocab_size: int = 50257
    val_tokens: int = 10485760  # how many tokens of validation data
    val_loss_every: int = 125  # num steps between validation loss calculations
    val_snapshot_every: int = 1000
    save_checkpoint: bool = True
    init_checkpoint: str | None = None
    num_layers: int = None
    num_heads: int = None
    model_dim: int = None
    head_dim: int = None
    snapshot_skip: int = None
    embed_params_lr: float = 0.3
    scalar_params_lr: float = 0.015
    hidden_matrix_params_lr: float = 0.025
    adamw_weight_decay: float = 0.01
    # Control schedule behavior on resume/warm-start
    ignore_prior_schedule: bool = False
    # Schedule control (decouple from stop condition)
    schedule_total_iters: int | None = None
    # Model selection
    model_type: str = "gpt2"

def load_hparams_from_yaml(config_path: str | None) -> Hyperparameters:
    """
    Load Hyperparameters from a YAML file. If no path is provided, defaults to config/instruct_sft.yml.
    Validates keys and required fields against the Hyperparameters dataclass.
    """
    cfg_dict = {}
    if config_path:
        used_path = Path(config_path)
        with open(used_path, "r") as f:
            cfg_dict = yaml.safe_load(f) or {}
    else:
        used_path = Path("config/pretrain.yml")
        if used_path.exists():
            with open(used_path, "r") as f:
                cfg_dict = yaml.safe_load(f) or {}

    valid_names = {f.name for f in dataclass_fields(Hyperparameters)}
    unknown = set(cfg_dict) - valid_names
    if unknown:
        raise ValueError(f"Unknown hyperparameter(s) in {used_path}: {sorted(unknown)}")

    required = [f.name for f in dataclass_fields(Hyperparameters)
                if f.default is dataclasses.MISSING and getattr(f, "default_factory",
                                                                dataclasses.MISSING) is dataclasses.MISSING]
    missing = [name for name in required if name not in cfg_dict]
    if missing:
        raise ValueError(f"Missing required hyperparameter(s) in {used_path}: {missing}")

    return Hyperparameters(**cfg_dict)

from typing import get_origin, get_args

def _coerce_value(val_str: str, typ):
    # Support Optional[...] and unions with None
    origin = get_origin(typ)
    args_ = get_args(typ)
    is_optional = False
    target_types = ()
    if origin is None:
        target_types = (typ,)
    elif origin is list or origin is tuple or origin is dict:
        # Simple YAML-like JSON parsing for collections
        # Fall back to YAML safe_load for flexible parsing
        return yaml.safe_load(val_str)
    elif origin is type(None):
        # Only NoneType
        is_optional = True
        target_types = (type(None),)
    else:
        # Assume Union
        target_types = args_
        if type(None) in target_types:
            is_optional = True
    # None coercion
    if val_str.lower() in ("none", "null"):
        if is_optional:
            return None
        # If not optional but asked for None, keep as string 'None'
        return val_str
    # Try booleans explicitly
    if bool in target_types or typ is bool:
        if val_str.lower() in ("1", "true", "t", "yes", "y", "on"):
            return True
        if val_str.lower() in ("0", "false", "f", "no", "n", "off"):
            return False
        # Fall through to attempt other conversions
    # Numeric conversions
    if int in target_types or typ is int:
        try:
            return int(val_str)
        except ValueError:
            pass
    if float in target_types or typ is float:
        try:
            return float(val_str)
        except ValueError:
            pass
    # Fallback: try YAML to parse literals, else string
    try:
        parsed = yaml.safe_load(val_str)
        return parsed
    except Exception:
        return val_str

# Load config from YAML (first CLI arg if provided)
_config_path = sys.argv[1] if len(sys.argv) > 1 else None
args = load_hparams_from_yaml(_config_path)

# Apply overrides from the remaining CLI args: support --key=value or key=value
field_map = {f.name: f for f in dataclass_fields(Hyperparameters)}
for s in sys.argv[2:]:
    if s.startswith("--"):
        s = s[2:]
    if "=" not in s:
        continue
    k, v = s.split("=", 1)
    k = k.strip().replace("-", "_")
    if k not in field_map:
        # Ignore unknown keys to allow torchrun args if any; alternatively, raise
        continue
    f = field_map[k]
    coerced = _coerce_value(v.strip(), f.type)
    setattr(args, k, coerced)

run_id = int(os.environ.get("RUN_ID", 0))
# torchrun sets these env variables
rank = int(os.environ.get("RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
TORCH_COMPILE_OFF = os.environ.get("TORCH_COMPILE_OFF", "0") == "1"
use_distributed = world_size > 1
# if use_distributed and world_size != 8:
#     print("[warn] This script is designed to run with world_size=8.")
assert torch.cuda.is_available()
device = torch.device("cuda", local_rank)
torch.cuda.set_device(device)
if use_distributed:
    dist.init_process_group(backend="nccl", device_id=device, world_size=world_size)
    dist.barrier()
master_process = (rank == 0)  # this process will do logging, checkpointing etc.

# Run start timestamp truncated to the minute (UTC)
_run_start_dt = datetime.now(timezone.utc).replace(second=0, microsecond=0)
run_start_minute = _run_start_dt.strftime("%Y%m%dT%H%M")
# Track last checkpoint file saved for this run so we can replace it
_last_run_ckpt_path: str | None = None

def print0(st):
    if master_process:
        print(st)


def _build_hparams_from_args(args: Hyperparameters, schedule_total_iters_den: int) -> dict:
    """Build a checkpoint hparams dict from training args."""
    return {
        "vocab_size": args.vocab_size,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "model_dim": args.model_dim,
        "head_dim": args.head_dim,
        "max_seq_len": args.max_seq_len,
        "val_seq_len": args.val_seq_len,
        "eos_token_id": 50256,
        "model_type": getattr(args, "model_type", "gpt2"),
        "schedule_total_iters": int(schedule_total_iters_den),
    }


def _save_run_checkpoint(
    *,
    val_value: float | int | None,
    step: int,
    args: Hyperparameters,
    model: nn.Module,
    optimizers: list[torch.optim.Optimizer],
    tokens_per_step: int,
    run_start_minute: str,
    run_id: int,
    last_ckpt_path: str | None,
    best_val: float | int | None,
    schedule_total_iters_den: int,
    print_message: bool = True,
) -> tuple[str | None, str]:
    """
    Save a checkpoint file for the current run, replacing the previous one from the same run if present.
    Returns (new_last_ckpt_path, filename).
    """
    os.makedirs("checkpoints", exist_ok=True)
    _val = float("nan") if val_value is None else val_value
    _val_trunc = math.trunc(_val * 100) / 100 if isinstance(_val, (int, float)) else float("nan")
    fname = f"checkpoints/{run_start_minute}-val{_val_trunc:.3f}-step{step:06d}-run{run_id}.pt"
    if last_ckpt_path and os.path.exists(last_ckpt_path) and last_ckpt_path != fname:
        try:
            os.remove(last_ckpt_path)
        except OSError:
            pass
    hparams = _build_hparams_from_args(args, schedule_total_iters_den)
    save_checkpoint(
        fname,
        model=model,
        optimizers=optimizers,
        step=step,
        best_val=best_val,
        hparams=hparams,
        tokens_per_step=tokens_per_step,
    )
    if print_message:
        # Only print if we have an actual numeric val_value
        try:
            _vv = float(val_value) if val_value is not None else float("nan")
            print0(f"Saved checkpoint to {fname} with val loss {_vv:.6f}")
        except Exception:
            print0(f"Saved checkpoint to {fname}")
    return fname, fname


########################################
#    Construct model and optimizer     #
########################################

# Rehydrate critical hyperparameters from checkpoint if available
best_val_from_ckpt = None
resume_from_step = None
_resume_tokens_per_step: int | None = None
_ckpt_obj = None
if args.init_checkpoint:
    _ckpt_obj = load_checkpoint(args.init_checkpoint, map_location=device)
    _saved_hparams = _ckpt_obj.hparams if _ckpt_obj is not None else None
    if isinstance(_saved_hparams, dict):
        # Only adopt architecture/sequence related fields required to restore the model
        for k in [
            "vocab_size", "num_layers", "num_heads", "model_dim", "head_dim", "max_seq_len", "val_seq_len", "model_type",
        ]:
            if k in _saved_hparams and _saved_hparams[k] is not None:
                setattr(args, k, _saved_hparams[k])
        print0("Rehydrated model hyperparameters from checkpoint.")

# Determine schedule denominator (decoupled from stop condition) after (optional) checkpoint rehydration
_schedule_total_iters_den = int(args.schedule_total_iters) if getattr(args, "schedule_total_iters", None) not in (None, 0) else int(args.num_iterations)
print0(json.dumps({**asdict(args), "_schedule_total_iters_den": _schedule_total_iters_den}, indent=2, sort_keys=True))

# Now we can safely build the model with possibly rehydrated args
_ModelClass = get_model_class(getattr(args, "model_type", "gpt2"))
model: nn.Module = _ModelClass(
    vocab_size=args.vocab_size,
    num_layers=args.num_layers,
    num_heads=args.num_heads,
    model_dim=args.model_dim,
    max_seq_len=max(args.max_seq_len, args.val_seq_len),
    head_dim=args.head_dim,
).cuda()

# If a checkpoint was provided, load weights and training metadata
if args.init_checkpoint:
    _sd = _ckpt_obj.model
    _missing, _unexpected = apply_model_state(model, _sd, strict=False)
    if _missing or _unexpected:
        raise ValueError(f"init_checkpoint:{args.init_checkpoint} missing:{_missing} unexpected:{_unexpected}")
    # Weights-only init: do not adopt training metadata (step, best_val, tokens_per_step)


for m in model.modules():
    if isinstance(m, nn.Embedding):
        m.bfloat16()
for param in model.parameters():
    if use_distributed:
        dist.broadcast(param.detach(), 0)
# collect the parameters to optimize
hidden_matrix_params = sorted((p for p in model.blocks.parameters() if p.ndim >= 2), key=lambda x: x.size(),
                              reverse=True)
embed_params = [*model.embed.parameters(), *model.value_embeds.parameters()]
scalar_params = [model.scalars]
# noinspection PyTypeChecker
head_params: list[nn.Parameter] = [model.lm_head_w]
# sanity check
params_collections = [hidden_matrix_params, embed_params, scalar_params, head_params]
optimized_parameters_set = {p for params in params_collections for p in params}
assert optimized_parameters_set == {*model.parameters()}
assert len(optimized_parameters_set) == sum(len(lst) for lst in params_collections)

# init the optimizer(s)
adam_param_groups = [dict(params=head_params, lr=1 / 320), dict(params=embed_params, lr=args.embed_params_lr),
                     dict(params=scalar_params, lr=args.scalar_params_lr)]
# small adam epsilon by @YouJiacheng. this is an alternate method of fixing the world_size dependence
# discovered by @fernbear.bsky.social https://x.com/hi_tysam/status/1879692937589875094
optimizer1 = torch.optim.AdamW(adam_param_groups, betas=(0.8, 0.95), eps=1e-10, weight_decay=args.adamw_weight_decay, fused=True)
optimizer2 = Muon(hidden_matrix_params, lr=args.hidden_matrix_params_lr, momentum=0.95, rank=rank, world_size=world_size)
optimizers: list[torch.optim.Optimizer] = [optimizer1, optimizer2]


def opt_params(opt: torch.optim.Optimizer) -> list[nn.Parameter]:
    return [p for g in opt.param_groups for p in g["params"]]


opt2params = {opt: opt_params(opt) for opt in optimizers}
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]


if not TORCH_COMPILE_OFF:
    print0("Compiling model...")
    model: nn.Module = torch.compile(model, dynamic=False)
    print0("Finished compiling model.")
else:
    print0(f"Compiling disabled: TORCH_COMPILE_OFF={TORCH_COMPILE_OFF}")

########################################
#        Training and validation       #
########################################

torch.cuda.reset_peak_memory_stats()
# Optional beginning shard (1-based) from environment
_begin_shard_env = os.environ.get("BEGIN_SHARD")
_begin_shard = int(_begin_shard_env) if _begin_shard_env not in (None, "",) else None
train_loader = distributed_data_generator(args.train_files, world_size * args.max_seq_len, rank, world_size, start_shard=_begin_shard)
tokens_per_step = world_size * args.max_seq_len
tokens_seen = 0
last_val_loss = None
last_val_tokens = 0
ema_dloss_per_token = None
training_time_ms = 0
best_val = float("inf") if best_val_from_ckpt is None else best_val_from_ckpt
val_iter = 0
# start the clock
if use_distributed:
    dist.barrier()
t0 = time.perf_counter()
# begin training
train_steps = args.num_iterations
start_step = resume_from_step if resume_from_step is not None else 0
if resume_from_step is not None:
    # Use tokens_per_step from the checkpoint if available to compute accurate tokens_seen
    _tps_for_resume = _resume_tokens_per_step if _resume_tokens_per_step is not None else tokens_per_step
    tokens_seen = resume_from_step * _tps_for_resume
    last_val_tokens = tokens_seen  # we've validated up to resume step
for step in range(start_step, train_steps + 1):
    last_step = (step == train_steps)

    # --------------- VALIDATION SECTION -----------------
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0 and step != 0):
        # stop the clock
        if use_distributed:
            dist.barrier()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        model.eval()
        val_batch_size = world_size * args.val_seq_len
        assert args.val_tokens % val_batch_size == 0
        val_steps = args.val_tokens // val_batch_size
        val_loader = distributed_data_generator(args.val_files, val_batch_size, rank, world_size)
        val_loss = torch.zeros((), device="cuda", dtype=torch.float32)
        with torch.no_grad():
            for _ in range(val_steps):
                inputs, targets = next(val_loader)
                val_loss = val_loss + model(inputs, get_window_size_blocks(step, _schedule_total_iters_den), targets)
        val_loss = val_loss / val_steps
        del val_loader
        # Average across ranks before using the value
        if use_distributed:
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        cur_val = float(val_loss.item())
        tokens_since_last = tokens_seen - last_val_tokens
        if last_val_loss is not None and tokens_since_last > 0:
            dpt = (cur_val - last_val_loss) / tokens_since_last
            ema_dloss_per_token = dpt if ema_dloss_per_token is None else 0.7 * ema_dloss_per_token + 0.3 * dpt
            print0(f"delta loss per 1e9 tokens:{dpt * 1e9:.6f} [ema:{(ema_dloss_per_token or 0.0) * 1e9:.6f}]")
        last_val_loss = cur_val
        last_val_tokens = tokens_seen
        print0( f"step:{step}/{train_steps} val_loss:{val_loss:.6f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms")

        if master_process:
            val_iter += 1
            improved = cur_val < best_val

            if improved:
                best_val = cur_val

            if (args.save_checkpoint and improved
                    and val_iter % args.val_snapshot_every == 0
                    and val_iter > args.snapshot_skip
                    and not last_step):
                _last_run_ckpt_path, _ = _save_run_checkpoint(
                    val_value=cur_val,
                    step=step,
                    args=args,
                    model=model,
                    optimizers=optimizers,
                    tokens_per_step=tokens_per_step,
                    run_start_minute=run_start_minute,
                    run_id=run_id,
                    last_ckpt_path=_last_run_ckpt_path,
                    best_val=best_val,
                    schedule_total_iters_den=_schedule_total_iters_den,
                    print_message=True,
                )


        model.train()
        # start the clock again
        if use_distributed:
            dist.barrier()
        t0 = time.perf_counter()

    if last_step:
        if master_process and args.save_checkpoint:
            _last_run_ckpt_path, _ = _save_run_checkpoint(
                val_value=last_val_loss,
                step=step,
                args=args,
                model=model,
                optimizers=optimizers,
                tokens_per_step=tokens_per_step,
                run_start_minute=run_start_minute,
                run_id=run_id,
                last_ckpt_path=_last_run_ckpt_path,
                best_val=best_val,
                schedule_total_iters_den=_schedule_total_iters_den,
                print_message=False,
            )

        break

    # --------------- TRAINING SECTION -----------------
    inputs, targets = next(train_loader)
    model(inputs, get_window_size_blocks(step, _schedule_total_iters_den), targets).backward()
    opt2futures = {
        opt: ([dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, async_op=True).get_future() for p in params if p.grad is not None]
              if use_distributed else [])
        for opt, params in opt2params.items()
    }
    # set optimization hyperparameters
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * get_lr(step, _schedule_total_iters_den, args.cooldown_frac)
    for group in optimizer2.param_groups:
        frac = min(step / 300, 1)  # momentum warmup for muon
        group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
    # step the optimizers
    for opt in optimizers:
        if use_distributed:
            torch.futures.collect_all(opt2futures[opt]).wait()
        opt.step()
    # null the gradients
    model.zero_grad(set_to_none=True)
    tokens_seen += tokens_per_step
    # logging
    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f"step:{step + 1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / (step + 1):.2f}ms")

print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
      f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB")
if use_distributed and dist.is_initialized():
    dist.destroy_process_group()
