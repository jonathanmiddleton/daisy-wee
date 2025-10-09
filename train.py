import dataclasses
import os
import sys
import time
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from dataclasses import asdict

from models import get_model_class
from training.data_gen import DistributedDataGenerator
from training.optim import Muon
from training.optim import get_lr_s, get_num_window_blocks, set_full_windows
from training.optim import build_optimizers_from_cfg
from training.eval import Evaluator
from tools.checkpoint import load_checkpoint, save_checkpoint, apply_model_state
from training.hparams import Hyperparameters, load_hparams_from_yaml, apply_cli_overrides
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import torch

from torch import nn
import torch.distributed as dist

torch._inductor.config.coordinate_descent_tuning = True
torch._dynamo.config.compiled_autograd = True
torch._dynamo.config.error_on_nested_fx_trace = False  # temp workaround/diagnostic for dynamo error

# Load config from YAML (first CLI arg if provided)
_config_path = sys.argv[1] if len(sys.argv) > 1 else None
args = load_hparams_from_yaml(_config_path)

# Apply overrides from the remaining CLI args: support --key=value or key=value
args = apply_cli_overrides(args, sys.argv[2:])

# Apply scheduling toggle for attention windows
set_full_windows(args.full_windows)

run_id = int(os.environ.get("RUN_ID", 0))
# torchrun sets these env variables
rank = int(os.environ.get("RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
TORCH_COMPILE_OFF = os.environ.get("TORCH_COMPILE_OFF", "0") == "1"
use_distributed = world_size > 1
assert torch.cuda.is_available()
device = torch.device("cuda", local_rank)
torch.cuda.set_device(device)
torch.empty(1, device="cuda", requires_grad=True).backward()  # prevents a bug on some systems
if use_distributed:
    dist.init_process_group(backend="nccl", device_id=device, world_size=world_size)
    dist.barrier()
master_process = (rank == 0)  # this process will do logging, checkpointing etc.

# Run start timestamp truncated to the minute (UTC)
_run_start_dt = datetime.now(timezone.utc).replace(second=0, microsecond=0)
run_start_minute = _run_start_dt.strftime("%Y%m%dT%H%M")
# Track last checkpoint file saved for this run so we can replace it
_last_run_ckpt_path: str | None = None

# Optional Weights & Biases logging (minimal config)
_wandb = None
_wandb_enabled = False
if master_process and args.wandb_log:
    try:
        import wandb as _wandb

        _project = args.wandb_project or "daisy-wee"
        _name = args.wandb_run_name or f"{run_start_minute}-run{run_id}"
        _wandb.init(project=_project, name=_name, config=asdict(args))
        _wandb_enabled = True
        print(f"wandb logging enabled: project={_project} name={_name}")
        art = _wandb.Artifact(name=f"models-src-{run_id}", type="code")
        import glob
        files = [p for p in glob.glob("models/**/*.py", recursive=True) if os.path.basename(p) != "__init__.py"]
        for p in files:
            art.add_file(p, name=os.path.relpath(p, start="models"))
        _wandb.log_artifact(art)
    except Exception as _e:
        print(f"[warn] Failed to initialize wandb logging: {_e}")
        _wandb = None
        _wandb_enabled = False

# noinspection PyShadowingNames
def log_wandb(d: dict):
    if _wandb_enabled:
        try:
            _wandb.log(d)
        except Exception as _e:
            print0(f"[warn] wandb.log failed: {_e}")

def print0(st):
    if master_process:
        print(st)

# noinspection PyShadowingNames
def _build_hparams_from_args(args: Hyperparameters) -> dict:
    """Build a checkpoint hparams dict from training args."""
    return {
        "vocab_size": args.vocab_size,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "model_dim": args.model_dim,
        "head_dim": args.head_dim,
        "training_sequence_length": args.training_sequence_length,
        "val_seq_len": args.val_seq_len,
        "train_attention_window_len": args.train_attention_window_len,
        "window_block_size": args.window_block_size,
        "eos_token_id": args.eos_token_id,
        "model_class": args.model_class,
    }

# noinspection PyShadowingNames
def _run_ckpt_filename(
        *, val_value: float | None, step: int, run_start_minute: str, run_id: int
) -> str:
    os.makedirs("checkpoints", exist_ok=True)
    _val_trunc = math.trunc(val_value * 100) / 100 if val_value is not None else float("nan")
    return f"checkpoints/{run_start_minute}-val{_val_trunc:.3f}-step{step:06d}-run{run_id}.pt"


# noinspection PyShadowingNames
def _save_run_checkpoint(
        *,
        val_value: float | None,
        step: int,
        run_start_minute: str,
        run_id: int,
        model: nn.Module,
        best_val: float,
        args: Hyperparameters,
        tokens_per_step: int,
        progress,
        overwrite: bool = False,
) -> str:
    """Create a run-scoped checkpoint filename, remove the previous run checkpoint if different,
    save the new checkpoint, and remember its path.

    Returns the path to the saved checkpoint.
    """
    global _last_run_ckpt_path
    fname = _run_ckpt_filename(
        val_value=val_value,
        step=step,
        run_start_minute=run_start_minute,
        run_id=run_id,
    )
    if overwrite and _last_run_ckpt_path and os.path.exists(_last_run_ckpt_path) and _last_run_ckpt_path != fname:
        try:
            os.remove(_last_run_ckpt_path)
        except OSError:
            pass

    save_checkpoint(
        fname,
        model=model,
        step=step,
        best_val=best_val,
        hparams=_build_hparams_from_args(args),
        tokens_per_step=tokens_per_step,
        progress_state=progress.state_dict(),
    )
    _last_run_ckpt_path = fname
    return fname


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
            "vocab_size", "num_layers", "num_heads", "model_dim", "head_dim",
            "training_sequence_length", "val_seq_len", "train_attention_window_len", "window_block_size",
            "eos_token_id", "model_class",
        ]:
            if k in _saved_hparams and _saved_hparams[k] is not None:
                setattr(args, k, _saved_hparams[k])
        print0("Rehydrated model hyperparameters from checkpoint.")

# Determine schedule denominator (decoupled from stop condition) after (optional) checkpoint rehydration
print0(json.dumps(asdict(args), indent=2, sort_keys=True))
# Ensure wandb sees the final effective hyperparameters (after overrides and any rehydration)
if _wandb_enabled:
    try:
        _wandb.config.update(asdict(args), allow_val_change=True)
    except Exception as _e:
        print0(f"[warn] Failed to update wandb config: {_e}")

# Now we can safely build the model with possibly rehydrated args
_ModelClass = get_model_class(args.model_class)
model: nn.Module = _ModelClass(
    vocab_size=args.vocab_size,
    num_layers=args.num_layers,
    num_heads=args.num_heads,
    model_dim=args.model_dim,
    max_seq_len=max(args.training_sequence_length, args.val_seq_len),
    head_dim=args.head_dim,
    window_block_size=args.window_block_size,
    eos_token_id=args.eos_token_id,
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
# Build optimizer(s) from YAML config
if not args.optimizers:
    raise ValueError("Training config must provide 'optimizers' list")
optimizers: list[torch.optim.Optimizer] = build_optimizers_from_cfg(
    cfg_list=args.optimizers,
    model=model,
    rank=rank,
    world_size=world_size,
)


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
_train_ddg = DistributedDataGenerator(args.train_shards, world_size * args.training_sequence_length, rank, world_size,
                                      start_shard=_begin_shard)
val_batch_size = world_size * args.val_seq_len
if args.tot_val_tokens % val_batch_size != 0:
    raise ValueError(f"tot_val_tokens ({args.tot_val_tokens}) must be divisible by val_batch_size ({val_batch_size})")
# Build a persistent validation data generator and evaluator
_val_ddg = DistributedDataGenerator(args.val_shards, val_batch_size, rank, world_size)
_evaluator: Evaluator = Evaluator(
    data_generator=_val_ddg,
    distributed_enabled=use_distributed,
    rank=rank,
    train_attention_window_len=args.train_attention_window_len,
    window_block_size=args.window_block_size,
)

# Tokens per training micro-step (includes padding by design)
tokens_per_step = world_size * args.training_sequence_length
# Effective tokens per optimizer step (accounts for gradient accumulation)
_ga_steps_cfg = max(1, int(args.grad_acc_steps))
_tokens_per_optim_step = tokens_per_step * _ga_steps_cfg

# Progress and tracking
from training.progress import ProgressMeter

progress = ProgressMeter(
    target_tokens=int(args.target_tokens),
    eval_every_tokens=int(args.val_loss_every_tokens) if int(args.val_loss_every_tokens) > 0 else None,
    snapshot_per_n_tokens=int(args.snapshot_per_n_tokens) if int(args.snapshot_per_n_tokens) > 0 else None,
    snapshot_warmup_tokens=int(args.snapshot_warmup_tokens),
)

# Tracking for eval stats and ETA
last_val_loss = None
last_tot_val_tokens = 0
ema_dloss_per_token = math.inf
training_time_ms = 0
best_val = float("inf") if best_val_from_ckpt is None else best_val_from_ckpt

if use_distributed:
    dist.barrier()
step = 0
t0 = time.perf_counter()
warmup_end = 0.0

while progress.tokens_processed < progress.target_tokens:
    # --------------- Evaluation -----------------
    if progress.should_eval():
        if use_distributed:
            dist.barrier()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        model.eval()
        # Evaluate using the Evaluator (per-rank tokens)
        eval_out = _evaluator.eval(model=model, total_tokens=args.tot_val_tokens)
        cur_val = float(eval_out.get("val_loss", float("nan")))
        last_val_loss = cur_val
        ema_dloss_per_token = eval_out.get("ema_dloss_per_token", ema_dloss_per_token)
        print0(
            f"step:{step} tokens:{progress.tokens_processed:,}/{progress.target_tokens:,} (s={progress.s:.4f}) "
            f"val_loss:{cur_val:.6f} train_time:{training_time_ms:,.0f}ms ema_dloss_per_1e6_tokens:{ema_dloss_per_token*1e6:.6f}")
        log_wandb({
                    "val/loss": cur_val,
                    "val/ppl": math.exp(cur_val) if cur_val < 20 else float("inf"),
                    "val/ema_dloss_per_token": ema_dloss_per_token,
                    "tokens": progress.tokens_processed,
                    "s": progress.s,
                    "train/time_ms": training_time_ms + 1000 * (time.perf_counter() - t0),
                    "step": step,
                })
        # checkpoints by tokens (save only if validation improves)
        if master_process and args.save_checkpoint and progress.should_snapshot():
            if cur_val < best_val:
                best_val = cur_val
                # Save checkpoint (handles filename, cleanup, and bookkeeping)
                fname = _save_run_checkpoint(
                    val_value=cur_val,
                    step=step,
                    run_start_minute=run_start_minute,
                    run_id=run_id,
                    model=model,
                    best_val=best_val,
                    args=args,
                    tokens_per_step=_tokens_per_optim_step,
                    progress=progress,
                    overwrite=True,
                )
                print0(f"Saved checkpoint to {fname} with val loss {float(cur_val):.6f}")
            else:
                print0(f"No improvement in val loss: best={best_val:.6f}, current={cur_val:.6f}. Skipping checkpoint.")

            progress.mark_snapshot_done()
        # resume training clock
        model.train()
        if use_distributed:
            dist.barrier()
        t0 = time.perf_counter()
        progress.mark_eval_done()

    # --------------- TRAINING SECTION -----------------
    ga_steps = max(1, int(args.grad_acc_steps))
    total_train_loss = 0.0

    for micro_step in range(ga_steps):
        inputs, targets = next(_train_ddg)
        loss = model(inputs, get_num_window_blocks(progress.s, attention_window_len=args.train_attention_window_len, window_block_size=args.window_block_size), targets)
        # scale loss so that gradients are averaged across micro-steps
        loss_to_backward = loss / ga_steps
        if use_distributed:
            model.require_backward_grad_sync = (micro_step == ga_steps - 1) # no_sync()
        loss_to_backward.backward()
        total_train_loss += float(loss.item())

    # collect the futures for all the optimizers (do distributed grad average once after accumulation)
    opt2futures = {
        opt: ([dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, async_op=True).get_future() for p in params if
               p.grad is not None]
              if use_distributed else [])
        for opt, params in opt2params.items()
    }
    # set optimization hyperparameters based on s
    s = progress.s
    lr_scale = get_lr_s(s, args.cooldown_frac)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lr_scale
    # Momentum warmup for Muon optimizers driven by progress s
    for opt in optimizers:
        if isinstance(opt, Muon):
            for group in opt.param_groups:
                frac = s
                group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
    # step the optimizers
    for opt in optimizers:
        if use_distributed:
            torch.futures.collect_all(opt2futures[opt]).wait()
        opt.step()
    # null the gradients
    model.zero_grad(set_to_none=True)

    # Update tokens and counters (only once per accumulation cycle)
    progress.update(tokens_per_step * ga_steps)
    step += 1

    # logging (only at accumulation boundary)
    train_loss_est = total_train_loss / ga_steps
    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    if step == 9:
        warmup_end = approx_training_time_ms
    avg_step = f"avg_step:{(approx_training_time_ms - warmup_end) / max(step - 9, 1):.2f}ms" if step >= 10 else "avg_step: (warmup to step 10)"
    print0(
        f"step:{step} train_loss:{train_loss_est:.4f} tokens:{progress.tokens_processed:,}/{progress.target_tokens:,} (s={progress.s:.4f}) train_time:{approx_training_time_ms:,.0f}ms {avg_step}")
    log_wandb({
                "train/loss": train_loss_est,
                "train/ppl": math.exp(train_loss_est) if train_loss_est < 20 else float("inf"),
                "tokens": progress.tokens_processed,
                "s": progress.s,
                "lr_scale": lr_scale,
                "train/time_ms": approx_training_time_ms,})

# End of training: save final checkpoint
if master_process and args.save_checkpoint:
    _ = _save_run_checkpoint(
        val_value=last_val_loss,
        step=step,
        run_start_minute=run_start_minute,
        run_id=run_id,
        model=model,
        best_val=best_val,
        args=args,
        tokens_per_step=_tokens_per_optim_step,
        progress=progress,
        overwrite=False
    )

print0(
    f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB")
if _wandb_enabled:
    # noinspection PyBroadException
    try:
        _wandb.finish()
    except Exception:
        pass
if use_distributed and dist.is_initialized():
    dist.destroy_process_group()
