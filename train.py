import os
import sys
import time
import json
import math
from datetime import datetime, timezone

from dataclasses import asdict
from typing import Optional

from tools.checkpoint import model_from_checkpoint
from tools.model_report import build_report, format_report_text
from models import get_model_class, model_from_spec
from data.data_gen_stream import DistributedDataGenerator
from training.optim import Muon, get_lr_scale
from training.optim import get_num_window_blocks, set_full_windows
from training.optim import build_optimizers_from_cfg
from training.eval import Evaluator
from tools.checkpoint import load_checkpoint, save_checkpoint, apply_model_state
from training.hparams import Hyperparameters, load_hparams_from_yaml, apply_cli_overrides

########################################
#        App Config & Setup            #
########################################
# Load config from YAML (first CLI arg if provided)
_config_path = sys.argv[1] if len(sys.argv) > 1 else None
args = load_hparams_from_yaml(_config_path)
# Apply overrides from the remaining CLI args: support --key=value or key=value
args = apply_cli_overrides(args, sys.argv[2:])
# Apply optional scheduling toggle for attention windows
set_full_windows(args.full_windows)
run_id = int(os.environ.get("RUN_ID", 0))
### End App Config ###


########################################
#        Torch Setup                   #
########################################
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import torch
from torch import nn
import torch.distributed as dist
# Configure inductor/dynamo compile/tuning
torch._inductor.config.coordinate_descent_tuning = bool(getattr(args, "torch_coordinate_descent_tuning", False))
torch._dynamo.config.compiled_autograd = True
torch._dynamo.config.error_on_nested_fx_trace = False  # temp workaround/diagnostic for dynamo error related to FlexAttention
# torchrun sets these env variables
rank = int(os.environ.get("RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
TORCH_COMPILE_OFF = os.environ.get("TORCH_COMPILE_OFF", "0") == "1"
use_distributed = world_size > 1

device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device('mps') if torch.mps.is_available() else torch.device('cpu')
if device.type == 'cuda':
    torch.cuda.set_device(device)
    if use_distributed:
        dist.init_process_group(backend="nccl", device_id=device, world_size=world_size)
        dist.barrier()
    is_master = (rank == 0)
elif device.type == 'mps':
    if use_distributed:
        raise ValueError("Distributed training is not supported on macOS/MPS")
    is_master = True
else:
    is_master = True

def maybe_compile(model: nn.Module, dynamic: bool = False) -> nn.Module:
    if TORCH_COMPILE_OFF:
        logger.info(f"Compiling disabled: TORCH_COMPILE_OFF={TORCH_COMPILE_OFF}")
        return model
    else:
        logger.info(f"Compiling model (dynamic={dynamic}). This may take several minutes.")
        logger.info("Note: on CUDA you may see a torch internal warning (per-device) about the use of a deprecated API. This is normal and can be ignored.")
        model: nn.Module = torch.compile(model, dynamic=dynamic)
        return model

def maybe_reset_peak_memory_stats() -> None:
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    else:
        logger.debug(f"reset_memory_stats() unsupported on device.type={device.type}")

def get_max_memory_allocated() -> Optional[int]:
    if device.type == 'cuda':
        return torch.cuda.max_memory_allocated()
    else:
        logger.debug(f"max_memory_allocated() unsupported on device.type={device.type}")
        return None
### End Torch Setup ###


########################################
#        Logging                       #
########################################
from tools.master_logger import MasterLogger
logger = MasterLogger
### End Logging ###


# Run start timestamp truncated to the minute (UTC)
_run_start_dt = datetime.now(timezone.utc).replace(second=0, microsecond=0)
run_start_minute = _run_start_dt.strftime("%Y%m%dT%H%M")
# Track last checkpoint file saved for this run so we can replace it
_last_run_ckpt_path: str | None = None

########################################
#       Weights & Biases (Optional)    #
########################################
_wandb = None
_wandb_enabled = False
if is_master and args.wandb_log:
    try:
        import wandb as _wandb

        _project = args.wandb_project or "daisy-wee"
        _name = args.wandb_run_name or f"{run_start_minute}-run{run_id}"
        _wandb.init(project=_project, name=_name, config=asdict(args))
        _wandb_enabled = True
        logger.info(f"wandb logging enabled: project={_project} name={_name}")
        art = _wandb.Artifact(name=f"models-src-{run_id}", type="code")
        import glob
        files = [p for p in glob.glob("models/**/*.py", recursive=True) if os.path.basename(p) != "__init__.py"]
        for p in files:
            art.add_file(p, name=os.path.relpath(p, start="models"))
        _wandb.log_artifact(art)
    except Exception as _e:
        logger.error(f"[warn] Failed to initialize wandb logging: {_e}")
        _wandb = None
        _wandb_enabled = False

# noinspection PyShadowingNames
def log_wandb(d: dict):
    if _wandb_enabled:
        try:
            _wandb.log(d)
        except Exception as _e:
            logger.error(f"[warn] wandb.log failed: {_e}")

def update_wandb_config(d: dict):
    if _wandb_enabled:
        try:
            _wandb.config.update(d, allow_val_change=True)
        except Exception as _e:
            logger.error(f"[warn] Failed to update wandb config: {_e}")


########################################
#               Helpers                #
########################################

# noinspection PyShadowingNames
def _build_hparams_from_args(args: Hyperparameters) -> dict:
    """Build a checkpoint hparams dict from training args."""
    return asdict(args)

# noinspection PyShadowingNames
def _get_ckpt_filename(*, val_value: float | None, step: int, run_start_minute: str, run_id: int, suffix: str | None = None, ) -> str:
    os.makedirs("checkpoints", exist_ok=True)
    _val_trunc = math.trunc(val_value * 100) / 100 if val_value is not None else float("nan")
    return f"checkpoints/{run_start_minute}-val{_val_trunc:.3f}-step{step:06d}-run{run_id}" + (f"-{suffix}" if suffix else "") + ".pt"

# noinspection PyShadowingNames
def _save_checkpoint(
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
        suffix: str | None = None,
) -> str:
    """Create a run-scoped checkpoint filename, remove the previous run checkpoint if different,
    save the new checkpoint, and remember its path.

    Returns the path to the saved checkpoint.
    """
    global _last_run_ckpt_path
    fname = _get_ckpt_filename(
        val_value=val_value,
        step=step,
        run_start_minute=run_start_minute,
        run_id=run_id,
        suffix=suffix,
    )
    if overwrite and _last_run_ckpt_path and os.path.exists(_last_run_ckpt_path) and _last_run_ckpt_path != fname:
        try:
            os.remove(_last_run_ckpt_path)
        except OSError:
            logger.warning(f"Failed to remove previous run checkpoint: {_last_run_ckpt_path}")

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
#        Model & Optimizers            #
########################################

# Rehydrate critical hyperparameters from checkpoint if available
best_val_from_ckpt = None
resume_from_step = None
_resume_tokens_per_step: int | None = None
_ckpt_obj = None
if args.init_checkpoint:
    model, hparams = model_from_checkpoint(args.init_checkpoint, device=device)
    # TODO diff args/hparams
    logger.info("Rehydrated model from checkpoint.")
else:
    model = model_from_spec(args.model_spec, device=device.type)
    hparams = _build_hparams_from_args(args)

logger.info("Hyperparameters:\n" + json.dumps(asdict(args), indent=2, sort_keys=True))
# Ensure wandb sees the final effective hyperparameters (after overrides and any rehydration)
update_wandb_config(asdict(args)) # TODO without diff of checkpoint/model_spec_from_args the reported model args may be incorrect

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

report = build_report(model)
logger.info(f"Model report:\n{format_report_text(report)}")
model: nn.Module = maybe_compile(model, dynamic=False)


########################################
#    Training and Validation Setup     #
########################################

maybe_reset_peak_memory_stats()
# Optional beginning shard (1-based) from environment
_begin_shard_env = os.environ.get("BEGIN_SHARD")
_begin_shard = int(_begin_shard_env) if _begin_shard_env not in (None, "",) else None
_train_ddg = DistributedDataGenerator(args.train_shards, world_size * args.training_sequence_length, rank, world_size,
                                      start_shard=_begin_shard, device=device.type)
val_batch_size = world_size * args.val_seq_len
if args.tot_val_tokens % val_batch_size != 0:
    raise ValueError(f"tot_val_tokens ({args.tot_val_tokens}) must be divisible by val_batch_size ({val_batch_size})")
# Build persistent validation data generators and evaluators for each configured dataset
_val_evals: list[tuple[str, Evaluator]] = []
for _v in args.val_shards:
    _label = _v.get("type")
    _path = _v.get("path")
    _ddg = DistributedDataGenerator(_path, val_batch_size, rank, world_size, device=device.type)
    _eval = Evaluator(
        data_generator=_ddg,
        distributed_enabled=use_distributed,
        rank=rank,
        train_attention_window_len=args.train_attention_window_len,
        window_block_size=args.window_block_size,
    )
    _val_evals.append((_label, _eval))

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
    checkpoint_per_n_tokens=int(args.checkpoint_per_n_tokens),  # allow 0 to mean every update after warmup
    checkpoint_warmup_tokens=int(args.checkpoint_warmup_tokens),
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
        logger.info("[eval] starting evaluations...")
        if use_distributed:
            dist.barrier()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        model.eval()
        # Evaluate using all configured Evaluators (per-rank tokens)
        per_ds_results: list[tuple[str, dict]] = []
        for _label, _ev in _val_evals:
            _world_batch = val_batch_size
            _steps = args.tot_val_tokens // _world_batch if _world_batch > 0 else 0
            logger.info(f"[eval] dataset={_label} steps={_steps} (global_batch={_world_batch}, tot_tokens={args.tot_val_tokens})")
            _ev.reset_generator()
            _out = _ev.eval(model=model, total_tokens=args.tot_val_tokens)
            per_ds_results.append((_label, _out))
        # Canonical/primary val metrics use the first dataset
        primary_label, primary_out = per_ds_results[0]
        cur_val = float(primary_out.get("val_loss", float("nan")))
        last_val_loss = cur_val
        ema_dloss_per_token = primary_out.get("ema_dloss_per_token", ema_dloss_per_token)
        # Print a compact per-dataset summary line
        parts = [f"{lbl}:{float(out.get('val_loss', float('nan'))):.6f}" for lbl, out in per_ds_results]
        logger.info(
            f"step:{step} tokens:{progress.tokens_processed:,}/{progress.target_tokens:,} (s={progress.s:.4f}) "
            + " ".join(parts) +
            f" train_time:{training_time_ms:,.0f}ms ema_dloss_per_1e6_tokens:{ema_dloss_per_token*1e6:.6f}")
        # W&B logging: primary under legacy keys; all datasets under namespaced keys
        wb = {
            "val/loss": cur_val,
            "val/ppl": math.exp(cur_val) if cur_val < 20 else float("inf"),
            "val/ema_dloss_per_token": ema_dloss_per_token,
            "tokens": progress.tokens_processed,
            "s": progress.s,
            "train/time_ms": training_time_ms + 1000 * (time.perf_counter() - t0),
            "step": step,
        }
        for lbl, out in per_ds_results:
            _loss = float(out.get("val_loss", float("nan")))
            wb[f"val/{lbl}/loss"] = _loss
            wb[f"val/{lbl}/ppl"] = math.exp(_loss) if _loss < 20 else float("inf")
        log_wandb(wb)
        # checkpoints by tokens (save only if validation improves) using primary dataset
        if is_master and args.save_checkpoint and progress.should_checkpoint():
            if cur_val < best_val:
                best_val = cur_val
                # Save checkpoint (handles filename, cleanup, and bookkeeping)
                fname = _save_checkpoint(
                    val_value=cur_val,
                    step=step,
                    run_start_minute=run_start_minute,
                    run_id=run_id,
                    model=model,
                    best_val=best_val,
                    args=args,
                    tokens_per_step=_tokens_per_optim_step,
                    progress=progress,
                    overwrite=False,
                    suffix="best",
                )
                logger.info(f"Saved checkpoint to {fname} with val loss {float(cur_val):.6f}")
            else:
                logger.info(f"No improvement in val loss: best={best_val:.6f}, current={cur_val:.6f}. Skipping checkpoint.")

            progress.mark_checkpoint_done()
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
        n_blocks = get_num_window_blocks(progress.s, attention_window_len=args.train_attention_window_len, window_block_size=args.window_block_size).to(device.type)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            logger.debug(f"inputs.shape={inputs.shape} inputs.device.type={inputs.device.type} targets.shape={targets.shape} targets.device.type={targets.device.type} n_blocks={n_blocks}")
            loss = model(inputs, n_blocks, targets)
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
    # Compute LR scale by selected schedule via dispatch in training.optim
    lr_scale = get_lr_scale(args.learning_rate_schedule, s, args.cooldown_frac)

    for opt in optimizers:
        if isinstance(opt, Muon):
            continue
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
    logger.info(
        f"step:{step} train_loss:{train_loss_est:.4f} tokens:{progress.tokens_processed:,}/{progress.target_tokens:,} (s={progress.s:.4f}) train_time:{approx_training_time_ms:,.0f}ms {avg_step} lr_scale:{lr_scale:.4f}")
    log_wandb({
                "train/loss": train_loss_est,
                "train/ppl": math.exp(train_loss_est) if train_loss_est < 20 else float("inf"),
                "tokens": progress.tokens_processed,
                "s": progress.s,
                "lr_scale": lr_scale,
                "train/time_ms": approx_training_time_ms,})

# End of training: save final checkpoint
if is_master and args.save_checkpoint:
    _ = _save_checkpoint(
        val_value=last_val_loss,
        step=step,
        run_start_minute=run_start_minute,
        run_id=run_id,
        model=model,
        best_val=best_val,
        args=args,
        tokens_per_step=_tokens_per_optim_step,
        progress=progress,
        overwrite=False,
        suffix="final"
    )

_peak_mem = get_max_memory_allocated()
if _peak_mem is not None:
    logger.info(f"peak memory allocated: {_peak_mem // 1024 // 1024} MiB")
if _wandb_enabled:
    # noinspection PyBroadException
    try:
        _wandb.finish()
    except Exception as _e:
        logger.warning("wandb.finish(): " + str(_e))
if use_distributed and dist.is_initialized():
    dist.destroy_process_group()
