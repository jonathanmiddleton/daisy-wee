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
from model_specs import load_model_spec

from models import get_model_class
from training.data_gen import distributed_data_generator, DistributedDataGenerator
from training.optim import Muon
from training.optim import get_lr_s, get_window_size_blocks_s, set_full_windows
from training.eval import Evaluator
from tools.checkpoint import load_checkpoint, save_checkpoint, apply_model_state
from tools.helpers import _coerce_value

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
    train_shards: str
    val_shards: str
    max_seq_len: int
    val_seq_len: int
    target_tokens: int
    cooldown_frac: float
    # Common fields with defaults
    vocab_size: int = 50257
    val_tokens: int = 10485760  # how many tokens of validation data
    val_loss_every_tokens: int = 0  # num tokens between validation passes (0 disables)
    snapshot_warmup_tokens: int = 0  # tokens to skip before taking snapshots
    snapshot_per_n_tokens: int | None = None  # interval in tokens between snapshots
    save_checkpoint: bool = True
    init_checkpoint: str | None = None
    num_layers: int = None
    num_heads: int = None
    model_dim: int = None
    head_dim: int = None
    head_params_lr: float = 0.008
    embed_params_lr: float = 0.3
    scalar_params_lr: float = 0.015
    hidden_matrix_params_lr: float = 0.025
    adamw_weight_decay: float = 0.01
    # Force full attention windows (useful when resuming after smaller windows)
    full_windows: bool = False
    # Optional: legacy schedule control (not used for stopping)
    schedule_total_iters: int | None = None
    # Gradient accumulation
    grad_acc_steps: int = 1
    # Model selection
    model_spec: str | None = None  # name of model spec under model_specs/, or a path to a spec file
    model_type: str = "gpt2"
    # Weights & Biases minimal logging config
    wandb_log: bool = False
    wandb_project: str = ""
    wandb_run_name: str = ""

def load_hparams_from_yaml(config_path: str | None) -> Hyperparameters:
    """
    Load Hyperparameters from a YAML file. If a 'model_spec' key is present, also load and merge
    the named spec from model_specs/<name>.yml (or a provided file path). Training config values
    take precedence over spec values. Validates against the Hyperparameters dataclass.
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

    # If a model_spec name/path is provided, load the spec and merge recognized fields
    model_spec_name = cfg_dict.get("model_spec")
    if model_spec_name:
        spec_dict = load_model_spec(str(model_spec_name))
        # Only merge keys that are valid Hyperparameters fields and not explicitly set in training cfg
        valid_names_for_merge = {f.name for f in dataclass_fields(Hyperparameters)}
        for k, v in (spec_dict.items() if isinstance(spec_dict, dict) else []):
            if k in valid_names_for_merge and (k not in cfg_dict or cfg_dict[k] is None):
                cfg_dict[k] = v

    # Validate keys after potential spec merge
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

# Apply scheduling toggle for attention windows
set_full_windows(getattr(args, "full_windows", False))

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

# Optional Weights & Biases logging (minimal config)
_wandb = None
_wandb_enabled = False
if master_process and getattr(args, "wandb_log", False):
    try:
        import wandb as _wandb
        _project = args.wandb_project or "daisy-wee"
        _name = args.wandb_run_name or f"{run_start_minute}-run{run_id}"
        _wandb.init(project=_project, name=_name, config=asdict(args))
        _wandb_enabled = True
        print(f"wandb logging enabled: project={_project} name={_name}")
    except Exception as _e:
        print(f"[warn] Failed to initialize wandb logging: {_e}")
        _wandb = None
        _wandb_enabled = False

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
    val_value: float | None,
    step: int,
    args: Hyperparameters,
    model: nn.Module,
    optimizers: list[torch.optim.Optimizer],
    tokens_per_step: int,
    run_start_minute: str,
    run_id: int,
    last_ckpt_path: str | None,
    best_val: float | None,
    schedule_total_iters_den: int,
    print_message: bool = True,
    progress_state: dict | None = None,
) -> tuple[str | None, str]:
    """
    Save a checkpoint file for the current run, replacing the previous one from the same run if present.
    Returns (new_last_ckpt_path, filename).
    """
    os.makedirs("checkpoints", exist_ok=True)
    _val_trunc = math.trunc(val_value * 100) / 100 if val_value is not None else float("nan")
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
        step=step,
        best_val=best_val,
        hparams=hparams,
        tokens_per_step=tokens_per_step,
        progress_state=progress_state,
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
_schedule_total_iters_den = int(args.schedule_total_iters) if getattr(args, "schedule_total_iters", None) not in (None, 0) else 0
print0(json.dumps({**asdict(args), "_schedule_total_iters_den": _schedule_total_iters_den}, indent=2, sort_keys=True))
# Ensure wandb sees the final effective hyperparameters (after overrides and any rehydration)
if _wandb_enabled:
    try:
        _wandb.config.update(asdict(args), allow_val_change=True)
        # Also log derived scheduling denominator for transparency
        _wandb.config.update({"_schedule_total_iters_den": _schedule_total_iters_den}, allow_val_change=True)
    except Exception as _e:
        print0(f"[warn] Failed to update wandb config: {_e}")

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
adam_param_groups = [dict(params=head_params, lr=args.head_params_lr), dict(params=embed_params, lr=args.embed_params_lr),
                     dict(params=scalar_params, lr=args.scalar_params_lr)]
# small adam epsilon by @YouJiacheng. this is an alternate method of fixing the world_size dependence
# discovered by @fernbear.bsky.social https://x.com/hi_tysam/status/1879692937589875094
optimizer1 = torch.optim.AdamW(adam_param_groups, betas=(0.9, 0.95), eps=1e-10, weight_decay=args.adamw_weight_decay, fused=True)
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
train_loader = distributed_data_generator(args.train_shards, world_size * args.max_seq_len, rank, world_size, start_shard=_begin_shard)
val_batch_size = world_size * args.val_seq_len
if args.val_tokens % val_batch_size != 0:
    raise ValueError(f"val_tokens ({args.val_tokens}) must be divisible by val_batch_size ({val_batch_size})")
val_steps = args.val_tokens // val_batch_size
# Build a persistent validation data generator and evaluator
_val_ddg = DistributedDataGenerator(args.val_shards, val_batch_size, rank, world_size)
_evaluator = Evaluator(
    wandb_enabled=_wandb_enabled,
    data_generator=_val_ddg,
    distributed_enabled=use_distributed,
    rank=rank,
)

# Tokens per training micro-step (includes padding by design)
tokens_per_step = world_size * args.max_seq_len

# Progress and tracking
from training.progress import ProgressMeter

progress = ProgressMeter(
    target_tokens=int(args.target_tokens),
    eval_every_tokens=int(args.val_loss_every_tokens) if getattr(args, 'val_loss_every_tokens', 0) else None,
    snapshot_per_n_tokens=int(args.snapshot_per_n_tokens) if getattr(args, 'snapshot_per_n_tokens', None) else None,
    snapshot_warmup_tokens=int(getattr(args, 'snapshot_warmup_tokens', 0) or 0),
)

# Tracking for eval stats and ETA
last_val_loss = None
last_val_tokens = 0
ema_dloss_per_token = None
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
        _per_rank_tokens = args.val_tokens // world_size
        eval_out = _evaluator.eval(model, _per_rank_tokens, tokens=progress.tokens_processed)
        cur_val = float(eval_out.get("val_loss", float("nan")))
        last_val_loss = cur_val
        ema_dloss_per_token = eval_out.get("ema_dloss_per_token", ema_dloss_per_token)
        print0(f"step:{step} tokens:{progress.tokens_processed}/{progress.target_tokens} (s={progress.s:.4f}) val_loss:{cur_val:.6f} train_time:{training_time_ms:.0f}ms")
        # checkpoints by tokens (save only if validation improves)
        if master_process and args.save_checkpoint and progress.should_snapshot():
            if cur_val < best_val:
                best_val = cur_val
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
                    progress_state=progress.state_dict(),
                )
            else:
                try:
                    print0(f"No improvement in val loss: best={best_val:.6f}, current={cur_val:.6f}. Skipping checkpoint.")
                except Exception:
                    print0("No improvement in val loss. Skipping checkpoint.")
            progress.mark_snapshot_done()
        # resume training clock
        model.train()
        if use_distributed:
            dist.barrier()
        t0 = time.perf_counter()
        progress.mark_eval_done()

    # --------------- TRAINING SECTION -----------------
    ga_steps = max(1, int(getattr(args, "grad_acc_steps", 1) or 1))
    total_train_loss = 0.0

    for micro_step in range(ga_steps):
        inputs, targets = next(train_loader)
        loss = model(inputs, get_window_size_blocks_s(progress.s), targets)
        # scale loss so that gradients are averaged across micro-steps
        loss_to_backward = loss / ga_steps
        if use_distributed:
            model.require_backward_grad_sync = (micro_step == ga_steps - 1)
        loss_to_backward.backward()
        total_train_loss += float(loss.item())

    # collect the futures for all the optimizers (do distributed grad average once after accumulation)
    opt2futures = {
        opt: ([dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, async_op=True).get_future() for p in params if p.grad is not None]
              if use_distributed else [])
        for opt, params in opt2params.items()
    }
    # set optimization hyperparameters based on s
    s = progress.s
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * get_lr_s(s, args.cooldown_frac)
    for group in optimizer2.param_groups:
        frac = s  # momentum warmup for muon driven by progress
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
    avg_step = f"avg_step:{(approx_training_time_ms-warmup_end) / max(step-9,1):.2f}ms" if step >= 10 else "avg_step: (warmup to step 10)"
    print0(f"step:{step} train_loss:{train_loss_est:.4f} tokens:{progress.tokens_processed}/{progress.target_tokens} (s={progress.s:.4f}) train_time:{approx_training_time_ms:.0f}ms {avg_step}")
    if _wandb_enabled:
        try:
            _wandb.log({
                "train/loss": train_loss_est,
                "train/ppl": math.exp(train_loss_est) if train_loss_est < 20 else float("inf"),
                "tokens": progress.tokens_processed,
                "s": progress.s,
                "train/time_ms": approx_training_time_ms,
                "step": step,
            })
        except Exception as _e:
            print0(f"[warn] wandb.log (train) failed: {_e}")

# End of training: save final checkpoint
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
        progress_state=progress.state_dict(),
    )

print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB")
if _wandb_enabled:
    try:
        _wandb.finish()
    except Exception:
        pass
if use_distributed and dist.is_initialized():
    dist.destroy_process_group()
