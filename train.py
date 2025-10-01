import dataclasses
import os
import sys
import time
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dataclasses import dataclass, fields as dataclass_fields, asdict

from models.gpt_core import GPTCore
from training.data_gen import distributed_data_generator
from training.optim import Muon
from training.optim import get_lr, get_window_size_blocks

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

args = load_hparams_from_yaml(sys.argv[1] if len(sys.argv) > 1 else None)
for s in sys.argv[1:]:
    if s.startswith('--init-checkpoint=') or s.startswith('--init_checkpoint='):
        args.init_checkpoint = s.split('=', 1)[1]

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

def print0(st):
    if master_process:
        print(st)

print0(json.dumps(asdict(args), indent=2, sort_keys=True))

########################################
#    Construct model and optimizer     #
########################################


model: nn.Module = GPTCore(vocab_size=args.vocab_size, num_layers=args.num_layers, num_heads=args.num_heads, model_dim=args.model_dim,
                           max_seq_len=max(args.max_seq_len, args.val_seq_len), head_dim=args.head_dim).cuda()
best_val_from_ckpt = None
resume_from_step = None
if args.init_checkpoint:
    _obj = torch.load(args.init_checkpoint, map_location=device)
    _sd = _obj.get('model', _obj) if isinstance(_obj, dict) else _obj
    if isinstance(_sd, dict) and any(k.startswith('_orig_mod.') for k in _sd.keys()):
        _sd = {k.replace('_orig_mod.', '', 1): v for k, v in _sd.items()}
    _missing, _unexpected = model.load_state_dict(_sd, strict=False)
    print0(f"init_checkpoint:{args.init_checkpoint} missing:{len(_missing)} unexpected:{len(_unexpected)}")
    if isinstance(_obj, dict) and "best_val" in _obj:
        best_val_from_ckpt = float(_obj["best_val"])
    if isinstance(_obj, dict) and "step" in _obj:
        resume_from_step = int(_obj["step"])
    print(f"Resuming from step {resume_from_step} with best val {best_val_from_ckpt}")

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
train_loader = distributed_data_generator(args.train_files, world_size * args.max_seq_len, rank, world_size)
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
if resume_from_step is not None:
    train_steps = resume_from_step + args.num_iterations
    tokens_seen = resume_from_step * tokens_per_step
    last_val_tokens = tokens_seen - last_val_tokens
for step in range(train_steps + 1):
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
        val_loss = 0
        with torch.no_grad():
            for _ in range(val_steps):
                inputs, targets = next(val_loader)
                val_loss += model(inputs, get_window_size_blocks(step, train_steps), targets)
        val_loss /= val_steps
        del val_loader
        tokens_since_last = tokens_seen - last_val_tokens
        if last_val_loss is not None and tokens_since_last > 0:
            dpt = (val_loss.item() - last_val_loss) / tokens_since_last
            ema_dloss_per_token = dpt if ema_dloss_per_token is None else 0.7 * ema_dloss_per_token + 0.3 * dpt
            print0(f"delta loss per 1e9 tokens:{dpt * 1e9:.6f} [ema:{(ema_dloss_per_token or 0.0) * 1e9:.6f}]")
        last_val_loss = float(val_loss.item())
        last_val_tokens = tokens_seen

        if use_distributed:
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        print0( f"step:{step}/{train_steps} val_loss:{val_loss:.6f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms")

        cur_val = float(val_loss.item())

        if master_process:
            val_iter += 1
            improved = cur_val < best_val

            if improved:
                best_val = cur_val

            if (args.save_checkpoint and improved
                    and val_iter % args.val_snapshot_every == 0
                    and val_iter > args.snapshot_skip):
                os.makedirs("checkpoints", exist_ok=True)
                fname = f"checkpoints/checkpoint-run{run_id}.pt"
                _model_to_state = model._orig_mod if hasattr(model, "_orig_mod") else model
                log = dict(
                    step=step,
                    model=_model_to_state.state_dict(),
                    optimizers=[opt.state_dict() for opt in optimizers],
                    best_val=best_val,
                )
                torch.save(log, fname)
                print0(f"Saved checkpoint to {fname} with val loss {cur_val:.6f}")


        model.train()
        # start the clock again
        if use_distributed:
            dist.barrier()
        t0 = time.perf_counter()

    if last_step:
        if master_process and args.save_checkpoint:
            _model_to_state = model._orig_mod if hasattr(model, "_orig_mod") else model
            log = dict(step=step, model=_model_to_state.state_dict(),
                       optimizers=[opt.state_dict() for opt in optimizers],
                       best_val=best_val)
            os.makedirs(f"checkpoints", exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            fname = f"checkpoints/{ts}-step{step:06d}-run{run_id}.pt"
            torch.save(log, fname)

        break

    # --------------- TRAINING SECTION -----------------
    inputs, targets = next(train_loader)
    model(inputs,get_window_size_blocks(step, train_steps), targets).backward()
    opt2futures = {
        opt: ([dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, async_op=True).get_future() for p in params if p.grad is not None]
              if use_distributed else [])
        for opt, params in opt2params.items()
    }
    # set optimization hyperparameters
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * get_lr(step, args.num_iterations, args.cooldown_frac)
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
