import math
import time
from argparse import ArgumentParser
from typing import Iterable, List, Tuple, Dict, Any

import torch
import json
import copy
from datetime import datetime
from pathlib import Path

from training.data_gen import DistributedDataGenerator
from training.optim import get_num_window_blocks


# Utilities for group introspection ---------------------------------------------------------------

def _enumerate_param_groups(optimizers: Iterable[torch.optim.Optimizer]):
    """
    Yield tuples (opt_idx, group_idx, group_dict) for every param group across all optimizers.
    If the optimizer param group contains a 'name' key (added by our optimizer builder), we expose it.
    """
    for oi, opt in enumerate(optimizers):
        for gi, g in enumerate(opt.param_groups):
            yield oi, gi, g


def _group_key(oi: int, gi: int, g: Dict[str, Any]) -> str:
    """Stable identifier string for a param group."""
    name = g.get("name")
    return name if isinstance(name, str) else f"opt{oi}/group{gi}"


# LR sweep ----------------------------------------------------------------------------------------

def lr_sweep(
    model: torch.nn.Module,
    optimizers: List[torch.optim.Optimizer] | torch.optim.Optimizer,
    data_generator: DistributedDataGenerator,
    *,
    # window schedule for attention
    window_schedule: float = 1.0,
    attention_window_tokens: int = 3456,
    window_block_size: int = 128,
    # sweep setup
    num_scales: int = 200,
    scale_min: float = 1e-6,
    scale_max: float = 1.0,
    steps_per_scale: int = 20,
    smooth: float = 0.98,  # EMA on loss (computed within each scale window)
    device: str = "cuda",
    accum_steps: int = 1,
    clip_norm: float | None = None,
    blowup_pct: float = 0.30,  # early stop when EMA > (1+blowup_pct)*best
    # group control
    freeze: Iterable[str | Tuple[int, int]] | None = None,
    sweep_only: Iterable[str | Tuple[int, int]] | None = None,
):
    """
    Sweep a multiplicative LR scale between [scale_min, scale_max] across `num_scales` points.

    Semantics:
    - We apply a scalar s to selected optimizer param groups such that lr_group = base_lr[group] * s.
    - The scalar s increases exponentially from scale_min to scale_max over `num_scales` points.
    - For each scalar value, we run `steps_per_scale` optimizer steps and compute an EMA of the loss
      within that window. The final EMA for the window is recorded as the score for that scalar.

    Group control:
    - Freeze: groups listed in `freeze` keep their base LR throughout the sweep.
    - sweep_only: if provided, restricts sweeping to these groups only (others are frozen).
    - Group identifiers can be:
        * the canonical group name string (if optimizer param_groups include 'name'), e.g., 'embed_params'
        * a tuple (optimizer_index, group_index), e.g., (0, 2)

    Returns (scales, ema_losses, meta) where:
      - scales: list[float] of the applied multiplicative scalar per point
      - ema_losses: list[float] of EMA losses per point (one per scalar)
      - meta: dict with keys:
          'groups': dict[group_key -> {oi, gi, name, base_lr, frozen: bool}],
          'steps_per_scale': int,
          'num_scales': int,
          'scale_min': float,
          'scale_max': float,
    """
    # Normalize optimizers input
    if not isinstance(optimizers, (list, tuple)):
        optimizers = [optimizers]

    # Move model
    model = model.to(device)
    model.train()

    # Build param-group registry and capture base LRs
    group_infos: Dict[str, Dict[str, Any]] = {}

    for oi, gi, g in _enumerate_param_groups(optimizers):
        key = _group_key(oi, gi, g)
        # Ensure base_lr is recorded and stable
        base_lr = g.get("base_lr", g.get("lr", optimizers[oi].defaults.get("lr", 1e-3)))
        g["base_lr"] = float(base_lr)
        group_infos[key] = {
            "oi": oi,
            "gi": gi,
            "name": g.get("name"),
            "base_lr": float(base_lr),
            "frozen": False,  # set below
        }

    # Resolve freeze/sweep sets
    def _matches(gkey: str, spec: str | Tuple[int, int]) -> bool:
        if isinstance(spec, tuple) and len(spec) == 2:
            oi, gi = spec
            return gkey == _group_key(oi, gi, optimizers[oi].param_groups[gi])
        if isinstance(spec, str):
            # name match or identifier match
            info = group_infos[gkey]
            return spec == gkey or (info.get("name") == spec)
        return False

    all_keys = list(group_infos.keys())
    frozen_keys: set[str] = set()
    if freeze:
        for gkey in all_keys:
            if any(_matches(gkey, f) for f in freeze):
                frozen_keys.add(gkey)
    if sweep_only:
        # everything not in sweep_only is frozen
        allowed = {gkey for gkey in all_keys if any(_matches(gkey, s) for s in sweep_only)}
        for gkey in all_keys:
            if gkey not in allowed:
                frozen_keys.add(gkey)

    # Mark frozen flags
    for gkey in all_keys:
        group_infos[gkey]["frozen"] = gkey in frozen_keys

    # Function to set LRs based on a scalar
    def set_lrs(scalar: float):
        for oi, gi, g in _enumerate_param_groups(optimizers):
            key = _group_key(oi, gi, g)
            if key in frozen_keys:
                # keep base LR
                g["lr"] = g.get("base_lr", g.get("lr", 1e-3))
            else:
                g["lr"] = float(g.get("base_lr", 1e-3)) * float(scalar)

    # Initialize sweep (compute geometric positions for scales)
    def _scale_at(i: int) -> float:
        if num_scales <= 1:
            return float(scale_min)
        ratio = max(scale_max, 1e-12) / max(scale_min, 1e-12)
        return float(scale_min) * (ratio ** (i / (num_scales - 1)))

    t0 = time.time()
    print(
        f"[lr_sweep] num_scales={num_scales}, scale_min={scale_min:.3e}, scale_max={scale_max:.3e}, "
        f"steps_per_scale={steps_per_scale}, accum_steps={accum_steps}, smooth={smooth}, "
        f"blowup_pct={blowup_pct*100:.1f}%",
        flush=True,
    )
    print("[lr_sweep] Groups:", flush=True)
    for key in all_keys:
        info = group_infos[key]
        fr = "(frozen)" if info["frozen"] else "(sweep)"
        name = info.get("name") or key
        print(f"  - {name}: base_lr={info['base_lr']:.3e} {fr}", flush=True)

    print_every = max(1, num_scales // 50) if num_scales else 1

    # Capture baselines to reset between scales
    model_baseline = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    opt_baselines = [copy.deepcopy(opt.state_dict()) for opt in optimizers]

    global_best = float("inf")
    lrs_scalars: List[float] = []
    losses: List[float] = []

    # Sweep loop over scales; reset model/optimizer/data at each new scale
    for i in range(num_scales):
        scalar = _scale_at(i)
        # Reset states for fair comparison
        model.load_state_dict(model_baseline, strict=True)
        for oi, opt in enumerate(optimizers):
            opt.load_state_dict(copy.deepcopy(opt_baselines[oi]))
        data_generator.reset()
        set_lrs(scalar)

        ema = None
        early = False
        reason = ""

        for step in range(steps_per_scale):
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)

            total = 0.0
            for _ in range(accum_steps):
                train, target = next(data_generator)
                loss = model(
                    input_seq=train,
                    target_seq=target,
                    sliding_window_num_blocks=get_num_window_blocks(
                        window_schedule,
                        attention_window_tokens=attention_window_tokens,
                        window_block_size=window_block_size,
                    ),
                )
                (loss / accum_steps).backward()
                total += float(loss.detach().item())

            if clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

            for opt in optimizers:
                opt.step()

            val = total / accum_steps
            ema = val if ema is None else smooth * ema + (1 - smooth) * val

            # early stop check within this scale window
            if not math.isfinite(ema):
                early = True
                reason = "non-finite EMA"
                break
            if math.isfinite(global_best) and step > 4 and ema > (1.0 + blowup_pct) * global_best:
                early = True
                reason = "EMA exceeded blowup threshold vs global best"
                break

        # record results for this scale
        lrs_scalars.append(scalar)
        losses.append(ema if ema is not None else float("nan"))

        # update global best
        if ema is not None and math.isfinite(ema) and ema < global_best:
            global_best = ema

        # periodic outer-loop progress line
        elapsed = time.time() - t0
        done = i + 1
        pct = 100.0 * done / max(1, num_scales)
        eta_s = (elapsed / max(1e-9, done)) * max(0, num_scales - done)
        print(
            f"[lr_sweep] {done}/{num_scales} ({pct:.1f}%) scalar={scalar:.3e} ema={losses[-1]:.6f} early={early} eta={eta_s:.1f}s",
            flush=True,
        )

    # summary print
    if losses:
        i_min = min(range(len(losses)), key=lambda i: losses[i])
        print(
            f"[lr_sweep] collected {len(losses)} points; min EMA at step {i_min+1} scalar={lrs_scalars[i_min]:.3e} ema={losses[i_min]:.6f}",
            flush=True,
        )
    else:
        print("[lr_sweep] collected 0 points", flush=True)

    meta = {
        "groups": group_infos,
        "steps_per_scale": steps_per_scale,
        "num_scales": num_scales,
        "scale_min": float(scale_min),
        "scale_max": float(scale_max),
    }
    return lrs_scalars, losses, meta


# Peak selection utility (compatible with old API) -----------------------------------------------

def pick_peak_lr(lrs: List[float], losses: List[float], blowup_pct: float = 0.30, c: float = 0.2):
    i_min = min(range(len(losses)), key=lambda i: losses[i])
    thr = (1.0 + blowup_pct) * losses[i_min]
    i_bu = next((i for i in range(i_min + 1, len(losses)) if losses[i] > thr), len(losses) - 1)
    lr_blowup = lrs[i_bu]
    lr_at_min = lrs[i_min]
    lr_peak = min(c * lr_blowup, 2 * lr_at_min)
    return {"lr_peak": lr_peak, "lr_blowup": lr_blowup, "lr_at_min": lr_at_min}


if __name__ == "__main__":
    from models import get_model_class
    from training.hparams import load_hparams_from_yaml
    import os

    device = "cuda"
    parser = ArgumentParser("Sweep learning rate scales across optimizer param groups")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML training config.")
    parser.add_argument("--num_scales", type=int, default=200)
    parser.add_argument("--steps_per_scale", type=int, default=20)
    parser.add_argument("--scale_min", type=float, default=None, help="Multiplicative LR scale min (default 1e-6)")
    parser.add_argument("--scale_max", type=float, default=None, help="Multiplicative LR scale max (default 1.0)")
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--clip_norm", type=float, default=None)
    parser.add_argument("--smooth", type=float, default=0.98)
    parser.add_argument("--blowup_pct", type=float, default=0.30)
    parser.add_argument(
        "--freeze",
        type=str,
        default="",
        help="Comma-separated list of group names or oi:gi pairs to freeze, e.g., 'embed_params,0:1'",
    )
    parser.add_argument(
        "--sweep_only",
        type=str,
        default="",
        help="If set, only these groups are swept; others frozen. Comma-separated names or oi:gi",
    )
    cli = parser.parse_args()

    from training.hparams import load_hparams_from_yaml

    params = load_hparams_from_yaml(cli.config)
    Model = get_model_class(params.model_class)
    model = Model(
        vocab_size=params.vocab_size,
        num_layers=params.num_layers,
        num_heads=params.num_heads,
        model_dim=params.model_dim,
        max_seq_len=max(params.training_sequence_length, params.val_seq_len),
        head_dim=params.head_dim,
        window_block_size=params.window_block_size,
        eos_token_id=params.eos_token_id,
    )
    model.to(device)
    model = torch.compile(model, dynamic=False)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    data_loader = DistributedDataGenerator(
        params.train_shards,
        world_size * params.training_sequence_length,
        rank=rank,
        world_size=world_size,
        device=device,
    )

    from training.optim import build_optimizers_from_cfg

    optimizers = build_optimizers_from_cfg(
        cfg_list=params.optimizers, model=model, rank=rank, world_size=world_size
    )

    def _parse_spec(s: str) -> List[str | Tuple[int, int]]:
        out: List[str | Tuple[int, int]] = []
        for tok in [t for t in s.split(",") if t.strip()]:
            tok = tok.strip()
            if ":" in tok:
                a, b = tok.split(":", 1)
                try:
                    out.append((int(a), int(b)))
                except Exception:
                    pass
            else:
                out.append(tok)
        return out

    freeze = _parse_spec(cli.freeze)
    sweep_only = _parse_spec(cli.sweep_only)

    # Resolve deprecated aliases
    n_scales = cli.num_scales if cli.steps is None else cli.steps
    sc_min = cli.scale_min if cli.scale_min is not None else (cli.lr_min if cli.lr_min is not None else 1e-6)
    sc_max = cli.scale_max if cli.scale_max is not None else (cli.lr_max if cli.lr_max is not None else 1.0)

    lrs, losses, meta = lr_sweep(
        model,
        optimizers=optimizers,
        data_generator=data_loader,
        attention_window_tokens=params.attention_window_tokens,
        window_block_size=params.window_block_size,
        num_scales=n_scales,
        scale_min=sc_min,
        scale_max=sc_max,
        steps_per_scale=cli.steps_per_scale,
        accum_steps=cli.accum_steps,
        clip_norm=cli.clip_norm,
        smooth=cli.smooth,
        blowup_pct=cli.blowup_pct,
        freeze=freeze,
        sweep_only=sweep_only,
    )

    # Log JSON and print a table
    results = [{"index": i, "scale": float(s), "ema": float(l)} for i, (s, l) in enumerate(zip(lrs, losses))]
    log_obj = {"results": results, "meta": meta}
    print(json.dumps(log_obj, indent=2))
    # Write to logs dir
    logs_dir = Path(__file__).resolve().parent.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    out_path = logs_dir / f"lr_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(log_obj, f, indent=2)
    print(f"[lr_sweep] wrote JSON results to {out_path}")

    # Print ASCII table
    print("\nIdx  Scale        EMA")
    for i, (s, l) in enumerate(zip(lrs, losses)):
        print(f"{i:3d}  {s:10.3e}  {l:10.6f}")

    print(pick_peak_lr(lrs, losses))
