import math, torch, time
from argparse import ArgumentParser
from training.data_gen import DistributedDataGenerator
from training.optim import build_optimizers_from_cfg, get_num_window_blocks

# noinspection PyShadowingNames
def lr_range_test(
    model,
    optimizers,              # optimizer or list[optimizer]
    data_generator: DistributedDataGenerator,
    window_schedule=1.0,
    attention_window_tokens: int = 3456,
    window_block_size: int = 128,
    steps=2000,
    lr_min=1e-6,
    lr_max=1.0,
    smooth=0.98,             # EMA on loss
    device='cuda',
    accum_steps=1,
    clip_norm=None,          # e.g., 1.0 or None
    blowup_pct=0.30          # early stop when EMA > (1+blowup_pct)*best
):
    if not isinstance(optimizers, (list, tuple)):
        optimizers = [optimizers]

    _model = model.to(device)
    _model.train()

    model_state = {k: v.detach().cpu().clone() for k, v in _model.state_dict().items()}
    opt_states  = [opt.state_dict() for opt in optimizers]

    def set_lr(lr_scalar):
        for opt in optimizers:
            for g in opt.param_groups:
                s = g.get('lr_scale', 1.0)
                g['lr'] = s * lr_scalar

    lr    = lr_min
    mult  = (lr_max / lr_min) ** (1.0 / max(1, steps))
    set_lr(lr)

    t0 = time.time()
    print(f"[lr_range_test] steps={steps}, lr_min={lr_min:.3e}, lr_max={lr_max:.3e}, accum_steps={accum_steps}, smooth={smooth}, blowup_pct={blowup_pct*100:.1f}%", flush=True)
    print_every = max(1, steps // 50) if steps else 1

    ema, best = None, float('inf')
    best_step = -1
    lrs, losses = [], []

    for step in range(steps):
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        total = 0.0
        for _ in range(accum_steps):
            train, target = next(data_generator)
            loss = _model(input_seq=train, target_seq=target, sliding_window_num_blocks=get_num_window_blocks(window_schedule, attention_window_tokens=attention_window_tokens, window_block_size=window_block_size))
            (loss / accum_steps).backward()
            total += float(loss.detach().item())

        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(_model.parameters(), clip_norm)

        for opt in optimizers:
            opt.step()

        val = total / accum_steps
        ema = val if ema is None else smooth * ema + (1 - smooth) * val

        lrs.append(lr)
        losses.append(ema)

        # periodic progress line
        if (step % print_every == 0) or (step == steps - 1):
            elapsed = time.time() - t0
            done = step + 1
            pct = 100.0 * done / max(1, steps)
            eta_s = (elapsed / max(1e-9, done)) * max(0, steps - done)
            best_str = f"{best:.6f}" if best < float('inf') else "NaN"
            print(
                f"[lr_range_test] {done}/{steps} ({pct:.1f}%) lr={lr:.3e} loss={val:.6f} ema={ema:.6f} best={best_str} eta={eta_s:.1f}s",
                flush=True,
            )

        # early stop check
        early = False
        reason = ""
        if not math.isfinite(ema):
            early = True
            reason = "non-finite EMA"
        elif step > 20 and ema > (1.0 + blowup_pct) * best:
            early = True
            reason = "EMA exceeded blowup threshold"

        if early:
            thr = (1.0 + blowup_pct) * best if math.isfinite(best) else float('nan')
            print(
                f"[lr_range_test] Early stopping at step {step+1}: lr={lr:.3e} ema={ema:.6f} best={best:.6f} threshold={thr:.6f} ({reason})",
                flush=True,
            )
            break

        if ema < best:
            best = ema
            best_step = step

        lr *= mult
        set_lr(lr)

    # summary print
    if losses:
        i_min = min(range(len(losses)), key=lambda i: losses[i])
        print(
            f"[lr_range_test] collected {len(losses)} points; min EMA at step {i_min+1} lr={lrs[i_min]:.3e} ema={losses[i_min]:.6f}",
            flush=True,
        )
    else:
        print("[lr_range_test] collected 0 points", flush=True)

    _model.load_state_dict(model_state)
    for opt, st in zip(optimizers, opt_states):
        opt.load_state_dict(st)

    return lrs, losses

# noinspection PyShadowingNames
def pick_peak_lr(lrs, losses, blowup_pct=0.30, c=0.2):
    i_min = min(range(len(losses)), key=lambda i: losses[i])
    thr = (1.0 + blowup_pct) * losses[i_min]
    i_bu = next((i for i in range(i_min + 1, len(losses)) if losses[i] > thr), len(losses) - 1)
    lr_blowup = lrs[i_bu]
    lr_at_min = lrs[i_min]
    lr_peak   = min(c * lr_blowup, 2 * lr_at_min)
    return {'lr_peak': lr_peak, 'lr_blowup': lr_blowup, 'lr_at_min': lr_at_min}

if __name__ == '__main__':
    from models import get_model_class
    from training.optim import build_optimizers_from_cfg, get_num_window_blocks
    from training.hparams import load_hparams_from_yaml
    import os
    device = 'cuda'
    argument_parser = ArgumentParser('Exponentially sweep a learning rate range.')
    argument_parser.add_argument('--config', type=str, required=True, help='Path to YAML training config.')
    cli = argument_parser.parse_args()
    params = load_hparams_from_yaml(cli.config)
    Model = get_model_class(params.model_type)
    model = Model(
        vocab_size=params.vocab_size,
        num_layers=params.num_layers,
        num_heads=params.num_heads,
        model_dim=params.model_dim,
        max_seq_len=max(params.training_sequence_length, params.val_seq_len),
        head_dim=params.head_dim,
        window_block_size=params.window_block_size,
    )
    model.to(device)
    model = torch.compile(model, dynamic=False)
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    data_loader = DistributedDataGenerator(params.train_shards, world_size * params.training_sequence_length, rank=rank, world_size=world_size, device=device)
    optimizers = build_optimizers_from_cfg(cfg_list=params.optimizers, model=model, rank=rank, world_size=world_size)

    lrs, losses = lr_range_test(
        model,
        optimizers=optimizers,
        data_generator=data_loader,
        attention_window_tokens=params.attention_window_tokens,
        window_block_size=params.window_block_size,
    )

    print(pick_peak_lr(lrs, losses))