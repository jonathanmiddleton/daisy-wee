import torch
from torch import Tensor
from functools import lru_cache
import torch.distributed as dist
from typing import Any
from torch import nn


def derive_named_param_groups(model: nn.Module) -> dict[str, list[nn.Parameter]]:
    """Derive canonical named parameter groups from the model.

    Returns a mapping with keys:
      - hidden_matrix_params: all module parameters with ndim >= 2 inside model.blocks (sorted by size desc)
      - embed_params: parameters of model.embed and model.value_embeds
      - scalar_params: [model.scalars]
      - head_params: [model.lm_head_w]
    """
    # Hidden matrices (e.g., linear/attention weight matrices) from transformer blocks
    hidden_matrix_params = sorted(
        (p for p in model.blocks.parameters() if p.ndim >= 2),
        key=lambda x: x.size(),
        reverse=True,
    )
    # Embedding parameters
    embed_params = [*model.embed.parameters(), *model.value_embeds.parameters()]
    # Learned scalar gates
    scalar_params = [model.scalars]
    # Output head weights
    head_params: list[nn.Parameter] = [model.lm_head_w]

    # Sanity: ensure exact partitioning of all model parameters
    params_collections = [hidden_matrix_params, embed_params, scalar_params, head_params]
    optimized_parameters_set = {p for params in params_collections for p in params}
    all_params = set(model.parameters())
    assert optimized_parameters_set == all_params
    assert len(optimized_parameters_set) == sum(len(lst) for lst in params_collections)

    return {
        "hidden_matrix_params": hidden_matrix_params,
        "embed_params": embed_params,
        "scalar_params": scalar_params,
        "head_params": head_params,
    }

# -----------------------------------------------------------------------------
# Muon optimizer

def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ∈ [1 - l, 1 + r], which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for a, b, c in [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]:
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

@torch.compile
def update(acc_bf16_view_u16: Tensor, mantissa: Tensor, momentum_buffer: Tensor, grad: Tensor, momentum: Tensor, eff_lr: Tensor, eff_weight_decay: Tensor):
    assert acc_bf16_view_u16.dtype == mantissa.dtype == torch.uint16
    grad = grad.float()
    momentum_buffer.copy_(momentum * momentum_buffer + (1 - momentum) * grad)
    v = zeropower_via_newtonschulz5(momentum * momentum_buffer + (1 - momentum) * grad)

    acc_m_u32 = (acc_bf16_view_u16.to(torch.uint32) << 16) | mantissa.to(torch.uint32)
    acc_m_u32.view(torch.float32).mul_(1 - eff_weight_decay)
    acc_m_u32.view(torch.float32).add_(other=v, alpha=-eff_lr)
    acc_bf16_view_u16.copy_((acc_m_u32 >> 16).to(torch.uint16))
    mantissa.copy_(acc_m_u32.to(torch.uint16))


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Warning: This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    """
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)
        assert all(p.dtype == torch.bfloat16 for group in self.param_groups for p in group["params"])

    @torch.no_grad()
    def step(self):
        futures: list[torch.Future] = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * self.world_size
            momentum = torch._as_tensor_fullprec(group["momentum"])
            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    # Skip if no gradient for this parameter on this step/rank
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["mantissa"] = torch.zeros_like(p, dtype=torch.uint16)
                        state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
                    update(
                        p.view(torch.uint16), state["mantissa"], state["momentum_buffer"],
                        p.grad, momentum,
                        eff_lr=torch._as_tensor_fullprec(group["lr"] * max(1, p.size(-2) / p.size(-1)) ** 0.5),
                        eff_weight_decay=torch._as_tensor_fullprec(group["lr"] * group["weight_decay"] * getattr(p, "wd_mul", 1.0)),
                    )
                if self.world_size > 1 and dist.is_available() and dist.is_initialized():
                    futures.append(
                        dist.all_gather(
                            params_pad[base_i:base_i + self.world_size],
                            params_pad[base_i + self.rank],
                            async_op=True
                        ).get_future()
                    )
                if futures:
                    torch.futures.collect_all(futures).wait()

# learning rate schedule: stable then decay
# New s-based schedule variants that take normalized progress s ∈ [0,1]

def get_lr_s(s: float, cooldown_frac: float) -> float:
    """Return LR scale factor given normalized progress s in [0,1].
    1.0 during main phase, cosine-linear or linear decay during cooldown.
    Mirrors previous get_lr(step, num_iterations, cooldown_frac) semantics but decoupled from steps.
    """
    # Clamp progress to [0, 1]
    x = 0.0 if s < 0 else (1.0 if s > 1 else s)
    if x < 1 - cooldown_frac:
        return 1.0
    else:
        # Linear decay to zero over the cooldown tail
        return max((1 - x) / max(cooldown_frac, 1e-8), 0.0)

# Global flag to force full-sized attention windows regardless of training progress
_force_full_windows: bool = False

def set_full_windows(flag: bool):
    global _force_full_windows
    _force_full_windows = bool(flag)

# attention window size schedule: linearly increase
def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

@lru_cache(1)
def get_window_size_blocks_helper(window_size_tokens: int, window_block_size: int):
    return torch.tensor(window_size_tokens // window_block_size, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

def get_num_window_blocks(schedule: float, *, attention_window_len: int, window_block_size: int) -> torch.Tensor:
    """Attention window schedule driven by normalized progress schedule s∈[0,1].
    Returns the number of blocks for the sliding window, parameterized by attention_window_len and window_block_size.
    """
    if _force_full_windows:
        x = 1.0
    else:
        x = 0.0 if schedule < 0 else (1.0 if schedule > 1 else schedule)
    # Cubic increase (by @jadenj3o)
    factor = 4 * x ** 3 - 6 * x ** 2 + 3 * x
    window_tokens = next_multiple_of_n(attention_window_len * factor, n=window_block_size)
    return get_window_size_blocks_helper(window_tokens, window_block_size)

def resolve_optimizer_class(opt_type: str):
    """Resolve optimizer type name to a class.
    Supports torch.optim.* and custom Muon optimizer.

    Important: prefer the local Muon implementation (supports batched >2D params)
    over torch.optim.Muon, which only supports 2D parameters.
    """
    if opt_type == "Muon":
        # Always use the local Muon defined in this module
        return Muon
    if hasattr(torch.optim, opt_type):
        return getattr(torch.optim, opt_type)
    raise ValueError(f"Unknown optimizer type: {opt_type}")

def get_referenced_groups(cfg_list: list[dict[str, Any]]) -> list[str]:
    referenced_group_names: set[str] = set()
    for idx, opt_cfg in enumerate(cfg_list):
        pg_cfgs = opt_cfg.get("params")
        for pg in pg_cfgs:
            referenced_group_names.add(pg.get("group"))
    return list(referenced_group_names)


def build_optimizers_from_cfg(
    *,
    cfg_list: list[dict[str, Any]],
    model: nn.Module,
    rank: int,
    world_size: int,
    frozen_groups: list[str] | None = None,
) -> list[torch.optim.Optimizer]:
    """Build optimizers from configuration list and sets no_grad on frozen groups.

    Each item in cfg_list is a dict with keys:
      - type: optimizer class name (e.g., "AdamW" or "Muon")
      - params: list of param-group dicts containing at least {group: <name>} and any group overrides like lr
      - other keys are passed as kwargs to the optimizer constructor (betas, eps, weight_decay, fused, momentum, ...)
      - cfg_list should include frozen groups as well, if any

    This function also enforces that the union of all param groups referenced in the config
    exactly matches the required named groups for the model (no missing or extra groups).
    """
    if not isinstance(cfg_list, list) or not cfg_list:
        raise ValueError("optimizers config must be a non-empty list")

    # Derive canonical named parameter groups from the model
    param_groups_by_name = derive_named_param_groups(model)
    required_group_names = set(param_groups_by_name.keys())
    frozen_groups = frozen_groups or []
    for name in frozen_groups:
        if name not in required_group_names:
            raise ValueError(f"Frozen group '{name}' is not present in the model")
        for p in param_groups_by_name[name]:
            p.requires_grad_(False)

    # Collect all group names referenced across all optimizers in config
    referenced_group_names: set[str] = set()

    optimizers: list[torch.optim.Optimizer] = []

    for idx, opt_cfg in enumerate(cfg_list):
        if not isinstance(opt_cfg, dict):
            raise ValueError(f"Optimizer #{idx} must be a mapping, got {type(opt_cfg).__name__}")
        opt_type = opt_cfg.get("type")
        if not opt_type:
            raise ValueError(f"Optimizer #{idx} is missing 'type'")

        pg_cfgs = opt_cfg.get("params")
        if not isinstance(pg_cfgs, list) or not pg_cfgs:
            raise ValueError(f"Optimizer '{opt_type}' requires a non-empty 'params' list")

        param_groups: list[dict[str, Any]] = []
        for pg in pg_cfgs:
            if not isinstance(pg, dict):
                raise ValueError(f"param group for optimizer '{opt_type}' must be a mapping, got {type(pg).__name__}")
            name = pg.get("group")
            if name not in param_groups_by_name:
                raise ValueError(
                    f"Unknown param group '{name}' for optimizer '{opt_type}'. "
                    f"Known groups: {sorted(param_groups_by_name.keys())}"
                )
            if frozen_groups and pg.get("group") in frozen_groups:
                continue
            referenced_group_names.add(name)
            group_opts = {k: v for k, v in pg.items() if k != "group"}
            # Preserve the canonical group name for downstream tooling (e.g., LR sweeps)
            group_opts["name"] = name
            group_opts["params"] = param_groups_by_name[name]
            param_groups.append(group_opts)

        if not param_groups:
            continue
        OptClass = resolve_optimizer_class(opt_type)
        opt_kwargs = {k: v for k, v in opt_cfg.items() if k not in ("type", "params")}

        if OptClass is Muon:
            opt_kwargs.setdefault("rank", rank)
            opt_kwargs.setdefault("world_size", world_size)

        optimizer = OptClass(param_groups, **opt_kwargs)
        optimizers.append(optimizer)

    # Validate exhaustive coverage: all required groups must be present exactly once across optimizers
    missing = required_group_names - referenced_group_names - set(frozen_groups)
    extra = referenced_group_names - required_group_names
    if missing or extra:
        msgs = []
        if missing:
            msgs.append(f"missing groups: {sorted(missing)}")
        if extra:
            msgs.append(f"unknown groups: {sorted(extra)}")
        raise ValueError(
            "Optimizer config param groups must be exhaustive and valid; " + ", ".join(msgs) +
            f". Required groups: {sorted(required_group_names)}"
        )

    # Additionally ensure that the parameters are fully covered without overlap
    # Only consider trainable parameters (requires_grad=True) for coverage; frozen groups are excluded
    covered_params = {p for name in referenced_group_names for p in param_groups_by_name[name]}
    all_trainable_params = {p for p in model.parameters() if p.requires_grad}
    if covered_params != all_trainable_params:
        raise ValueError(
            "Param groups in optimizer config do not cover trainable model parameters exactly. "
            f"covered={len(covered_params)} total_trainable={len(all_trainable_params)}"
        )

    return optimizers
