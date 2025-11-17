from __future__ import annotations

import torch
from torch import Tensor
from typing import Optional, Any
from tools.master_logger import MasterLogger
import torch.nn.functional as F

#TODO enable logger only for debug, no-op for production
logger = MasterLogger


from torch.library import custom_op, register_kernel, register_fake, register_autograd

from fla.ops.kda.gate import kda_gate_fwd as _kda_gate_fwd_impl
from fla.ops.kda.gate import kda_gate_bwd as _kda_gate_bwd_impl

def _meta_like(x: Tensor) -> Tensor:
    return torch.empty_like(x, device="meta")

def _meta_opt_like(x: Optional[Tensor]) -> Optional[Tensor]:
    return torch.empty_like(x, device="meta") if isinstance(x, Tensor) else None

def _gate_impl(
    g: Tensor,
    A_log: Tensor,
    head_k_dim: int,
    g_bias: Optional[Tensor],
    b: Optional[Tensor],
    beta: float,
    threshold: float,
) -> Tensor:
    *prefix, last = g.shape
    torch._assert(last % head_k_dim == 0, "last dim must be multiple of head_k_dim")
    H = last // head_k_dim

    g_view = g.view(*prefix, H, head_k_dim)

    if g_bias is not None:
        g_view = g_view + g_bias.view(*([1] * len(prefix)), H, head_k_dim)
    if b is not None:
        g_view = g_view + b.view(*([1] * len(prefix)), H, head_k_dim)

    A = A_log.exp()
    dt = F.softplus(g_view - threshold)
    y = -beta * A * dt

    return y.view(*prefix, H * head_k_dim)


@custom_op(
    "fla_kda::gate",
    mutates_args=(),
    schema="(Tensor g, Tensor A_log, int head_k_dim, Tensor? g_bias, Tensor? b, float beta, float threshold) -> Tensor",
)
def _gate_default(g, A_log, head_k_dim, g_bias, b, beta, threshold):
    s = torch.sigmoid(A_log)
    y = g * s
    if isinstance(g_bias, Tensor): y = y + g_bias
    if isinstance(b, Tensor): y = y + b
    return y

@_gate_default.register_kernel("cuda")
def _gate_cuda_kernel(g, A_log, head_k_dim, g_bias, b, beta, threshold):

    return _kda_gate_fwd_impl(g, A_log, head_k_dim, g_bias, b, beta, threshold)

@_gate_default.register_fake
def _gate_fake(g, A_log, head_k_dim, g_bias, b, beta, threshold):
    return _meta_like(g)


@custom_op(
    "fla_kda::gate_bwd",
    mutates_args=(),
    schema=("("
            "Tensor grad_out, Tensor g, Tensor A_log, int head_k_dim, Tensor? g_bias, Tensor? b, float beta, float threshold"
            ") -> (Tensor, Tensor, Tensor?, Tensor?)"),
)
def _gate_bwd_default(grad_out, g, A_log, head_k_dim, g_bias, b, beta, threshold):
    s = torch.sigmoid(A_log)
    dg = grad_out * s
    dA = grad_out * g * s * (1 - s)
    dgb = grad_out if isinstance(g_bias, Tensor) else None
    dbb = grad_out if isinstance(b, Tensor) else None
    return dg, dA, dgb, dbb

@_gate_bwd_default.register_kernel("cuda")
def _gate_bwd_cuda_kernel(grad_out, g, A_log, head_k_dim, g_bias, b, beta, threshold):
    # if _kda_gate_bwd_impl is None:
    #     return _gate_bwd_default(grad_out, g, A_log, head_k_dim, g_bias, b, beta, threshold)
    # our gate op returns ONLY the main tensor, so there is no grad for the optional b_sigmoid
    gb = None
    return _kda_gate_bwd_impl(grad_out, g, A_log, head_k_dim, g_bias, b, gb, beta, threshold)

@_gate_bwd_default.register_fake
def _gate_bwd_fake(grad_out, g, A_log, head_k_dim, g_bias, b, beta, threshold):
    return _meta_like(g), _meta_like(A_log), _meta_opt_like(g_bias), _meta_opt_like(b)


def _gate_setup(ctx, inputs, output):
    g, A_log, head_k_dim, g_bias, b, beta, threshold = inputs
    ctx.head_k_dim = int(head_k_dim)
    ctx.beta = float(beta)
    ctx.threshold = float(threshold)
    ctx.has_gb = isinstance(g_bias, Tensor)
    ctx.has_b = isinstance(b, Tensor)
    ctx.save_for_backward(g, A_log, (g_bias if ctx.has_gb else torch.tensor(0, device=g.device)),
                          (b if ctx.has_b else torch.tensor(0, device=g.device)))

def _gate_backward(ctx, grad_out: Tensor):
    g, A_log, g_bias, b = ctx.saved_tensors
    dg, dA, dgb, dbb = torch.ops.fla_kda.gate_bwd(
        grad_out, g, A_log, ctx.head_k_dim, (g_bias if ctx.has_gb else None),
        (b if ctx.has_b else None), ctx.beta, ctx.threshold
    )
    return dg, dA, None, (dgb if ctx.has_gb else None), (dbb if ctx.has_b else None), None, None

_gate_default.register_autograd(_gate_backward, setup_context=_gate_setup)

@custom_op("fla_kda::chunk", mutates_args=())
def kda_chunk(
    q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor,
    initial_state: Optional[Tensor],
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool,
    cu_seqlens: Optional[Tensor],
) -> tuple[Tensor, Tensor]:
    from fla.ops.kda import chunk_kda
    return chunk_kda(
        q=q, k=k, v=v, g=g, beta=beta,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
    )


@kda_chunk.register_fake
def _(
    q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor,
    initial_state: Optional[Tensor],
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool,
    cu_seqlens: Optional[Tensor],
) -> tuple[Tensor, Tensor]:
    o = torch.empty(q.shape[:-1] + (v.shape[-1],), dtype=v.dtype, device="meta")
    fs = torch.empty_like(initial_state, device="meta") if output_final_state and isinstance(initial_state, Tensor) else None
    return o, fs

def _chunk_setup(ctx, inputs, output):
    q, k, v, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel, cu_seqlens = inputs
    ctx.output_final_state = bool(output_final_state)
    ctx.use_qk = bool(use_qk_l2norm_in_kernel)
    ctx.has_init = isinstance(initial_state, Tensor)
    ctx.has_cu = isinstance(cu_seqlens, Tensor)
    to_save = [q, k, v, g, beta]
    if ctx.has_init: to_save.append(initial_state)
    if ctx.has_cu:   to_save.append(cu_seqlens)
    ctx.save_for_backward(*to_save)


def _chunk_backward(ctx, *grads):
    # Flatten grads in case they're wrapped in extra containers
    flat_grads = []
    for g in grads:
        if isinstance(g, (list, tuple)):
            flat_grads.extend(g)
        else:
            flat_grads.append(g)

    if len(flat_grads) == 1:
        grad_o = flat_grads[0]
        grad_rec = None
    elif len(flat_grads) == 2:
        grad_o, grad_rec = flat_grads
    else:
        raise ValueError(f"Expected 1 or 2 gradients, got {len(flat_grads)}")

    saved = list(ctx.saved_tensors)
    q, k, v, g, beta = saved[:5]
    off = 5
    init = saved[off] if ctx.has_init else None; off += int(ctx.has_init)
    cu   = saved[off] if ctx.has_cu   else None
    grad_rec = grad_rec if ctx.output_final_state else None
    result = torch.ops.fla_kda.chunk_bwd(
        grad_o, grad_rec, q, k, v, g, beta, init,
        ctx.output_final_state, ctx.use_qk, cu
    )
    # result is a list [dq, dk, dv, dg, dbeta, dinit]
    # Unpack and return as tuple (required by autograd)
    dq, dk, dv, dg, dbeta, dinit = result
    if not ctx.has_init:
        dinit = None
    return dq, dk, dv, dg, dbeta, dinit, None, None, None

kda_chunk.register_autograd(_chunk_backward, setup_context=_chunk_setup)

@custom_op("fla_kda::chunk_bwd", mutates_args=())
def kda_chunk_bwd(
    grad_o: Tensor, grad_rec: Optional[Tensor],
    q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor,
    initial_state: Optional[Tensor],
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool,
    cu_seqlens: Optional[Tensor],
) -> list[Tensor]:
    try:
        from fla.ops.kda import chunk_kda_bwd
        result = chunk_kda_bwd(
            grad_o, grad_rec, q, k, v, g, beta, initial_state,
            output_final_state, use_qk_l2norm_in_kernel, cu_seqlens
        )
        # Convert tuple to list if necessary
        if isinstance(result, tuple):
            return list(result)
        return result
    except Exception:
        # Correctness fallback (slower): differentiate through forward
        with torch.enable_grad():
            q_, k_, v_, g_, beta_ = [t.detach().requires_grad_(True) for t in (q, k, v, g, beta)]
            init_ = initial_state.detach().requires_grad_(True) if isinstance(initial_state, Tensor) else None
            from fla.ops.kda import chunk_kda as _fwd
            o, rec = _fwd(q=q_, k=k_, v=v_, g=g_, beta=beta_,
                          initial_state=init_, output_final_state=output_final_state,
                          use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, cu_seqlens=cu_seqlens)
            outs, gouts = (o,), (grad_o,)
            if output_final_state:
                outs = (o, rec)
                gouts = (grad_o, grad_rec if isinstance(grad_rec, Tensor) else torch.zeros_like(rec))
            grads = torch.autograd.grad(outs, tuple([t for t in (q_, k_, v_, g_, beta_, init_) if t is not None]),
                                        grad_outputs=gouts, allow_unused=True)
            it = iter(grads)
            dq, dk, dv, dg, dbeta = next(it), next(it), next(it), next(it), next(it)
            dinit = next(it) if isinstance(initial_state, Tensor) else None
            return [dq, dk, dv, dg, dbeta, dinit]

@kda_chunk_bwd.register_fake
def _(
    grad_o: Tensor, grad_rec: Optional[Tensor],
    q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor,
    initial_state: Optional[Tensor],
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool,
    cu_seqlens: Optional[Tensor],
) -> list[Tensor]:
    return [_meta_like(q), _meta_like(k), _meta_like(v),
            _meta_like(g), _meta_like(beta),
            (_meta_like(initial_state) if isinstance(initial_state, Tensor) else torch.empty(0, device="meta"))]



def kda_gate(g, A_log, head_k_dim, g_bias, b, beta, threshold):
    if b is None:
        # construct a bias to avoid a defect in fla.ops.kda.gate when passing None
        s = g.shape
        if g.ndim == 2:
            s = torch.Size((s[0], s[-1] // head_k_dim))
        elif g.ndim == 3:
            s = torch.Size((s[0], s[1], s[-1] // head_k_dim))
        else:
            raise ValueError(f"Unsupported number of dimensions: {g.ndim}")
        b = torch.zeros(s, device=g.device, dtype=g.dtype)
    return torch.ops.fla_kda.gate(g, A_log, head_k_dim, g_bias, b, beta, threshold)

def kda_chunk(q, k, v, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel, cu_seqlens):
    return torch.ops.fla_kda.chunk(q, k, v, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel, cu_seqlens)
