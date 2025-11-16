from __future__ import annotations

import typing
import torch
from torch import Tensor
from typing import Optional, Any

try:
    from torch.library import custom_op, register_kernel, register_fake, register_autograd
except Exception:
    from torch.library import custom_op, register_kernel  # PyTorch <=2.3 compatibility
    def register_fake(op_name, func=None, *, lib=None, allow_override=False):
        from torch.library import register_fake as impl_abstract
        return impl_abstract(op_name) if func is None else impl_abstract(op_name, func)

# ---- optional project kernels (rename if your symbols differ) ----
try:
    from fla.ops.kda.gate import fused_kda_gate as _kda_gate_cuda
    from fla.ops.kda.gate import fused_kda_gate_bwd as _kda_gate_bwd_cuda
except Exception:
    _kda_gate_cuda = None
    _kda_gate_bwd_cuda = None

try:
    from fla.ops.kda import chunk_kda as _kda_chunk_cuda
    from fla.ops.kda import chunk_kda_bwd as _kda_chunk_bwd_cuda
except Exception:
    _kda_chunk_cuda = None
    _kda_chunk_bwd_cuda = None

try:
    from fla.ops.kda import fused_recurrent_kda as _rmsnorm_gated_cuda
    from fla.ops.kda import fused_recurrent_kda_bwd as _rmsnorm_gated_bwd_cuda
except Exception:
    _rmsnorm_gated_cuda = None
    _rmsnorm_gated_bwd_cuda = None

def _meta_like(x: Tensor) -> Tensor:
    return torch.empty_like(x, device="meta")

def _meta_opt_like(x: Optional[Tensor], like: Tensor) -> Optional[Tensor]:
    return torch.empty_like(x, device="meta") if isinstance(x, Tensor) else None

@custom_op(
    "fla_kda::gate",
    mutates_args=(),
    schema="(Tensor g, Tensor A_log, int head_k_dim, Tensor? g_bias, Tensor? b, float beta, float threshold) -> Tensor",
)
def _gate_default(g: Tensor, A_log: Tensor, head_k_dim: int,
                  g_bias: Optional[Tensor], b: Optional[Tensor],
                  beta: float, threshold: float) -> Tensor:
    y = torch.sigmoid(A_log) * g
    if g_bias is not None:
        y = y + g_bias
    if b is not None:
        y = y + b
    return y

@register_kernel("fla_kda::gate", "cuda")
def _gate_cuda(g, A_log, head_k_dim, g_bias, b, beta, threshold):
    if _kda_gate_cuda is None:
        return _gate_default(g, A_log, head_k_dim, g_bias, b, beta, threshold)
    return _kda_gate_cuda(g, A_log, head_k_dim, g_bias, b, beta, threshold)

@register_fake("fla_kda::gate")
def _gate_fake(g, A_log, head_k_dim, g_bias, b, beta, threshold):
    return _meta_like(g)

@custom_op(
    "fla_kda::gate_bwd",
    mutates_args=(),
    schema="(Tensor grad_out, Tensor g, Tensor A_log, int head_k_dim, Tensor? g_bias, Tensor? b, float beta, float threshold) -> (Tensor, Tensor, Tensor?, Tensor?)",
)
def _gate_bwd_default(grad_out, g, A_log, head_k_dim, g_bias, b, beta, threshold):
    sig = torch.sigmoid(A_log)
    dg = grad_out * sig
    dA = grad_out * g * sig * (1 - sig)
    dgb = grad_out if g_bias is not None else None
    dbb = grad_out if b is not None else None
    return dg, dA, dgb, dbb

@register_kernel("fla_kda::gate_bwd", "cuda")
def _gate_bwd_cuda(grad_out, g, A_log, head_k_dim, g_bias, b, beta, threshold):
    if _kda_gate_bwd_cuda is None:
        return _gate_bwd_default(grad_out, g, A_log, head_k_dim, g_bias, b, beta, threshold)
    return _kda_gate_bwd_cuda(grad_out, g, A_log, head_k_dim, g_bias, b, beta, threshold)

@register_fake("fla_kda::gate_bwd")
def _gate_bwd_fake(grad_out, g, A_log, head_k_dim, g_bias, b, beta, threshold):
    return _meta_like(g), _meta_like(A_log), _meta_opt_like(g_bias, g), _meta_opt_like(b, g)

def _gate_setup(ctx, inputs, output):
    g, A_log, head_k_dim, g_bias, b, beta, threshold = inputs
    ctx.head_k_dim = int(head_k_dim)
    ctx.beta = float(beta)
    ctx.threshold = float(threshold)
    ctx.has_gb = isinstance(g_bias, Tensor)
    ctx.has_b = isinstance(b, Tensor)
    # Don't create placeholder tensors - save None for missing optional tensors
    ctx.save_for_backward(
        g,
        A_log,
        g_bias if ctx.has_gb else None,
        b if ctx.has_b else None
    )

def _gate_backward(ctx, grad_out: Tensor):
    saved = ctx.saved_tensors
    g, A_log, g_bias, b = saved[0], saved[1], saved[2], saved[3]
    dg, dA, dgb, dbb = torch.ops.fla_kda.gate_bwd(
        grad_out, g, A_log, ctx.head_k_dim,
        (g_bias if ctx.has_gb else None),
        (b if ctx.has_b else None),
        ctx.beta, ctx.threshold
    )
    return dg, dA, None, (dgb if ctx.has_gb else None), (dbb if ctx.has_b else None), None, None

register_autograd("fla_kda::gate", _gate_backward, setup_context=_gate_setup)

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


@custom_op(
    "fla_kda::rmsnorm_gated",
    mutates_args=(),
    schema="(Tensor x, Tensor w, Tensor? b, float eps) -> Tensor",
)
def _rmsnorm_gated_default(x: Tensor, w: Tensor, b: Optional[Tensor], eps: float) -> Tensor:
    xfp = x.to(torch.float32)
    var = xfp.pow(2).mean(-1, keepdim=True)
    y = xfp * torch.rsqrt(var + eps)
    y = (y.to(x.dtype) * w)
    if b is not None:
        y = y + b
    return y

@register_kernel("fla_kda::rmsnorm_gated", "cuda")
def _rmsnorm_gated_cuda(x, w, b, eps):
    if _rmsnorm_gated_cuda is None:
        return _rmsnorm_gated_default(x, w, b, eps)
    return _rmsnorm_gated_cuda(x, w, b, eps)

@register_fake("fla_kda::rmsnorm_gated")
def _rmsnorm_gated_fake(x, w, b, eps):
    return _meta_like(x)

@custom_op(
    "fla_kda::rmsnorm_gated_bwd",
    mutates_args=(),
    schema="(Tensor grad_out, Tensor x, Tensor w, Tensor? b, float eps) -> (Tensor, Tensor, Tensor?)",
)
def _rmsnorm_gated_bwd_default(grad_out, x, w, b, eps):
    xfp = x.to(torch.float32).requires_grad_(True)
    w_ = w.detach().requires_grad_(True)
    b_ = b.detach().requires_grad_(True) if isinstance(b, Tensor) else None
    var = xfp.pow(2).mean(-1, keepdim=True)
    y = xfp * torch.rsqrt(var + eps)
    y = (y.to(x.dtype) * w_)
    if b_ is not None:
        y = y + b_
    gx, gw, gb = torch.autograd.grad(y, (xfp, w_, b_), grad_out, allow_unused=True)
    return (gx.to(x.dtype) if gx is not None else torch.zeros_like(x),
            gw if gw is not None else torch.zeros_like(w),
            (gb if (b is not None and gb is not None) else (torch.zeros_like(b) if isinstance(b, Tensor) else None)))

@register_kernel("fla_kda::rmsnorm_gated_bwd", "cuda")
def _rmsnorm_gated_bwd_cuda(grad_out, x, w, b, eps):
    if _rmsnorm_gated_bwd_cuda is None:
        return _rmsnorm_gated_bwd_default(grad_out, x, w, b, eps)
    return _rmsnorm_gated_bwd_cuda(grad_out, x, w, b, eps)

@register_fake("fla_kda::rmsnorm_gated_bwd")
def _rmsnorm_gated_bwd_fake(grad_out, x, w, b, eps):
    return _meta_like(x), _meta_like(w), _meta_opt_like(b, x)

def _rms_setup(ctx, inputs, output):
    x, w, b, eps = inputs
    ctx.has_bias = isinstance(b, Tensor)
    ctx.eps = float(eps)
    ctx.save_for_backward(x, w, b if ctx.has_bias else None)

def _rms_backward(ctx, grad_out: Tensor):
    saved = ctx.saved_tensors
    x, w, b = saved[0], saved[1], saved[2]
    gx, gw, gb = torch.ops.fla_kda.rmsnorm_gated_bwd(
        grad_out, x, w, (b if ctx.has_bias else None), ctx.eps
    )
    return gx, gw, (gb if ctx.has_bias else None), None

register_autograd("fla_kda::rmsnorm_gated", _rms_backward, setup_context=_rms_setup)


def kda_gate(g, A_log, head_k_dim, g_bias, b, beta, threshold):
    return torch.ops.fla_kda.gate(g, A_log, head_k_dim, g_bias, b, beta, threshold)

def kda_chunk(q, k, v, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel, cu_seqlens):
    return torch.ops.fla_kda.chunk(q, k, v, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel, cu_seqlens)

def rmsnorm_gated(x, w, b, eps):
    return torch.ops.fla_kda.rmsnorm_gated(x, w, b, eps)
