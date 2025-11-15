from typing import Optional, Tuple
import torch
from torch import Tensor
from torch.library import custom_op

from fla.ops.kda import chunk_kda as _chunk_kda
from fla.ops.kda.gate import fused_kda_gate as _fused_kda_gate
from fla.modules.fused_norm_gate import rms_norm_gated as _rms_norm_gated


@custom_op("fla_kda::gate", mutates_args=())
def kda_gate(g: Tensor, A_log: Tensor, head_k_dim: int,
             g_bias: Optional[Tensor] = None,
             b: Optional[Tensor] = None,
             beta: float = 1.0,
             threshold: float = 20.0) -> Tensor:
    out = _fused_kda_gate(g, A_log, head_k_dim, g_bias=g_bias, b=b, beta=beta, threshold=threshold)
    return out[0] if isinstance(out, tuple) else out

@kda_gate.register_fake
def _kda_gate_fake(g, A_log, head_k_dim: int, g_bias=None, b=None, beta: float = 1.0, threshold: float = 20.0):
    return torch.empty_like(g)

def _kda_gate_setup(ctx, inputs, output=None):
    g, A_log, head_k_dim, g_bias, b, beta, threshold = inputs
    gb = g_bias if isinstance(g_bias, Tensor) else torch.empty(0, device=g.device, dtype=g.dtype)
    bb = b if isinstance(b, Tensor) else torch.empty(0, device=g.device, dtype=g.dtype)
    ctx.save_for_backward(g, A_log, gb, bb)
    ctx.has_gb = isinstance(g_bias, Tensor)
    ctx.has_b = isinstance(b, Tensor)
    ctx.head_k_dim = int(head_k_dim)
    ctx.beta = float(beta)
    ctx.threshold = float(threshold)

def _kda_gate_backward(ctx, grad_out: Tensor):
    g, A_log, gb, bb = ctx.saved_tensors
    with torch.enable_grad():
        g_ = g.detach().requires_grad_(g.requires_grad)
        A_ = A_log.detach().requires_grad_(A_log.requires_grad)
        gb_ = gb.detach().requires_grad_(ctx.has_gb) if ctx.has_gb else None
        bb_ = bb.detach().requires_grad_(ctx.has_b) if ctx.has_b else None
        y = kda_gate(g_, A_, ctx.head_k_dim, g_bias=gb_ if ctx.has_gb else None,
                    b=bb_ if ctx.has_b else None, beta=ctx.beta, threshold=ctx.threshold)
        vars_ = (g_, A_) + ((gb_,) if ctx.has_gb else ()) + ((bb_,) if ctx.has_b else ())
        grads = torch.autograd.grad(y, vars_, grad_out, allow_unused=True)
    dg = grads[0]
    dA = grads[1]
    off = 2
    dgb = grads[off] if ctx.has_gb else None
    dbb = grads[off + (1 if ctx.has_gb else 0)] if ctx.has_b else None
    return dg, dA, None, dgb, dbb, None, None

kda_gate.register_autograd(_kda_gate_backward, setup_context=_kda_gate_setup)


@custom_op("fla_kda::chunk", mutates_args=())
def kda_chunk(q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor,
              initial_state: Optional[Tensor],
              output_final_state: bool,
              use_qk_l2norm_in_kernel: bool,
              cu_seqlens: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
    o, rec = _chunk_kda(q=q, k=k, v=v, g=g, beta=beta,
                        initial_state=initial_state,
                        output_final_state=output_final_state,
                        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                        cu_seqlens=cu_seqlens)
    return o, rec

@kda_chunk.register_fake
def _kda_chunk_fake(q, k, v, g, beta, initial_state, output_final_state: bool, use_qk_l2norm_in_kernel: bool, cu_seqlens):
    o = torch.empty_like(v)
    rec = initial_state if (output_final_state and isinstance(initial_state, Tensor)) else torch.empty(0, dtype=v.dtype, device=v.device)
    return o, rec

def _kda_chunk_setup(ctx, inputs, output=None):
    q, k, v, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel, cu_seqlens = inputs
    init = initial_state if isinstance(initial_state, Tensor) else torch.empty(0, device=q.device, dtype=q.dtype)
    cu = cu_seqlens if isinstance(cu_seqlens, Tensor) else torch.empty(0, device=q.device, dtype=torch.int32)
    ctx.save_for_backward(q, k, v, g, beta, init, cu)
    ctx.has_init = init.numel() > 0
    ctx.has_cu = cu.numel() > 0
    ctx.output_final_state = bool(output_final_state)
    ctx.use_qk_l2norm_in_kernel = bool(use_qk_l2norm_in_kernel)

def _kda_chunk_backward(ctx, d_o: Tensor, d_rec: Tensor):
    q, k, v, g, beta, init, cu = ctx.saved_tensors
    with torch.enable_grad():
        q_ = q.detach().requires_grad_(q.requires_grad)
        k_ = k.detach().requires_grad_(k.requires_grad)
        v_ = v.detach().requires_grad_(v.requires_grad)
        g_ = g.detach().requires_grad_(g.requires_grad)
        beta_ = beta.detach().requires_grad_(beta.requires_grad)
        init_ = init.detach().requires_grad_(True) if ctx.has_init else None
        cu_ = cu if ctx.has_cu else None
        y, rec2 = kda_chunk(q_, k_, v_, g_, beta_,
                           init_,
                           ctx.output_final_state,
                           ctx.use_qk_l2norm_in_kernel,
                           cu_)
        if ctx.output_final_state:
            grads = torch.autograd.grad((y, rec2), (q_, k_, v_, g_, beta_, init_), (d_o, d_rec), allow_unused=True)
        else:
            grads = torch.autograd.grad(y, (q_, k_, v_, g_, beta_, init_), d_o, allow_unused=True)
    dq, dk, dv, dg, dbeta, dinit = grads
    return dq, dk, dv, dg, dbeta, dinit, None, None, None

kda_chunk.register_autograd(_kda_chunk_backward, setup_context=_kda_chunk_setup)



@custom_op("fla_kda::rmsnorm_gated", mutates_args=())
def rmsnorm_gated(x: Tensor, z: Tensor,
                  weight: Tensor,
                  bias: Optional[Tensor],
                  eps: float,
                  group_size: Optional[int],
                  norm_before_gate: bool) -> Tensor:
    return _rms_norm_gated(x=x, g=z, weight=weight, bias=bias,
                           eps=float(eps))

@rmsnorm_gated.register_fake
def _rms_fake(x, z, weight, bias, eps: float, group_size: Optional[int], norm_before_gate: bool):
    return torch.empty_like(x)

def _rms_setup(ctx, inputs, output=None):
    x, z, weight, bias, eps, group_size, norm_before_gate = inputs
    b = bias if isinstance(bias, Tensor) else torch.empty(0, device=x.device, dtype=weight.dtype)
    ctx.save_for_backward(x, z, weight, b)
    ctx.has_bias = isinstance(bias, Tensor)
    ctx.eps = float(eps)
    ctx.group_size = group_size
    ctx.norm_before_gate = bool(norm_before_gate)

def _rms_backward(ctx, d_out: Tensor):
    x, z, weight, b = ctx.saved_tensors
    with torch.enable_grad():
        x_ = x.detach().requires_grad_(x.requires_grad)
        z_ = z.detach().requires_grad_(z.requires_grad)
        w_ = weight.detach().requires_grad_(weight.requires_grad)
        b_ = b.detach().requires_grad_(True) if ctx.has_bias else None
        y = rmsnorm_gated(x=x_, z=z_, weight=w_, bias=b_,
                         eps=ctx.eps, group_size=ctx.group_size,
                         norm_before_gate=ctx.norm_before_gate)
        vars_ = (x_, z_, w_) + ((b_,) if ctx.has_bias else ())
        grads = torch.autograd.grad(y, vars_, d_out, allow_unused=True)
    dx = grads[0]
    dz = grads[1]
    dw = grads[2]
    db = grads[3] if ctx.has_bias and len(grads) > 3 else None
    return dx, dz, dw, db, None, None, None

rmsnorm_gated.register_autograd(_rms_backward, setup_context=_rms_setup)
