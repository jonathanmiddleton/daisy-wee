from __future__ import annotations
from typing import Optional
import torch
import torch.nn.functional as F
from torch import Tensor

@torch.library.custom_op(
    "daisy_fla_kda::gate",
    schema="(Tensor g, Tensor A_log, int head_k_dim, Tensor? g_bias, Tensor? b, float beta, float threshold) -> (Tensor, Tensor)",
    mutates_args={}
)
def _kda_gate_customop(
    g: Tensor,
    A_log: Tensor,
    head_k_dim: int,
    g_bias: Optional[Tensor] = None,
    b: Optional[Tensor] = None,
    beta: float = 1.0,
    threshold: float = 20.0,
):
    raise NotImplementedError

def _gate_fake(
    g: Tensor,
    A_log: Tensor,
    head_k_dim: int,
    g_bias: Optional[Tensor],
    b: Optional[Tensor],
    beta: float,
    threshold: float,
):
    last = g.shape[-1]
    if last % head_k_dim != 0:
        raise RuntimeError("last dim must be divisible by head_k_dim")
    H = last // head_k_dim
    g_out = torch.empty((*g.shape[:-1], H, head_k_dim), device=g.device, dtype=g.dtype)
    b_out = torch.empty((*g.shape[:-1], H), device=g.device, dtype=g.dtype)
    return g_out, b_out

_kda_gate_customop.register_fake(_gate_fake)

def _split_heads(g: Tensor, head_k_dim: int):
    H = g.shape[-1] // head_k_dim
    return g.view(*g.shape[:-1], H, head_k_dim), H

def _canon_A(A_log: Tensor, H: int, out_ndim: int, device, dtype):
    if A_log.dim() == 1 and A_log.numel() == H:
        return A_log.to(device=device, dtype=dtype).view(*([1] * (out_ndim - 2)), H, 1)
    if A_log.dim() == 4 and list(A_log.shape) == [1, 1, H, 1]:
        a = A_log.to(device=device, dtype=dtype).reshape(H)
        return a.view(*([1] * (out_ndim - 2)), H, 1)
    raise RuntimeError(f"A_log must be shape [H] or [1,1,H,1]; got {tuple(A_log.shape)} with H={H}")

def _canon_bias(g_bias: Optional[Tensor], H: int, D: int, device, dtype):
    if g_bias is None:
        return None
    x = g_bias.to(device=device, dtype=dtype)
    if x.dim() == 1 and x.numel() == H * D:
        return x.view(H, D)
    if x.dim() == 2 and list(x.shape) == [H, D]:
        return x
    if x.dim() == 4 and list(x.shape) == [1, 1, H, D]:
        return x.view(H, D)
    if x.dim() == 1 and x.numel() == H:
        return x.view(H, 1).expand(H, D)
    raise RuntimeError(f"g_bias must be one of [(H*D,), (H,D), (1,1,H,D), (H,)]; got {tuple(x.shape)} with H={H}, D={D}")

def _expand_to_prefix(x: Tensor, prefix_ndim: int):
    return x.view(*([1] * prefix_ndim), *x.shape)

def _gate_impl(
    g: Tensor,
    A_log: Tensor,
    head_k_dim: int,
    g_bias: Optional[Tensor],
    b: Optional[Tensor],
    beta: float,
    threshold: float,
):
    g_hd, H = _split_heads(g, head_k_dim)
    D = head_k_dim
    prefix_ndim = g_hd.ndim - 2
    device = g.device
    dtype = g.dtype

    g32 = g_hd.to(torch.float32)
    bias_hd = _canon_bias(g_bias, H, D, device, torch.float32)
    if bias_hd is not None:
        g32 = g32 + _expand_to_prefix(bias_hd, prefix_ndim)

    sp = F.softplus(g32, beta=float(beta), threshold=float(threshold))

    A_b = _canon_A(A_log, H, g_hd.ndim, device, torch.float32)
    a = -torch.exp(A_b)

    y32 = a * sp
    y = y32.to(dtype)

    if b is None:
        b_sigmoid = torch.zeros((*g.shape[:-1], H), device=device, dtype=dtype)
    else:
        b_sigmoid = torch.sigmoid(b.to(device=device, dtype=dtype))
    return y, b_sigmoid

_kda_gate_customop.register_kernel(("cpu", "cuda"), _gate_impl)

#noinspection PyUnusedLocal
def _gate_setup_ctx(ctx, inputs, output):
    g, A_log, head_k_dim, g_bias, b, beta, threshold = inputs
    ctx.head_k_dim = int(head_k_dim)
    ctx.beta = float(beta)
    ctx.threshold = float(threshold)
    ctx.has_bias = g_bias is not None
    ctx.has_b = b is not None
    tensors = [g, A_log]
    if ctx.has_bias:
        tensors.append(g_bias)
    if ctx.has_b:
        tensors.append(b)
    ctx.save_for_backward(*tensors)

def _gate_backward(ctx, grad_y: torch.Tensor, grad_bsig: torch.Tensor):
    saved = ctx.saved_tensors
    i = 0
    g = saved[i]; i += 1
    A_log = saved[i]; i += 1
    g_bias = saved[i] if ctx.has_bias else None
    if ctx.has_bias:
        i += 1
    b = saved[i] if ctx.has_b else None

    g_hd, H = _split_heads(g, ctx.head_k_dim)
    D = ctx.head_k_dim
    device = g.device
    dtype = g.dtype

    prefix_dims = tuple(range(g_hd.ndim - 2))

    g32 = g_hd.to(torch.float32)
    bias_hd = _canon_bias(g_bias, H, D, device, torch.float32)
    if bias_hd is not None:
        g32 = g32 + bias_hd.view(*([1] * len(prefix_dims)), H, D)

    beta = float(ctx.beta)
    thr = float(ctx.threshold)
    pre = g32
    sp = torch.nn.functional.softplus(pre, beta=beta, threshold=thr)

    A_b = _canon_A(A_log, H, g_hd.ndim, device, torch.float32)
    a = -torch.exp(A_b)

    gy32 = grad_y.to(torch.float32)

    sp_prime = torch.where(beta * pre > thr, torch.ones_like(pre), torch.sigmoid(beta * pre))
    dpre = gy32 * a * sp_prime

    grad_g = dpre.to(dtype).reshape_as(g)

    t = (gy32 * sp) * a
    if A_log.dim() == 1:
        grad_A_log = t.sum(dim=prefix_dims + (-1,)).to(A_log.dtype)
    else:
        gA = t.sum(dim=prefix_dims + (-1,), keepdim=False)  # [H]
        grad_A_log = gA.view(1, 1, H, 1).to(A_log.dtype)

    grad_g_bias = None
    if g_bias is not None:
        rb = dpre.sum(dim=prefix_dims)  # [H, D]
        if g_bias.dim() == 1 and g_bias.numel() == H * D:
            grad_g_bias = rb.reshape(H * D).to(g_bias.dtype)
        elif g_bias.dim() == 2 and list(g_bias.shape) == [H, D]:
            grad_g_bias = rb.to(g_bias.dtype)
        elif g_bias.dim() == 4 and list(g_bias.shape) == [1, 1, H, D]:
            grad_g_bias = rb.view(1, 1, H, D).to(g_bias.dtype)
        elif g_bias.dim() == 1 and g_bias.numel() == H:
            grad_g_bias = rb.sum(dim=-1).to(g_bias.dtype)
        else:
            raise RuntimeError("Unsupported g_bias shape in backward")

    grad_b = None
    if b is not None:
        sb = torch.sigmoid(b)
        grad_b = (grad_bsig * sb * (1 - sb)).to(b.dtype)

    return grad_g, grad_A_log, None, grad_g_bias, grad_b, None, None


_kda_gate_customop.register_autograd(_gate_backward, setup_context=_gate_setup_ctx)

def fused_kda_gate(
    g: Tensor,
    A_log: Tensor,
    head_k_dim: int,
    g_bias: Optional[Tensor] = None,
    b: Optional[Tensor] = None,
    beta: float = 1.0,
    threshold: float = 20.0,
):
    y, b_sig = torch.ops.daisy_fla_kda.gate(g, A_log, head_k_dim, g_bias, b, float(beta), float(threshold))
    return y if b is None else (y, b_sig)