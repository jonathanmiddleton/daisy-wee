import os

import torch
from torch import nn, Tensor
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from torch.nn import functional as F
from models.daisy.functional import norm, init_linear

def _apply_rope(x_BTHD , cos, sin):
    x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat((y1, y2), dim=-1).type_as(x_BTHD)

def _flex_call(q, k, v, block_mask, scale):
    return flex_attention(q, k, v, block_mask=block_mask, scale=scale)

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        # For small head_dim (< 4), do NOT truncate; otherwise keep half-truncation.
        half = dim // 2
        keep = dim // 4  # number of active frequencies when truncating by half
        base = 1024
        if keep == 0:
            # head_dim < 4 -> no truncation; use standard RoPE with 'half' frequencies
            angular_freq = (1 / base) ** torch.linspace(0, 1, steps=half, dtype=torch.float32)
        else:
            active = (1 / base) ** torch.linspace(0, 1, steps=keep, dtype=torch.float32)
            # pad with zeros to keep total = half (dim//2)
            angular_freq = torch.cat([active, active.new_zeros(half - keep)])
        # Register buffers so they move with module.to(device) exactly once
        self.inv_freq = nn.Buffer(angular_freq, persistent=False)
        # Preallocate cos/sin tables up to max_seq_len on construction; slice at runtime
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, self.inv_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)
        # Track capacity for assertions
        self._max_seq_len = int(max_seq_len)

    def _get_cos_sin(self, length: int):
        torch._assert(
            length <= self._max_seq_len,
            f"Rotary buffers too small: requested length={{length}} > preallocated max_seq_len={self._max_seq_len}."
        )
        return self.cos[:length], self.sin[:length]

    def forward(self, x_BTHD: Tensor):
        L = x_BTHD.size(-3)
        cos, sin = self._get_cos_sin(L)
        cos = cos[None, :L, None, :]
        sin = sin[None, :L, None, :]
        return _apply_rope(x_BTHD, cos, sin)

    def step(self, x_BTHD: Tensor, pos: int):
        # Use preallocated tables and select a single position
        assert x_BTHD.size(-3) == 1, "step() requires single position only"
        cos, sin = self._get_cos_sin(pos + 1)
        cos = cos[None, pos, None, :]
        sin = sin[None, pos, None, :]
        return _apply_rope(x_BTHD, cos, sin)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim =  head_dim
        m_dim = num_heads * head_dim
        # merged QKV weights: suggested by many, implemented by @fernbear.bsky.social, and further improved by @YouJiacheng
        # https://x.com/hi_tysam/status/1879699187107033311
        self.qkvo_w = nn.Parameter(init_linear(torch.empty(4, m_dim, dim)).bfloat16())
        if os.getenv("DISABLE_O_ZERO_INIT", "") != "1": # 1 for unittests
            self.qkvo_w.detach()[3].zero_() # zero-out init suggested by @Grad62304977
        self.rotary = Rotary(head_dim, max_seq_len)
        # scale the attention logits by given constant, instead of the default head_dim**-0.5, by @leloykun
        # inspired by learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283
        self.attn_scale = 0.12
        self.last_q = None
        self.last_k = None

    def reset_history(self):
        self.last_q = None
        self.last_k = None

    def forward(self, x: Tensor, ve: Tensor | None,  lambdas: Tensor, block_mask: BlockMask = None, attn_mask: Tensor | None = None):
        assert block_mask is None or attn_mask is None, "block_mask and attn_mask are mutually exclusive"
        B, T = x.size(0), x.size(1)
        x = x.to(self.qkvo_w.dtype)
        qkv = self.qkvo_w[:3].flatten(end_dim=1)
        q, k, v = F.linear(x, qkv).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k, v = norm(q), norm(k), norm(v)
        q, k = self.rotary(q), self.rotary(k)

        target_dtype = q.dtype
        v = v.to(target_dtype)
        if ve is not None:
            ve = ve.to(target_dtype)
            lambdas = lambdas.to(target_dtype)
            v = lambdas[0] * v + lambdas[1] * ve.view_as(v)
        else:
            lambdas = lambdas.to(target_dtype)
            v = lambdas[0] * v

        k_ = k.transpose(1, 2)
        v_ = v.transpose(1, 2)
        q_ = q.transpose(1, 2)

        if block_mask is not None:
             y = _flex_call(q_, k_, v_, block_mask=block_mask, scale=self.attn_scale)
        elif attn_mask is not None:
            y = F.scaled_dot_product_attention(q_, k_, v_, scale=self.attn_scale, attn_mask=attn_mask)
        else:
            y = F.scaled_dot_product_attention(q_, k_, v_, scale=self.attn_scale,  is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        y = F.linear(y, self.qkvo_w[3])
        return y


    def prefill(self, x: torch.Tensor, ve: torch.Tensor | None, lambdas: torch.Tensor, attn_mask: torch.Tensor | None = None, debug: bool = False,):
        B, T, _ = x.shape
        x = x.to(self.qkvo_w.dtype)
        w = self.qkvo_w[:3]
        w = w.flatten(end_dim=1)
        qkv = torch.nn.functional.linear(x, w).view(B, T, 3 * self.num_heads, self.head_dim)
        q, k, v = qkv.chunk(3, dim=-2)
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
        if debug:
            self.last_q = q
            self.last_k = k
        v = norm(v)
        target_dtype = q.dtype
        v = v.to(target_dtype)
        if ve is not None:
            ve = ve.to(target_dtype).view(B, T, self.num_heads, self.head_dim)
            lambdas = lambdas.to(target_dtype)
            v = lambdas[0] * v + lambdas[1] * ve
        else:
            lambdas = lambdas.to(target_dtype)
            v = lambdas[0] * v
        q_ = q.transpose(1, 2)
        k_ = k.transpose(1, 2)
        v_ = v.transpose(1, 2)
        if attn_mask is not None:
            attn_mask = attn_mask.to(target_dtype)
            y = torch.nn.functional.scaled_dot_product_attention(q_, k_, v_, attn_mask=attn_mask, is_causal=False,
                                                                 scale=self.attn_scale)
        else:
            y = torch.nn.functional.scaled_dot_product_attention(q_, k_, v_, is_causal=True, scale=self.attn_scale)
        y = y.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        y = torch.nn.functional.linear(y, self.qkvo_w[3])
        return y, k_, v_


    def step(self, x, k_ctx: Tensor, v_ctx: Tensor, pos: int, ve: Tensor | None, lambdas: Tensor, window: int):
        B, T = x.size(0), x.size(1)
        x = x.to(self.qkvo_w.dtype)
        qkv = self.qkvo_w[:3].flatten(end_dim=1)
        q, k, v = F.linear(x, qkv).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k, v = norm(q), norm(k), norm(v)
        q, k = self.rotary.step(q, pos), self.rotary.step(k, pos)

        target_dtype = q.dtype
        v = v.to(target_dtype)
        if ve is not None:
            ve = ve.to(target_dtype)
            lambdas = lambdas.to(target_dtype)
            v = lambdas[0] * v + lambdas[1] * ve.view_as(v)
        else:
            lambdas = lambdas.to(target_dtype)
            v = lambdas[0] * v

        n = k_ctx.size(1)
        r = n if window is None else min(n, max(window - 1, 0))
        k_ = torch.cat([k_ctx[:, n - r:n], k], 1).transpose(1, 2).to(target_dtype)
        v_ = torch.cat([v_ctx[:, n - r:n], v], 1).transpose(1, 2).to(target_dtype)
        q_ = q.transpose(1, 2)

        # since q holds a single position `is_causal` must be False or indices after 0 will be masked out
        y = F.scaled_dot_product_attention(q_, k_, v_, scale=self.attn_scale, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        y = F.linear(y, self.qkvo_w[3])
        return y, k, v