from typing import Optional

import torch
from torch import nn, Tensor
from einops import rearrange
from fla.ops.kda import chunk_kda, fused_recurrent_kda
from fla.ops.kda.gate import fused_kda_gate
from fla.modules import FusedRMSNormGated, ShortConvolution
#noinspection PyBroadException
try:
    from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
except Exception:
    def get_unpad_data(mask):
        torch._assert(mask is None or bool(mask.dtype == torch.long or mask.dtype == torch.int or mask.dtype == torch.bool), "mask dtype unsupported")
        torch._assert(mask is None or bool(mask.min().item() >= 1), "varlen requires flash-linear-attention install")
        B, S = mask.shape if mask is not None else (0, 0)
        idx = torch.arange(B*S, device=mask.device) if mask is not None else None
        return idx, None, S
    # noinspection PyUnusedLocal
    def index_first_axis(x, idx):
        return x
    # noinspection PyUnusedLocal
    def pad_input(x, indices, batch_size, q_len):
        return x.squeeze(0)

#noinspection PyBroadException
try:
    from fla.models.utils import Cache
except Exception:
    Cache = None

class KimiLinearSelfAttention(nn.Module):
    # noinspection PyUnusedLocal
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, head_dim: int, expand_v: float = 1.0, mode: str = "chunk", use_short_conv: bool = True, allow_neg_eigval: bool = False, conv_size: int = 4, conv_bias: bool = False, layer_idx: int = 0):
        super().__init__()
        self.mode = mode
        self.allow_neg_eigval = allow_neg_eigval
        self.hidden_size = dim
        self.num_heads = num_heads
        self.num_v_heads = num_heads
        self.head_dim = head_dim
        self.head_k_dim = head_dim
        self.head_v_dim = int(head_dim * expand_v)
        self.key_dim = int(self.num_heads * self.head_k_dim)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.layer_idx = layer_idx
        self.q_proj = nn.Linear(dim, self.key_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.key_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.value_dim, bias=False)
        if use_short_conv:
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation="silu", bias=conv_bias)
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation="silu", bias=conv_bias)
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation="silu", bias=conv_bias)
        self.f_proj = nn.Sequential(nn.Linear(dim, self.head_v_dim, bias=False), nn.Linear(self.head_v_dim, self.key_dim, bias=False))
        self.b_proj = nn.Linear(dim, self.num_heads, bias=False)
        self.A_log = nn.Parameter(
            torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16)).view(1, 1, -1, 1)
        )
        self.A_log._no_weight_decay = True
        self.dt_bias = nn.Parameter(torch.zeros(self.key_dim, dtype=torch.float32))
        self.dt_bias._no_weight_decay = True
        self.g_proj = nn.Sequential(nn.Linear(dim, self.head_v_dim, bias=False), nn.Linear(self.head_v_dim, self.value_dim, bias=True))
        self.o_norm = FusedRMSNormGated(self.head_v_dim, activation="sigmoid", eps=1e-5)
        self.o_proj = nn.Linear(self.value_dim, dim, bias=False)
        self._recurrent_state = None
        self._conv_state = None
        self._seen = 0
        self._cache = Cache() if Cache is not None else None

    def reset_history(self):
        self._recurrent_state = None
        self._conv_state = None
        self._seen = 0
        if self._cache is not None:
            self._cache = Cache()

    def _mix_v_with_ve(self, v, ve, sa_lambdas):
        if ve is None:
            return v
        if ve.ndim == 3 and ve.size(-1) == self.hidden_size:
            ve_v = self.v_proj(ve)
            if self.use_short_conv:
                ve_v, _ = self.v_conv1d(x=ve_v, cache=None, output_final_state=False, cu_seqlens=None)
            ve_v = rearrange(ve_v, "... (h d) -> ... h d", d=self.head_v_dim)
        elif ve.ndim == 4 and ve.size(-2) == self.num_v_heads and ve.size(-1) in (self.head_dim, self.head_v_dim):
            if ve.size(-1) == self.head_dim:
                torch._assert(self.head_v_dim == self.head_dim, "expand_v must be 1.0 to accept ve in head_dim")
            ve_v = ve
        else:
            raise RuntimeError("ve shape mismatch")
        lam0 = sa_lambdas[0].to(v.dtype)
        lam1 = sa_lambdas[1].to(v.dtype)
        return lam0 * v + lam1 * ve_v.to(v.dtype)

    def _forward_core(self, x: Tensor, ve: Optional[Tensor], sa_lambdas: Optional[Tensor], attn_mask: Optional[Tensor], use_cache: bool):
        b, s, _ = x.shape
        mode = "fused_recurrent" if s <= 64 and not self.training else self.mode
        last_state = None
        if self._cache is not None and use_cache and len(self._cache) > self.layer_idx:
            last_state = self._cache[self.layer_idx]
        elif self._recurrent_state is not None and use_cache:
            last_state = {"recurrent_state": self._recurrent_state, "conv_state": self._conv_state}
        indices, cu_seqlens, _ = get_unpad_data(attn_mask[:, -s:]) if attn_mask is not None else (None, None, None)
        if indices is not None:
            x = index_first_axis(rearrange(x, "b t d -> (b t) d"), indices).unsqueeze(0)
        if self.use_short_conv:
            cs_q, cs_k, cs_v = (last_state.get("conv_state") if isinstance(last_state, dict) else (None, None, None)) if last_state is not None else (None, None, None)
            q, cs_q = self.q_conv1d(x=self.q_proj(x), cache=cs_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            k, cs_k = self.k_conv1d(x=self.k_proj(x), cache=cs_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            v, cs_v = self.v_conv1d(x=self.v_proj(x), cache=cs_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            conv_state = (cs_q, cs_k, cs_v)
        else:
            q = torch.nn.functional.silu(self.q_proj(x))
            k = torch.nn.functional.silu(self.k_proj(x))
            v = torch.nn.functional.silu(self.v_proj(x))
            conv_state = None
        g = self.f_proj(x)
        g = fused_kda_gate(g, self.A_log, self.head_k_dim, g_bias=self.dt_bias)
        beta = self.b_proj(x).float().sigmoid()
        q, k = (rearrange(t, "... (h d) -> ... h d", d=self.head_k_dim) for t in (q, k))
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)
        if sa_lambdas is not None and ve is not None:
            v = self._mix_v_with_ve(v, ve, sa_lambdas)
        if self.allow_neg_eigval:
            beta = beta * 2.0
        rec = last_state["recurrent_state"] if isinstance(last_state, dict) else None
        if mode == "chunk":
            o, rec = chunk_kda(q=q, k=k, v=v, g=g, beta=beta, initial_state=rec, output_final_state=use_cache, use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens)
        else:
            o, rec = fused_recurrent_kda(q=q, k=k, v=v, g=g, beta=beta, initial_state=rec, output_final_state=use_cache, use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens)
        if use_cache:
            self._recurrent_state = rec
            self._conv_state = conv_state
            self._seen += s
            if self._cache is not None:
                self._cache.update(recurrent_state=rec, conv_state=conv_state, layer_idx=self.layer_idx, offset=s)
        o = self.o_norm(o, rearrange(self.g_proj(x), "... (h d) -> ... h d", d=self.head_v_dim))
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)
        if indices is not None:
            o = pad_input(o.squeeze(0), indices, b, s)
        return o

    def forward(self, x: Tensor, ve: Tensor, sa_lambdas: Tensor, attn_mask: Tensor):
        return self._forward_core(x, ve, sa_lambdas, attn_mask, use_cache=False)

    # noinspection PyUnusedLocal
    def prefill(self, x: Tensor, ve: Tensor, sa_lambdas: Tensor, attn_mask: Tensor, debug: bool = False):
        y = self._forward_core(x, ve, sa_lambdas, attn_mask, use_cache=True)
        return y, None, None

    # noinspection PyUnusedLocal
    def step(self, x: Tensor, k_ctx: Optional[Tensor], v_ctx: Optional[Tensor], pos: int, ve: Tensor, sa_lambdas: Tensor, window: int | None):
        y = self._forward_core(x, ve, sa_lambdas, attn_mask=None, use_cache=True)
        return y, None, None
