import torch.nn as nn
from models.daisy.attention import CausalSelfAttention, is_flex_available
from models.daisy.mlp import MLP
from models.daisy.functional import norm
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask

import math


def pick_attention_layers(total_layers, d_model=None, num_heads=None):
    """
    Sparse Attention Layer Selection

    For non-degenerate cases:
    - L: total number of layers (int >= 1)
    - d_model: model width (optional)
    - n_heads: number of attention heads (optional; constant across attention layers)
    - d_head: per-head width; if d_model and n_heads are provided, d_head = d_model / n_heads; otherwise d_head = 64

    1) Choose stride s (maximum gap between attention layers) from d_head:
       s = clip(round(8 * sqrt(d_head / 64)), 4, 12)
       Interpretation: if d_head = 64 then s ≈ 8. Wider heads allow slightly larger s,
       but s is always clamped to [4, 12].

    2) Target count K of attention layers:
       K = min(L, max(ceil(L / s), 2 + ceil(log2(L))))
       This ensures at least logarithmically many attention layers and bounds the
       maximum gap between attention layers.

    3) Index placement: pick K indices uniformly on [0, L-1] (inclusive), then deduplicate and sort:
       for i in {0, 1, ..., K-1}:
           idx_i = round(i * (L - 1) / (K - 1))
       By construction, idx_0 = 0 and idx_{K-1} = L - 1.
    """
    if total_layers <= 0: return []
    if total_layers == 1: return [0]
    if total_layers == 2: return [0, 1]
    if total_layers == 3: return [0, 2]
    if total_layers == 4: return [0, 1, 3]
    if total_layers == 5: return [0, 2, 4]
    if total_layers == 6: return [0, 1, 3, 5]
    d_head = (d_model // num_heads) if (d_model and num_heads) else 64
    s = max(4, min(12, round(8 * (d_head / 64) ** 0.5)))
    K = min(total_layers, max(math.ceil(total_layers / s), math.ceil(2 + math.log2(total_layers))))
    idx = [int(round(i * (total_layers - 1) / (K - 1))) for i in range(K)]
    return sorted(set(idx))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, layer_idx: int, head_dim: int, total_layers: int, sparse_attention: bool = False):
        super().__init__()

        if sparse_attention:
            attn_layers = pick_attention_layers(total_layers=total_layers, d_model=dim, num_heads=num_heads)
        else:
            attn_layers = [i for i in range(total_layers)]
        self.attn: CausalSelfAttention = CausalSelfAttention(dim, num_heads, max_seq_len, head_dim,
                                                             use_flex_attn=is_flex_available()) if layer_idx in attn_layers else None
        self.mlp = MLP(dim)

    def reset_history(self):
        if self.attn is not None:
            self.attn.reset_history()

    def forward(self, x: Tensor, ve: Tensor, x0: Tensor, lambdas: Tensor, sa_lambdas: Tensor,
                block_mask: BlockMask = None, attn_mask: Tensor = None, ):
        x = lambdas[0] * x + lambdas[1] * x0
        if self.attn is not None:
            x = x.to(self.attn.qkvo_w.dtype)
            if x.device.type == "cuda":
                x = x + self.attn(x, ve, sa_lambdas, block_mask=block_mask)
            else:
                x = x + self.attn(x, ve, sa_lambdas, attn_mask=attn_mask)
        x = x + self.mlp(norm(x))
        return x

    def step(self, x, ve, x0, k_ctx, v_ctx, pos, lambdas, sa_lambdas, window):
        x = lambdas[0] * x + lambdas[1] * x0
        if self.attn is not None:
            y_att, k_new, v_new = self.attn.step(x, k_ctx, v_ctx, pos, ve, sa_lambdas, window=window)
            x = x + y_att
        else:
            k_new = v_new = None
        x = x + self.mlp(norm(x))
        return x, k_new, v_new

    def prefill(self, x, ve: Tensor | None, x0, lambdas, sa_lambdas, attn_mask, debug=False):
        x = lambdas[0] * x + lambdas[1] * x0
        if self.attn is not None:
            y, k, v = self.attn.prefill(x, ve, sa_lambdas, attn_mask, debug=debug)
            x = x + y
        else:
            k = v = None
        x = x + self.mlp(norm(x))
        return x, k, v
