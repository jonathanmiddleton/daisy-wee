import torch.nn as nn
from models.attention import CausalSelfAttention
from models.mlp import MLP
from models.functional import norm
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, layer_idx: int):
        super().__init__()
        # skip attention for layers mod 8
        self.attn = CausalSelfAttention(dim, num_heads, max_seq_len) if (layer_idx + 1) % 8 != 0 else None
        self.mlp = MLP(dim)

    def forward(self, x: Tensor, ve: Tensor | None, x0: Tensor, block_mask: BlockMask, lambdas: Tensor, sa_lambdas: Tensor):
        x = lambdas[0] * x + lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(x, ve, block_mask, sa_lambdas)
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
