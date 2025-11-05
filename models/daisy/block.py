import torch.nn as nn
from models.daisy.attention import CausalSelfAttention, is_flex_available
from models.daisy.mlp import MLP
from models.daisy.functional import norm
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, layer_idx: int, head_dim: int, total_layers: int):
        super().__init__()

        attn_layers = [i for i in range(total_layers)] #TODO configurable
        self.attn: CausalSelfAttention  = CausalSelfAttention(dim, num_heads, max_seq_len, head_dim, use_flex_attn=is_flex_available() ) if layer_idx in attn_layers else None
        self.mlp = MLP(dim)

    def reset_history(self):
        if self.attn is not None:
            self.attn.reset_history()

    def forward(self, x: Tensor, ve: Tensor | None, x0: Tensor, lambdas: Tensor, sa_lambdas: Tensor, block_mask: BlockMask = None, attn_mask: Tensor | None = None,):
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
