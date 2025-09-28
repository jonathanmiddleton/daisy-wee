import torch
from torch import  nn, Tensor
import torch.nn.functional as F
from models.block import Block
from models.functional import norm, init_linear


def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

class GPTCore(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, max_seq_len: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim)
        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual implementation following https://arxiv.org/abs/2410.17897
        # value embedding code simplification inspired by @ragulpr https://github.com/KellerJordan/modded-nanogpt/pull/78
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, max_seq_len, i) for i in range(num_layers)])
        # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency.
        # suggested to me by @Grad62304977. this originates from Karpathy's experiments.
        self.lm_head_w = nn.Parameter(torch.zeros(next_multiple_of_n(vocab_size, n=128), model_dim))
        # Add learnable skip connection weights for decoder layers
        assert num_layers % 2 == 0
        self.scalars = nn.Parameter(torch.cat([
            torch.ones(num_layers), # skip_weights
            *[torch.tensor([1.0, 0.0]) for _ in range(num_layers)], # block lambdas
            *[torch.tensor([0.5, 0.5]) for _ in range(num_layers)], # SA lambdas
        ]))

    def forward(self, input_seq: Tensor, sliding_window_num_blocks: Tensor, target_seq: Tensor = None):
        assert input_seq.ndim == 1

        ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        # 012 ... 012 structure on token value embeddings by @YouJiacheng, improved on @leloykun's U-net structure
        ve = [ve[0], ve[1], ve[2]] + [None] * (len(self.blocks) - 6) + [ve[0], ve[1], ve[2]]
        assert len(ve) == len(self.blocks)

        long_bm, short_bm = self.create_blockmasks(input_seq, sliding_window_num_blocks)
        block_masks = [long_bm, short_bm, short_bm, short_bm, long_bm, short_bm, short_bm, short_bm, short_bm, short_bm, short_bm, long_bm, short_bm, short_bm, short_bm, long_bm]
        assert len(block_masks) == len(self.blocks)

        x = x0 = norm(self.embed(input_seq)[None]) # use of norm here by @Grad62304977

        skip_connections = []
        skip_map = {
            9: 6,
            10: 4,
            11: 2,
        }
        skip_weights = self.scalars[:len(self.blocks)]
        lambdas = self.scalars[1 * len(self.blocks): 3 * len(self.blocks)].view(-1, 2)
        sa_lambdas = self.scalars[3 * len(self.blocks): 5 * len(self.blocks)].view(-1, 2)
        for i in range(len(self.blocks)):
            if i in skip_map:
                x = x + skip_weights[skip_map[i]] * skip_connections[skip_map[i]]
            x = self.blocks[i](x, ve[i], x0, block_masks[i], lambdas[i], sa_lambdas[i])
            skip_connections.append(x)

        x = norm(x)
        if self.training:
            logits: Tensor = F.linear(x.flatten(end_dim=1), self.lm_head_w.bfloat16()).float()
            loss = F.cross_entropy(15 * logits * torch.rsqrt(logits.square() + 225), target_seq)
            return loss

        loss = 0
        for i in range(4):
            logits: Tensor = F.linear(x.flatten(end_dim=1).chunk(4)[i], self.lm_head_w.bfloat16()).float()
            loss += F.cross_entropy(15 * logits * torch.rsqrt(logits.square() + 225), target_seq.chunk(4)[i]) / 4
        return loss


    @torch.no_grad()
    def step(self, token_id: Tensor, k_ctxs, v_ctxs, pos: int, window: int):
        assert token_id.ndim == 0
        B = 1
        x0 = norm(self.embed(token_id)[None, None, :])
        h = None
        d = None
        for b in self.blocks:
            if getattr(b, "attn", None) is not None:
                h = b.attn.num_heads
                d = b.attn.head_dim
                break
        ve0 = self.value_embeds[0](token_id).view(B, 1, h, d)
        ve1 = self.value_embeds[1](token_id).view(B, 1, h, d)
        ve2 = self.value_embeds[2](token_id).view(B, 1, h, d)
        L = len(self.blocks)
        ve = [ve0, ve1, ve2] + [None] * (L - 6) + [ve0, ve1, ve2]
        skip_map = {9: 6, 10: 4, 11: 2}
        scalars = self.scalars
        skip_weights = scalars[:L]
        lambdas = scalars[1 * L:3 * L].view(-1, 2)
        sa_lambdas = scalars[3 * L:5 * L].view(-1, 2)
        x = x0
        k_new_list = []
        v_new_list = []
        skip_connections = []
        for i in range(L):
            if i in skip_map:
                x = x + skip_weights[skip_map[i]] * skip_connections[skip_map[i]]
            y, k_new, v_new = self.blocks[i].step(
                x, ve[i], x0, k_ctxs[i], v_ctxs[i], pos, lambdas[i], sa_lambdas[i], window
            )
            x = y
            skip_connections.append(x)
            k_new_list.append(k_new)
            v_new_list.append(v_new)
        x = norm(x)
        logits = F.linear(x.flatten(end_dim=1), self.lm_head_w.bfloat16()).float()
        return logits, k_new_list, v_new_list
