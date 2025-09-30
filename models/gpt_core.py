import torch
from torch import nn, Tensor
import torch.nn.functional as F
from models.block import Block
from models.functional import norm
from torch.nn.attention.flex_attention import BlockMask

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

class GPTCore(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, max_seq_len: int, head_dim=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, max_seq_len, i, head_dim) for i in range(num_layers)])
        assert num_layers % 2 == 0
        self.scalars = nn.Parameter(torch.cat([
            torch.ones(num_layers),
            *[torch.tensor([1.0, 0.0]) for _ in range(num_layers)],
            *[torch.tensor([0.5, 0.5]) for _ in range(num_layers)],
        ]))

    def create_blockmasks(self, input_seq: Tensor, sliding_window_num_blocks: Tensor):
        #n = len(input_seq)
        BLOCK_SIZE = 128 #if n % 128 == 0 else next(bs for bs in (64, 32, 16, 8, 4, 2, 1) if n % bs == 0)
        assert(len(input_seq) % BLOCK_SIZE == 0)
        device = input_seq.device
        docs = (input_seq == 50256).cumsum(0)

        def document_causal(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            return causal_mask & document_mask

        def dense_to_ordered(dense_blockmask: Tensor):
            num_blocks = dense_blockmask.sum(dim=-1, dtype=torch.int32)
            indices = dense_blockmask.argsort(dim=-1, descending=False, stable=True).flip(-1).to(torch.int32)
            return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

        assert len(input_seq) % BLOCK_SIZE == 0
        NUM_BLOCKS = len(input_seq) // BLOCK_SIZE
        block_idx = torch.arange(NUM_BLOCKS, dtype=torch.int32, device=device)
        causal_blockmask_any = block_idx[:, None] >= block_idx
        causal_blockmask_all = block_idx[:, None] > block_idx
        docs_low = docs.view(-1, BLOCK_SIZE)[:, 0].contiguous()
        docs_high = docs.view(-1, BLOCK_SIZE)[:, -1].contiguous()
        document_blockmask_any = (docs_low[:, None] <= docs_high) & (docs_high[:, None] >= docs_low)
        document_blockmask_all = (docs_low[:, None] == docs_high) & (docs_high[:, None] == docs_low)
        blockmask_any = causal_blockmask_any & document_blockmask_any
        blockmask_all = causal_blockmask_all & document_blockmask_all
        partial_kv_num_blocks, partial_kv_indices = dense_to_ordered(blockmask_any & ~blockmask_all)
        full_kv_num_blocks, full_kv_indices = dense_to_ordered(blockmask_all)

        def build_bm(window_size_blocks: Tensor) -> BlockMask:
            return BlockMask.from_kv_blocks(
                torch.clamp_max(partial_kv_num_blocks, torch.clamp_min(window_size_blocks - full_kv_num_blocks, 1)),
                partial_kv_indices,
                torch.clamp_max(full_kv_num_blocks, window_size_blocks - 1),
                full_kv_indices,
                BLOCK_SIZE=BLOCK_SIZE,
                mask_mod=document_causal,
            )

        # Long-short SWA block masks by @leloykun & @YouJiacheng, adapated from suggestion by @Grad62304977, following Gemma 2 paper
        return build_bm(sliding_window_num_blocks), build_bm(sliding_window_num_blocks // 2)

    def forward(self, input_seq: Tensor, sliding_window_num_blocks: Tensor, target_seq: Tensor = None):
        assert input_seq.ndim == 1
        L = len(self.blocks)

        ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        ve = [ve[0], ve[1], ve[2]] + [None] * (L - 6) + [ve[0], ve[1], ve[2]]

        long_bm, short_bm = self.create_blockmasks(input_seq, sliding_window_num_blocks)
        # TODO: hoursglass pattern
        cycle = [long_bm] + [short_bm] * 3
        block_masks = (cycle * ((L + 3) // 4))[:L-1] + [long_bm]

        x = x0 = norm(self.embed(input_seq)[None])

        stride = max(1, L // 4)
        skip_map = {}
        for m in range(3):
            i = L - 3 + m
            j = i - stride * (m + 1)
            if 0 <= j < i < L:
                skip_map[i] = j

        skip_weights = self.scalars[:L]
        lambdas = self.scalars[1 * L:3 * L].view(-1, 2)
        sa_lambdas = self.scalars[3 * L:5 * L].view(-1, 2)

        skip_connections = []
        for i in range(L):
            if i in skip_map:
                x = x + skip_weights[skip_map[i]] * skip_connections[skip_map[i]]
            # if torch.is_grad_enabled():
            #     x = torch.utils.checkpoint.checkpoint(
            #         lambda _x, lam, sa_lam: self.blocks[i](_x, ve[i], x0, block_masks[i], lam, sa_lam),
            #         x, lambdas[i], sa_lambdas[i], use_reentrant=False
            #     )
            # else:
                x = self.blocks[i](x, ve[i], x0, block_masks[i], lambdas[i], sa_lambdas[i])
            skip_connections.append(x)

        x = norm(x)
        if self.training:
            logits: Tensor = F.linear(x.flatten(end_dim=1), self.embed.weight.bfloat16()).float()
            loss = F.cross_entropy(15 * logits * torch.rsqrt(logits.square() + 225), target_seq)
            return loss

        loss = 0
        for i in range(4):
            logits: Tensor = F.linear(x.flatten(end_dim=1).chunk(4)[i], self.embed.weight.bfloat16()).float()
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

        stride = max(1, L // 4)
        skip_map = {}
        for m in range(3):
            i = L - 3 + m
            j = i - stride * (m + 1)
            if 0 <= j < i < L:
                skip_map[i] = j

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
        logits = F.linear(x.flatten(end_dim=1), self.embed.weight.bfloat16()).float()
        return logits, k_new_list, v_new_list
