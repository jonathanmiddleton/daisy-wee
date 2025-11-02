import os
from math import floor, log2

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from models.daisy.block import Block
from models.daisy.functional import norm
from torch.nn.attention.flex_attention import BlockMask

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)


class DaisyCore(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, max_seq_len: int, head_dim, window_block_size: int = 128, eos_token_id: int | None = None, desc: dict | None = None):
        super().__init__()
        if eos_token_id is None:
            raise ValueError("eos_token_id is required.")

        def _get_skip_map(L: int):
            """
            Side‑band residual mappings. Places targets just past the midpoint to avoid bypassing too much computation,
            while spacing sources by `s` partitions the first half into `K+1` chunks, giving progressively longer skips
            that cover diverse timescales.
            Parameters:
                L: int
                    Layer count

            Returns:
                dict[int, int]
                    A dictionary mapping target indices to source indices.
            """
            K = max(1, floor(log2(L)) - 1)
            c = L // 2
            s = max(1, c // (K + 1))
            m = {c + t: c - t * s for t in range(1, K + 1)}
            return {i: j for i, j in m.items() if 0 <= j < i < L}

        self.skip_map = _get_skip_map(num_layers)
        self.eos_token_id = int(eos_token_id)
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, max_seq_len, i, head_dim, num_layers) for i in range(num_layers)])
        if os.getenv("DISABLE_O_ZERO_INIT", "") != "1":
            # != 1 training
            self.lm_head_w = nn.Parameter(torch.zeros(next_multiple_of_n(vocab_size, n=128), model_dim))
        else:
            # == 1 to allow backpropagation for lr_sweep or cases where the LM head is frozen for testing
            self.lm_head_w = nn.Parameter(torch.empty(next_multiple_of_n(vocab_size, n=128), model_dim))
            nn.init.normal_(self.lm_head_w, mean=0.0, std=0.02)
        self.window_block_size = int(window_block_size)
        assert num_layers % 2 == 0
        self.scalars = nn.Parameter(torch.cat([
            torch.ones(num_layers),                                     # skip_weights
            *[torch.tensor([1.0, 0.0]) for _ in range(num_layers)],     # residual mixing
            *[torch.tensor([0.5, 0.5]) for _ in range(num_layers)],     # value embedding mixing
        ]))
        self.desc = desc # non-functional, self-describing metadata

    def reset_history(self):
        for b in self.blocks:
            b.reset_history()

    def create_blockmasks(self, input_seq: Tensor, sliding_window_num_blocks: Tensor):
        BLOCK_SIZE = self.window_block_size
        assert (len(input_seq) % BLOCK_SIZE == 0)
        device = input_seq.device
        docs = (input_seq == self.eos_token_id).cumsum(0)

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

        skip_map = self.skip_map
        skip_weights = self.scalars[:L]
        lambdas = self.scalars[1 * L:3 * L].view(-1, 2)
        sa_lambdas = self.scalars[3 * L:5 * L].view(-1, 2)

        skip_connections = []
        for i in range(L):
            if i in skip_map:
                x = x + skip_weights[skip_map[i]] * skip_connections[skip_map[i]]
            x = self.blocks[i](x, ve[i], x0, block_masks[i], lambdas[i], sa_lambdas[i])
            skip_connections.append(x)

        x = norm(x)
        if self.training:
            logits: Tensor = F.linear(x.flatten(end_dim=1).bfloat16(), self.lm_head_w.bfloat16()).float()
            loss = F.cross_entropy(15 * logits * torch.rsqrt(logits.square() + 225), target_seq)
            return loss

        # eval
        loss = 0
        for i in range(4):
            logits: Tensor = F.linear(x.flatten(end_dim=1).chunk(4)[i].bfloat16(), self.lm_head_w.bfloat16()).float()
            loss += F.cross_entropy(15 * logits * torch.rsqrt(logits.square() + 225), target_seq.chunk(4)[i]) / 4
        return loss

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

        skip_map = self.skip_map
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
        logits = F.linear(x.flatten(end_dim=1).bfloat16(), self.lm_head_w.bfloat16()).float()
        return logits, k_new_list, v_new_list

    def prefill(self, input_ids: Tensor, window: int | None = None, debug: bool = False):
        assert input_ids.ndim == 2
        B, T = input_ids.shape
        L = len(self.blocks)

        x = norm(self.embed(input_ids))
        x0 = x

        ve0, ve1, ve2 = [emb(input_ids) for emb in self.value_embeds]
        ve = [ve0, ve1, ve2] + [None] * (L - 6) + [ve0, ve1, ve2]

        skip_map = self.skip_map
        skip_weights = self.scalars[:L]
        lambdas = self.scalars[1 * L:3 * L].view(-1, 2)
        sa_lambdas = self.scalars[3 * L:5 * L].view(-1, 2)

        q = torch.arange(T, device=input_ids.device)[:, None]  # (T, 1)
        k = torch.arange(T, device=input_ids.device)[None, :]  # (1, T)
        d = q - k  # d[q, k] = q - k

        m = torch.zeros(T, T, device=input_ids.device, dtype=torch.float32)
        m[d < 0] = float("-inf")  # forbid future (k > q)
        m[d >= window] = float("-inf")  # forbid too-far past
        attn_mask = m[None, None, :, :]

        k_list, v_list, skip_connections = [], [], []
        for i in range(L):
            if i in skip_map:
                x = x + skip_weights[skip_map[i]] * skip_connections[skip_map[i]]
            x, k, v = self.blocks[i].prefill(x, ve[i], x0, lambdas[i], sa_lambdas[i], attn_mask, debug=debug)
            skip_connections.append(x)
            k_list.append(k)
            v_list.append(v)

        x = norm(x)
        logits = torch.nn.functional.linear(x[:, -1].bfloat16(), self.lm_head_w.bfloat16()).float()

        attn = next(b.attn for b in self.blocks if b.attn is not None)
        H, D = attn.num_heads, attn.head_dim
        device = x.device
        dtype = x.dtype

        K = []
        V = []
        for k, v in zip(k_list, v_list):
            if k is None:
                K.append(torch.zeros(B, H, T, D, device=device, dtype=dtype))
                V.append(torch.zeros(B, H, T, D, device=device, dtype=dtype))
            else:
                K.append(k)
                V.append(v)
        K = torch.stack(K, dim=0)
        V = torch.stack(V, dim=0)
        kv = torch.stack([K, V], dim=0)
        return logits, kv
