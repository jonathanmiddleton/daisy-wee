import os

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from models.daisy.block import Block
from models.daisy.functional import norm
from torch.nn.attention.flex_attention import BlockMask

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

_skip_map = None
def _get_skip_map(L: int):
    global _skip_map
    if _skip_map is None:
        K = max(1, L // 8)
        c = L // 2
        s = max(1, L // (2 * (K + 1)))
        _skip_map = {i: j for t in range(1, K + 1)
                for i in [c - K + (t - 1)]
                for j in [i - t * s]
                if 0 <= j < i}
    return _skip_map

class DaisyCore(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, max_seq_len: int, head_dim, window_block_size: int = 128, eos_token_id: int | None = None, desc: dict | None = None):
        super().__init__()
        if eos_token_id is None:
            raise ValueError("eos_token_id is required.")
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
        self._docs = None

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

    def _dense_window_mask_batch(self, input_ids: torch.Tensor, window_num_blocks: int):
        B, T = input_ids.shape
        bs = int(self.window_block_size)
        w = int(window_num_blocks) * bs
        device = input_ids.device

        docs = (input_ids == self.eos_token_id).cumsum(1)
        q = torch.arange(T, device=device)[:, None]
        k = torch.arange(T, device=device)[None, :]

        causal = q >= k
        within = k >= (q - w + 1).clamp_min(0)
        same_doc = docs[:, :, None] == docs[:, None, :]

        allowed = same_doc & (causal & within)[None].expand(B, -1, -1)

        m = torch.full((B, 1, T, T), float("-inf"), device=device, dtype=torch.float32)
        m[:, 0].masked_fill_(allowed, 0.0)
        return m

    def _layer_additive_masks(self, input_ids: torch.Tensor, window_tokens: int):
        assert window_tokens is not None
        bs = int(self.window_block_size)
        long_blocks = max(1, int(window_tokens) // bs)
        short_blocks = max(1, long_blocks // 2)

        long_m = self._dense_window_mask_batch(input_ids, long_blocks)
        short_m = self._dense_window_mask_batch(input_ids, short_blocks)

        L = len(self.blocks)
        cycle = [long_m, short_m, short_m, short_m]
        return (cycle * ((L + 3) // 4))[:L - 1] + [long_m]

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

        skip_map = _get_skip_map(L)
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
        h = d = None
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

        skip_map = _get_skip_map(L)
        scalars = self.scalars
        skip_weights = scalars[:L]
        lambdas = scalars[1 * L:3 * L].view(-1, 2)
        sa_lambdas = scalars[3 * L:5 * L].view(-1, 2)
        x = x0

        doc_prev = (self._docs[-1] if self._docs.numel() > 0 else torch.tensor(0, device=token_id.device))
        doc_q = (doc_prev + (token_id == self.eos_token_id).to(doc_prev.dtype)).to(self._docs.dtype)
        self._docs = torch.cat([self._docs, doc_q.view(1)], dim=0)

        k_new_list, v_new_list, skip_connections = [], [], []
        bs = int(self.window_block_size)

        for i in range(L):
            if i in skip_map:
                x = x + skip_weights[skip_map[i]] * skip_connections[skip_map[i]]

            use_long = (i % 4 == 0) or (i == L - 1)
            w_tokens = window if use_long else max(window // 2, bs)

            k_ctx_i = k_ctxs[i]
            v_ctx_i = v_ctxs[i]
            if getattr(self.blocks[i], "attn", None) is None:
                y, k_new, v_new = x + self.blocks[i].mlp(norm(x)), None, None
            else:
                n = 0 if k_ctx_i is None else k_ctx_i.size(1)
                r = n if w_tokens is None else min(n, max(w_tokens - 1, 0))
                if r > 0:
                    keys_docs = self._docs[n - r:n]
                    same = (keys_docs == doc_q)
                    row = torch.full((r + 1,), float("-inf"), device=token_id.device, dtype=torch.float32)
                    row[:r][same] = 0.0
                    row[r] = 0.0
                    attn_mask = row.view(1, 1, 1, -1)
                else:
                    attn_mask = torch.zeros(1, 1, 1, 1, device=token_id.device, dtype=torch.float32)

                y, k_new, v_new = self.blocks[i].step(
                    x, ve[i], x0, k_ctx_i, v_ctx_i, pos, lambdas[i], sa_lambdas[i], w_tokens, attn_mask
                )
                x = y

            skip_connections.append(x)
            k_new_list.append(k_new)
            v_new_list.append(v_new)

        return x.view(1, -1).float(), k_new_list, v_new_list

    def prefill_batch(self, input_ids: Tensor, window: int | None = None, debug: bool = False):
        assert input_ids.ndim == 2
        B, T = input_ids.shape
        L = len(self.blocks)

        x = norm(self.embed(input_ids))
        x0 = x

        ve0, ve1, ve2 = [emb(input_ids) for emb in self.value_embeds]
        ve = [ve0, ve1, ve2] + [None] * (L - 6) + [ve0, ve1, ve2]

        skip_map = _get_skip_map(L)
        skip_weights = self.scalars[:L]
        lambdas = self.scalars[1 * L:3 * L].view(-1, 2)
        sa_lambdas = self.scalars[3 * L:5 * L].view(-1, 2)

        layer_masks = self._layer_additive_masks(input_ids, window)

        k_list, v_list, skip_connections = [], [], []
        for i in range(L):
            if i in skip_map:
                x = x + skip_weights[skip_map[i]] * skip_connections[skip_map[i]]
            x, k, v = self.blocks[i].prefill(x, ve[i], x0, lambdas[i], sa_lambdas[i], layer_masks[i], debug=debug)
            skip_connections.append(x)
            k_list.append(k)
            v_list.append(v)

        x = norm(x)
        logits = torch.nn.functional.linear(x[:, -1].bfloat16(), self.lm_head_w.bfloat16()).float()

        attn = next(b.attn for b in self.blocks if b.attn is not None)
        H, D = attn.num_heads, attn.head_dim
        device = x.device
        dtype = x.dtype

        if input_ids.size(0) == 1:
            self._docs = (input_ids == self.eos_token_id).cumsum(1)[0].detach()

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
