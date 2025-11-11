import os
from functools import lru_cache
from math import floor, log2, ceil
from typing import Any, Optional

import torch
from torch import nn, Tensor, SymInt
import torch.nn.functional as F
from models.daisy.block import Block
from models.daisy.functional import norm


def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

class ZeroEmbedding(nn.Module):
    def __init__(self, end_dim: int, device: torch.device, dtype: torch.dtype = torch.int64, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.end_dim = end_dim
        self.zero = nn.Buffer(torch.zeros(1, dtype=dtype, device=device),
                              persistent=False)  # anchor for device/dtype so that we're moved when .to is called

    @lru_cache(maxsize=1, typed=True)
    def __call__(self, x: Tensor):
        # Return a zero tensor shaped like an embedding(x)
        out_shape = (*x.shape, self.end_dim)
        return torch.zeros(out_shape, dtype=self.zero.dtype, device=self.zero.device, requires_grad=False)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)

        try:
            self.__call__.cache_clear()
        except AttributeError:
            pass

        return self


def pick_value_embedding_layers(attn_layers, M=None):
    K = len(attn_layers)
    if K == 0:
        return []
    if K == 1:
        return attn_layers[:]  # trivial case

    if M is None:
        M = min(K, max(4, min(10, round(0.6 * K))))

    if M >= K:
        return attn_layers[:]

    idx = [int(round(i * (K - 1) / (M - 1))) for i in range(M)]
    return [attn_layers[i] for i in sorted(set(idx))]


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
    K = min(total_layers, max(ceil(total_layers / s), ceil(2 + log2(total_layers))))
    idx = [int(round(i * (total_layers - 1) / (K - 1))) for i in range(K)]
    return sorted(set(idx))


class DaisyCore(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, max_seq_len: int,
                 head_dim: int, window_size: int = 1024, eos_token_id: int | None = None, desc: dict | None = None,
                 value_embeddings: bool = True, tied_embeddings: bool = False, attn_all_layers: bool = False,
                 attn_impl: str = 'standard'):
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
        self.attn_layers = [i for i in range(num_layers)] if attn_all_layers else pick_attention_layers(num_layers)
        self.ve_layers = pick_value_embedding_layers(self.attn_layers) if value_embeddings else []
        self.zero_embedding = ZeroEmbedding(end_dim=self.embed.weight.size(1), device=self.embed.weight.device,
                                            dtype=torch.bfloat16)
        self.value_embeds = nn.ModuleList([
            nn.Embedding(vocab_size, model_dim) if i in self.ve_layers else self.zero_embedding for i in range(num_layers)
        ])
        # self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(len(self.ve_layers))])

        self.blocks = nn.ModuleList(
            [Block(model_dim, num_heads, max_seq_len, i, head_dim, i in self.attn_layers, attn_impl) for i in range(num_layers)])
        if tied_embeddings:
            nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
            self.lm_head_w = self.embed.weight
        else:
            if os.getenv("DISABLE_O_ZERO_INIT", "") != "1":
                # != 1 training
                self.lm_head_w = nn.Parameter(torch.zeros(next_multiple_of_n(vocab_size, n=128), model_dim))
            else:
                # == 1 to allow backpropagation for lr_sweep or cases where the LM head is frozen for testing
                self.lm_head_w = nn.Parameter(torch.empty(next_multiple_of_n(vocab_size, n=128), model_dim))
                nn.init.normal_(self.lm_head_w, mean=0.0, std=0.02)
        self.window_size = window_size
        assert num_layers % 2 == 0
        self.scalars = nn.Parameter(torch.cat([
            torch.ones(num_layers),  # skip_weights
            *[torch.tensor([1.0, 0.0]) for _ in range(num_layers)],  # residual mixing
            *[torch.tensor([0.5, 0.5]) for _ in range(num_layers)],  # value embedding mixing
        ]))
        self.desc = desc  # non-functional, self-describing metadata

    def reset_history(self):
        for b in self.blocks:
            b.reset_history()


    def compute_value_embeddings(self, input_seq: Tensor) -> list[Tensor]:
        ve = [norm(value_embed(input_seq)) for value_embed in self.value_embeds]
        return ve

    def forward(self, input_seq: Tensor, sliding_window_num_blocks: Tensor, target_seq: Tensor = None):
        assert input_seq.ndim == 1
        L = len(self.blocks)

        ve = self.compute_value_embeddings(input_seq)

        x = x0 = norm(self.embed(input_seq)[None])

        skip_map = self.skip_map
        skip_weights = self.scalars[:L]
        lambdas = self.scalars[1 * L:3 * L].view(-1, 2)
        sa_lambdas = self.scalars[3 * L:5 * L].view(-1, 2)

        skip_connections = []

        for i in range(L):
            if i in skip_map:
                x = x + skip_weights[skip_map[i]] * skip_connections[skip_map[i]]
            x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], sliding_window_num_blocks)
            skip_connections.append(x)

        x = norm(x)
        if self.training:
            logits: Tensor = F.linear(x.flatten(end_dim=1).bfloat16(), self.lm_head_w.bfloat16()).float()
            loss = F.cross_entropy(15 * logits * torch.rsqrt(logits.square() + 225), target_seq)
            return loss

        # eval, assuming 4xtrain_seq_len
        loss = 0
        for i in range(4):
            logits: Tensor = F.linear(x.flatten(end_dim=1).chunk(4)[i].bfloat16(), self.lm_head_w.bfloat16()).float()
            loss += F.cross_entropy(15 * logits * torch.rsqrt(logits.square() + 225), target_seq.chunk(4)[i]) / 4
        return loss

    def step(self, token_id: Tensor, k_ctxs, v_ctxs, pos: int, window: int):
        assert token_id.ndim == 0
        B = T = 1
        token_id = token_id.view(B, T)
        x0 = norm(self.embed(token_id))
        L = len(self.blocks)

        ve = self.compute_value_embeddings(token_id)

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
            y, k_new, v_new = self.blocks[i].step(x, ve[i], x0, k_ctxs[i], v_ctxs[i], pos, lambdas[i], sa_lambdas[i], window)
            x = y
            skip_connections.append(x)
            k_new_list.append(k_new)
            v_new_list.append(v_new)
        x = norm(x)
        logits = F.linear(x.flatten(end_dim=1).bfloat16(), self.lm_head_w.bfloat16()).float()
        return logits, k_new_list, v_new_list

    def prefill(self, input_seq: Tensor, window: Optional[int] = None, debug: bool = False):
        assert input_seq.ndim == 2
        B, T = input_seq.shape
        h = None
        d = None
        for b in self.blocks:
            if getattr(b, "attn", None) is not None:
                h = b.attn.num_heads
                d = b.attn.head_dim
                break
        L = len(self.blocks)

        x = norm(self.embed(input_seq))
        x0 = x

        ve = self.compute_value_embeddings(input_seq)

        skip_map = self.skip_map
        skip_weights = self.scalars[:L]
        lambdas = self.scalars[1 * L:3 * L].view(-1, 2)
        sa_lambdas = self.scalars[3 * L:5 * L].view(-1, 2)

        k_list, v_list, skip_connections = [], [], []
        for i in range(L):
            if i in skip_map:
                x = x + skip_weights[skip_map[i]] * skip_connections[skip_map[i]]
            x, k, v = self.blocks[i].prefill(x, ve[i], x0, lambdas[i], sa_lambdas[i], debug=debug)
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
