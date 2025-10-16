import torch

from models.gpt2 import GPT2Core
from models.gpt2.attention import CausalSelfAttention
from models.gpt2.functional import norm
import torch.nn.functional as F


def _row_softmax_with_causal_mask(scores: torch.Tensor) -> torch.Tensor:
    """Row‑wise softmax. scores: (H, T, T)."""
    T = scores.shape[-1]
    mask = torch.triu(torch.ones((T, T), dtype=torch.bool, device=scores.device), diagonal=1)
    scores = scores.masked_fill(mask, float('-inf'))
    return F.softmax(scores, dim=-1)

def attention_matrix_from_attn(attn: CausalSelfAttention) -> torch.Tensor:
    """
    Compute A for a single CausalSelfAttention module with merged QKVO weights.
    - attn.qkvo_w: (4, num_heads*head_dim, dim) with indices 0=Q, 1=K, 2=V, 3=O
    - attn.num_heads, attn.head_dim
    - X: (T, dim) hidden states entering this attention block
    Returns: A of shape (num_heads, T, T)
    """
    assert hasattr(attn, 'qkvo_w'), 'Expected attn.qkvo_w merged weights.'
    # H = int(attn.num_heads)
    # Dh = int(attn.head_dim)
    # T, D = X.shape
    #
    # # Extract Q and K weights and compute projected states.
    # Wq = attn.qkvo_w[0].to(dtype=torch.float32)  # (H*Dh, D)
    # Wk = attn.qkvo_w[1].to(dtype=torch.float32)  # (H*Dh, D)
    # Xf = X.to(dtype=torch.float32)
    # Q = Xf @ Wq.T  # (T, H*Dh)
    # K = Xf @ Wk.T  # (T, H*Dh)
    #
    # # Reshape to per‑head tensors: (B, T, H, Dh)
    # B = 1
    # Q, K = Q.view(B, T, H, Dh), K.view(B, T, H, Dh)
    # Q = attn.rotary(norm(Q))
    # K = attn.rotary(norm(K))

    Q, K = attn.last_q, attn.last_k
    assert(Q is not None and K is not None)
    assert(Q.ndim == K.ndim == 4)
    Q = Q.to(dtype=torch.float32); K = K.to(dtype=torch.float32)
    # Scores and A: (H, T, T)
    scores = torch.einsum('bthd,bshd->bhts', Q, K).squeeze(0) * attn.attn_scale
    A = _row_softmax_with_causal_mask(scores)  # B,H,T,T -> H,T,T
    return A

def effective_rank90(A_2d: torch.Tensor, device: str | torch.device = 'cpu') -> int:
    """A_2d: (T, T)."""
    # Use float64 on CPU for numerical stability
    S = torch.linalg.svdvals(A_2d.to(dtype=torch.float64, device=device))  # singular values
    S2 = S**2
    total = float(S2.sum())
    csum = torch.cumsum(S2, dim=0)
    k = int(torch.searchsorted(csum, 0.90 * total).item()) + 1
    return k

def columns90(A_2d: torch.Tensor, device: str | torch.device = 'cpu') -> int:
    """Fewest columns whose squared entries cover 90% of total squared entries."""
    # Use float64 on CPU for numerical stability
    col_mass = (A_2d.to(dtype=torch.float64, device=device)**2).sum(dim=0)  # (T,)
    vals, _ = torch.sort(col_mass, descending=True)
    csum = torch.cumsum(vals, dim=0)
    total = float(vals.sum())
    m = int(torch.searchsorted(csum, 0.90 * total).item()) + 1
    return m

def per_head_metrics(A: torch.Tensor, device: str | torch.device):
    """A: (H, T, T). Returns lists: ranks, masses, and their maxima."""
    H = A.shape[0]
    ranks, masses = [], []
    for h in range(H):
        r = effective_rank90(A[h], device=device)
        m = columns90(A[h], device=device)
        ranks.append(r)
        masses.append(m)
    return ranks, masses, max(ranks) if ranks else None

def average_per_head_over_sequences(model: GPT2Core, I: list[torch.Tensor], layer_id: int, device: str | torch.device) -> dict:
    attn = model.blocks[layer_id].attn
    H = int(attn.num_heads)
    sum_r = torch.zeros(H, dtype=torch.float64, device=device)
    sum_m = torch.zeros(H, dtype=torch.float64, device=device)
    cnt = torch.zeros(H, dtype=torch.int64, device=device)

    for i in range(len(I)):
        model.prefill_batch(I[i], window=int(I[i].size(-1)), debug=True)
        A = attention_matrix_from_attn(attn)
        for h in range(H):
            r = effective_rank90(A[h], device=device)
            m = columns90(A[h], device=device)
            sum_r[h] += r
            sum_m[h] += m
            cnt[h] += 1

    avg_r = (sum_r / cnt.clamp_min(1)).tolist()
    avg_m = (sum_m / cnt.clamp_min(1)).tolist()
    return {
        'avg_ranks_per_head': avg_r,
        'avg_columns90_per_head': avg_m,
        'MaxRank_layer': max(avg_r) if len(avg_r) else None
    }
