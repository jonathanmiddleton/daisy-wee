
import torch
import torch.nn.functional as F

from models.gpt2.attention import CausalSelfAttention
from models.gpt2.functional import norm
from models import model_from_spec
from models.gpt2.gpt_core import GPT2Core
import tiktoken


torch.set_printoptions(precision=4, sci_mode=True)
device = 'cpu'
CHECKPOINT_PATH = "/Users/jonathanmiddleton/models/checkpoints/350m-instruct/20251013T1953-val1.750-step000450-run1-best.pt"

def _row_softmax_with_causal_mask(scores: torch.Tensor) -> torch.Tensor:
    """Row‑wise softmax. scores: (H, T, T)."""
    T = scores.shape[-1]
    mask = torch.triu(torch.ones((T, T), dtype=torch.bool, device=scores.device), diagonal=1)
    scores = scores.masked_fill(mask, float('-inf'))
    return F.softmax(scores, dim=-1)

def attention_matrix_from_attn(attn: CausalSelfAttention, X: torch.Tensor) -> torch.Tensor:

    assert hasattr(attn, 'qkvo_w'), 'Expected attn.qkvo_w merged weights.'
    H = int(attn.num_heads)
    Dh = int(attn.head_dim)
    T, D = X.shape

    # Extract Q and K weights and compute projected states.
    Wq = attn.qkvo_w[0].to(dtype=torch.float32)  # (H*Dh, D)
    Wk = attn.qkvo_w[1].to(dtype=torch.float32)  # (H*Dh, D)
    Xf = X.to(dtype=torch.float32)
    Q = Xf @ Wq.T  # (T, H*Dh)
    K = Xf @ Wk.T  # (T, H*Dh)

    # Reshape to per‑head tensors: (B, T, H, Dh)
    B = 1
    Q, K = Q.view(B, T, H, Dh), K.view(B, T, H, Dh)
    Q = attn.rotary(norm(Q))
    K = attn.rotary(norm(K))

    scores = torch.einsum('bthd,bshd->bhts', Q, K) * attn.attn_scale

    ###### check equivalence with original implementation
    w = attn.qkvo_w[:3].to(dtype=torch.float32)
    w = w.flatten(end_dim=1)
    qkv = torch.nn.functional.linear(Xf, w).view(B, T, 3 * H, Dh)
    q, k, v = qkv.chunk(3, dim=-2)
    q, k = norm(q), norm(k)
    q, k = attn.rotary(q), attn.rotary(k)
    assert torch.allclose(q, Q, rtol=1e-5, atol=1e-5)
    assert torch.allclose(k, K, rtol=1e-5, atol=1e-5)
    ######

    A = _row_softmax_with_causal_mask(scores.squeeze(0)) # B,H,T,T -> H,T,T
    return A


def effective_rank90(A_2d: torch.Tensor) -> int:
    """A_2d: (T, T)."""
    # Use float64 on CPU for numerical stability
    S = torch.linalg.svdvals(A_2d.to(dtype=torch.float64, device=device))  # singular values
    S2 = S**2
    total = float(S2.sum())
    csum = torch.cumsum(S2, dim=0)
    k = int(torch.searchsorted(csum, 0.90 * total).item()) + 1
    return k

def columns90(A_2d: torch.Tensor) -> int:
    """Fewest columns whose squared entries cover 90% of total squared entries."""
    col_mass = (A_2d**2).sum(dim=0)  # (T,)
    vals, _ = torch.sort(col_mass.to(dtype=torch.float64, device=device), descending=True)
    csum = torch.cumsum(vals, dim=0)
    total = float(vals.sum())
    m = int(torch.searchsorted(csum, 0.90 * total).item()) + 1
    return m

def per_head_metrics(A: torch.Tensor):
    """A: (H, T, T). Returns lists: ranks, masses, and their maxima."""
    H = A.shape[0]
    ranks, masses = [], []
    for h in range(H):
        r = effective_rank90(A[h])
        m = columns90(A[h])
        ranks.append(r)
        masses.append(m)
    return ranks, masses, max(ranks) if ranks else None



with torch.no_grad():
    model: GPT2Core = model_from_spec('gpt2_350m', device=device).eval()
    enc = tiktoken.get_encoding("gpt2")
    s = " "
    tokens = enc.encode(s, allowed_special={"<|endoftext|>"})
    p = torch.tensor(tokens, dtype=torch.long)[None, :]
    if CHECKPOINT_PATH is not None:
        from tools.checkpoint import load_checkpoint, apply_model_state
        ckpt = load_checkpoint(CHECKPOINT_PATH, map_location=device)
        state_dict = ckpt.model
        apply_model_state(model, state_dict, strict=True)
    model.prefill_batch(p, 256)

    for i in range(len(model.blocks)):
        block = model.blocks[i]
        X: torch.Tensor = block.in_t[0] # (T, dim)
        attn: CausalSelfAttention = model.blocks[i].attn
        A = attention_matrix_from_attn(attn, X)
        print(A)
        ranks, masses, max_rank = per_head_metrics(A)
        print('Per‑head effective ranks:', ranks)
        print('Per‑head columns@90%:', masses)
        print('MaxRank(layer) =', max_rank)
