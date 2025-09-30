
import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask
from models.attention import CausalSelfAttention

def _make_block_mask(T: int, H: int, W: int, device: torch.device):
    dev_str = "cuda" if device.type == "cuda" else "cpu"
    def mask_mod(b, h, q_idx, kv_idx):
        return (kv_idx <= q_idx) & (kv_idx >= q_idx - (W - 1))
    return create_block_mask(mask_mod, 1, H, T, T, device=dev_str)

@pytest.mark.parametrize("T,W", [(9,9), (13,5)])
@pytest.mark.parametrize("use_ve", [False, True])
def test_causal_self_attention_forward_equals_step(T, W, use_ve, monkeypatch):
    monkeypatch.setenv("DISABLE_O_ZERO_INIT", "1")
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H, D = 4, 8
    dim, max_seq_len = H * D, 64
    m = CausalSelfAttention(dim=dim, num_heads=H, max_seq_len=max_seq_len, head_dim=D).to(device).eval()
    dtype = torch.bfloat16
    x = torch.randn(1, T, dim, device=device, dtype=dtype)
    lambdas = torch.tensor([1.0, 0.5], device=device, dtype=dtype)
    ve_full = None
    if use_ve:
        ve_full = torch.randn(1, T, H, D, device=device, dtype=dtype)
    bm = _make_block_mask(T, H, W, device)
    y_full = m.forward(x, ve_full, bm, lambdas)
    K = torch.zeros(1, T, H, D, device=device, dtype=dtype)
    V = torch.zeros(1, T, H, D, device=device, dtype=dtype)
    ys = []
    for pos in range(T):
        x_pos = x[:, pos:pos+1]
        ve_pos = None if ve_full is None else ve_full[:, pos:pos+1]
        y_pos, k_new, v_new = m.step(x_pos, K, V, pos, ve_pos, lambdas, W)
        K[:, pos:pos+1] = k_new
        V[:, pos:pos+1] = v_new
        ys.append(y_pos)
    y_step = torch.cat(ys, dim=1)
    assert y_full.shape == y_step.shape
    max_err = (y_full.float() - y_step.float()).abs().max().item()
    assert torch.allclose(y_full, y_step, atol=2e-2, rtol=2e-2), f"max_err={max_err}"
