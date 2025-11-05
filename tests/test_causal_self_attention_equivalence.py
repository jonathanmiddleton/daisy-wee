
import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask, BlockMask

from daisy.daisy_core import build_attn_mask
from models.daisy.attention import CausalSelfAttention
from optim import get_num_window_blocks


def _make_block_mask(T: int, H: int, W: int, device: torch.device):
    dev_str = "cuda" if device.type == "cuda" else "cpu"
    def mask_mod(b, h, q_idx, kv_idx):
        return (kv_idx <= q_idx) & (kv_idx >= q_idx - (W - 1))
    return create_block_mask(mask_mod, 1, H, T, T, device=dev_str)

def create_blockmasks(input_seq: torch.Tensor, sliding_window_num_blocks: torch.Tensor, L: int):
    BLOCK_SIZE = 128
    assert (len(input_seq) % BLOCK_SIZE == 0)
    device = input_seq.device
    docs = (input_seq == 50256).cumsum(0)

    def document_causal(b, h, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx
        document_mask = docs[q_idx] == docs[kv_idx]
        return causal_mask & document_mask

    def dense_to_ordered(dense_blockmask: torch.Tensor):
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

    def build_bm(window_size_blocks: torch.Tensor) -> BlockMask:
        return BlockMask.from_kv_blocks(
            torch.clamp_max(partial_kv_num_blocks, torch.clamp_min(window_size_blocks - full_kv_num_blocks, 1)),
            partial_kv_indices,
            torch.clamp_max(full_kv_num_blocks, window_size_blocks - 1),
            full_kv_indices,
            BLOCK_SIZE=BLOCK_SIZE,
            mask_mod=document_causal,
        )

    # Long-short SWA block masks by @leloykun & @YouJiacheng, adapated from suggestion by @Grad62304977, following Gemma 2 paper
    long_bm, short_bm = build_bm(sliding_window_num_blocks), build_bm(sliding_window_num_blocks // 2)

    cycle = [long_bm] + [short_bm] * 3
    block_masks = (cycle * ((L + 3) // 4))[:L - 1] + [long_bm]
    return block_masks

# @pytest.mark.parametrize("T,W", [(128,128), (131,128)])
# @pytest.mark.parametrize("use_ve", [False, True])
@pytest.mark.parametrize("T,W", [(2,128)])
@pytest.mark.parametrize("use_ve", [False])
def test_causal_self_attention_forward_equals_step(T, W, use_ve, monkeypatch):
    # TODO fix this test since it will spuriously fail for small T,W
    monkeypatch.setenv("DISABLE_O_ZERO_INIT", "1")
    with torch.inference_mode():
        torch.manual_seed(0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        H, D = 1, 2
        dim, max_seq_len = H * D, 256
        m = CausalSelfAttention(dim=dim, num_heads=H, max_seq_len=max_seq_len, head_dim=D).to(device).eval()
        dtype = torch.bfloat16
        x_t = torch.ones(T, dtype=torch.int) #pseudo-tokens
        x = torch.randn(1, T, dim, device=device, dtype=dtype) #pseudo-embedding
        lambdas = torch.tensor([1.0, 0.5], device=device, dtype=dtype)
        ve_full = None
        if use_ve:
            ve_full = torch.randn(1, T, H, D, device=device, dtype=dtype)
        # bm = create_blockmasks(x_t, window_size=swnb, L=0)[0]
        bm = None
        attn_mask = build_attn_mask(x_t, window_size=W)
        y_full, K_full, V_full, Q_full = m.forward(x, ve=ve_full, block_mask=bm, lambdas=lambdas, attn_mask=attn_mask)
        # Initialize empty KV caches and append per token to match causal semantics
        K = torch.empty(1, 0, H, D, device=device, dtype=dtype)
        V = torch.empty(1, 0, H, D, device=device, dtype=dtype)
        Q = torch.empty(1, 0, H, D, device=device, dtype=dtype)
        ys = []
        for pos in range(T):
            x_pos = x[:, pos:pos+1]
            ve_pos = None if ve_full is None else ve_full[:, pos:pos+1]
            # step(x, k_ctx: Tensor, v_ctx: Tensor, pos: int, ve: Tensor | None, lambdas: Tensor, window: int)
            y_pos, k_new, v_new, q_new = m.step(x_pos, k_ctx=K, v_ctx=V, pos=pos, ve=ve_pos, lambdas=lambdas, window=W)
            K = torch.cat([K, k_new], dim=1)
            V = torch.cat([V, v_new], dim=1)
            Q = torch.cat([Q, q_new], dim=1)
            ys.append(y_pos)
        y_step = torch.cat(ys, dim=1)
        assert K.shape == K_full.shape
        assert V.shape == V_full.shape
        assert torch.allclose(K, K_full, atol=1e-4, rtol=1e-4)
        assert torch.allclose(V, V_full, atol=1e-4, rtol=1e-4)
        assert torch.allclose(Q, Q_full, atol=1e-4, rtol=1e-4)
        assert y_full.shape == y_step.shape
        max_err = (y_full.float() - y_step.float()).abs().max().item()
        print("\n")
        print(f"y_full: {y_full}")
        print(f"y_step: {y_step}")
        assert torch.allclose(y_full, y_step, atol=2e-2, rtol=2e-2), f"max_err={max_err}"
