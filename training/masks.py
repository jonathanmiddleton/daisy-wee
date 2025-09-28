import torch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask


def create_blockmasks(input_seq: Tensor, sliding_window_num_blocks: Tensor):
    BLOCK_SIZE = 128
    docs = (input_seq == 50256).cumsum(0)

    def document_causal(b, h, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx
        document_mask = docs[q_idx] == docs[kv_idx]
        return causal_mask & document_mask

    def dense_to_ordered(dense_blockmask: Tensor):
        num_blocks = dense_blockmask.sum(dim=-1, dtype=torch.int32)
        indices = dense_blockmask.argsort(dim=-1, descending=False, stable=True).flip(-1).to(torch.int32)
        return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

    # manual block mask creation by @YouJiacheng
    assert len(input_seq) % BLOCK_SIZE == 0
    NUM_BLOCKS = len(input_seq) // BLOCK_SIZE
    block_idx = torch.arange(NUM_BLOCKS, dtype=torch.int32, device="cuda")
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