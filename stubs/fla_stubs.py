"""
FLA (Flash Linear Attention) stubs for development on non-CUDA platforms.
The fla library requires CUDA and triton, so we provide stubs for import compatibility.
"""
from typing import Optional, Tuple
import torch
from torch import nn, Tensor


# Stub for ShortConvolution - needs to be instantiable and callable
class ShortConvolution(nn.Module):
    """Stub for fla.modules.ShortConvolution that can be instantiated but not used."""

    def __init__(self, hidden_size: int, kernel_size: int = 4, activation: str = "silu", bias: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.activation = activation
        # Create a dummy conv1d so the module has parameters
        self.conv1d = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=bias,
            padding=kernel_size - 1,
        )

    def forward(self, x: Tensor, cache: Optional[Tensor] = None,
                output_final_state: bool = False, cu_seqlens: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass that raises error on actual use."""
        raise RuntimeError(
            "ShortConvolution requires CUDA (fla library). "
            "This stub allows import but cannot execute. "
            "Run on CUDA device to use KimiLinearSelfAttention."
        )


# Stub for FusedRMSNormGated - needs to be instantiable and callable
class FusedRMSNormGated(nn.Module):
    """Stub for fla.modules.FusedRMSNormGated that can be instantiated but not used."""

    def __init__(self, hidden_size: int, activation: str = "sigmoid", eps: float = 1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.activation = activation
        # Create dummy weight so the module has parameters
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: Tensor, gate: Optional[Tensor] = None) -> Tensor:
        """Forward pass that raises error on actual use."""
        raise RuntimeError(
            "FusedRMSNormGated requires CUDA (fla library). "
            "This stub allows import but cannot execute. "
            "Run on CUDA device to use KimiLinearSelfAttention."
        )


# Stub functions for operations
def chunk_kda(*args, **kwargs):
    raise RuntimeError(
        "chunk_kda requires CUDA (fla library). "
        "Run on CUDA device to use KimiLinearSelfAttention."
    )


def fused_recurrent_kda(*args, **kwargs):
    raise RuntimeError(
        "fused_recurrent_kda requires CUDA (fla library). "
        "Run on CUDA device to use KimiLinearSelfAttention."
    )


def fused_kda_gate(*args, **kwargs):
    raise RuntimeError(
        "fused_kda_gate requires CUDA (fla library). "
        "Run on CUDA device to use KimiLinearSelfAttention."
    )


# Stub for Cache class
class Cache:
    def __init__(self):
        self._cache = {}

    def __len__(self):
        return len(self._cache)

    def __getitem__(self, key):
        return self._cache.get(key)

    def update(self, **kwargs):
        layer_idx = kwargs.get('layer_idx', 0)
        self._cache[layer_idx] = kwargs


# Utility functions (these can have simple working implementations)
def get_unpad_data(mask):
    if mask is None:
        return None, None, None
    torch._assert(
        mask.dtype in (torch.long, torch.int, torch.bool),
        "mask dtype unsupported"
    )
    B, S = mask.shape
    idx = torch.arange(B * S, device=mask.device)
    return idx, None, S


def index_first_axis(x, idx):
    return x


def pad_input(x, indices, batch_size, q_len):
    return x.squeeze(0)


__all__ = [
    'ShortConvolution',
    'FusedRMSNormGated',
    'chunk_kda',
    'fused_recurrent_kda',
    'fused_kda_gate',
    'Cache',
    'get_unpad_data',
    'index_first_axis',
    'pad_input',
]