from typing import runtime_checkable, Protocol, Optional, Tuple
from torch import Tensor

@runtime_checkable
class AttentionProtocol(Protocol):
    """
    Common protocol for attention modules used in Daisy.
    """

    num_heads: int
    head_dim: int
    # qkvo_w: torch.nn.Parameter # was used to capture dtype, should not use this

    def reset_history(self) -> None: ...

    def forward(
        self,
        x: Tensor,
        ve: Tensor,
        sa_lambdas: Tensor,
        attn_mask: Tensor,
    ) -> Tensor: ...

    def __call__(self, *args, **kwargs) -> Tensor: ...

    def prefill(
        self,
        x: Tensor,
        ve: Tensor,
        sa_lambdas: Tensor,
        attn_mask: Tensor,
        debug: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]: ...

    def step(
        self,
        x: Tensor,
        k_ctx: Optional[Tensor],
        v_ctx: Optional[Tensor],
        pos: int,
        ve: Tensor,
        sa_lambdas: Tensor,
        window: Optional[int],
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]: ...
