import torch
import torch.nn as nn
import torch.nn.functional as F


class EppieMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
    ):
        super().__init__()

        self.gate_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=bias,
        )
        self.up_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=bias,
        )
        self.down_proj = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep the gate and value projections separate here instead of
        # immediately fusing them. It makes architecture ablations easier
        # and gives us a clean baseline before experimenting with fused kernels.
        gated = F.silu(self.gate_proj(x)) * self.up_proj(x)

        return self.down_proj(gated)