import torch
import torch.nn as nn


class EppieRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep the normalization statistic in fp32. This costs very little
        # compared with attention/MLP compute, but makes mixed-precision
        # pretraining less sensitive to small numerical errors.
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)

        # Return to the model parameter dtype so RMSNorm does not accidentally
        # force the rest of the block to stay in fp32.
        return self.weight * x.to(self.weight.dtype)