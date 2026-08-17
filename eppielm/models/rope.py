import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class EppieRotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
    ):
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError(
                f"RoPE requires an even head_dim, got {head_dim}."
            )

        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # inv_freq depends only on the architecture, not on training.
        # Keeping it as a non-persistent buffer moves it with the model
        # without unnecessarily storing it in every checkpoint.
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32)
                / head_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self,
        position_ids: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.max().item() >= self.max_position_embeddings:
            raise ValueError(
                "position_ids exceed max_position_embeddings: "
                f"{position_ids.max().item()} >= "
                f"{self.max_position_embeddings}"
            )

        # Build angles in fp32 even when attention runs in bf16/fp16.
        # RoPE is cheap relative to attention, while low-precision trig
        # functions can accumulate avoidable phase error at long positions.
        positions = position_ids.to(
            device=self.inv_freq.device,
            dtype=torch.float32,
        )

        freqs = torch.einsum("bt,d->btd", positions, self.inv_freq)
        angles = torch.cat((freqs, freqs), dim=-1)

        cos = angles.cos().to(dtype=dtype)
        sin = angles.sin().to(dtype=dtype)

        # Attention tensors will use [batch, heads, seq, head_dim].
        # Keeping the head axis here avoids repeated unsqueeze operations
        # inside every attention layer.
        return cos.unsqueeze(1), sin.unsqueeze(1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed