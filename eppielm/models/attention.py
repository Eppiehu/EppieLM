import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configuration_eppielm import EppieLMConfig
from .rope import apply_rotary_pos_emb


PastKeyValue = Tuple[torch.Tensor, torch.Tensor]


class EppieAttention(nn.Module):
    def __init__(self, config: EppieLMConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = config.num_key_value_groups

        self.dropout = config.attention_dropout
        self.attention_impl = config.attention_impl

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
        )

        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )

        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
        )

    def _shape_q(
        self,
        x: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        return (
            x.view(
                batch_size,
                seq_len,
                self.num_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )

    def _shape_kv(
        self,
        x: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        return (
            x.view(
                batch_size,
                seq_len,
                self.num_kv_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )

    def _repeat_kv(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if self.num_kv_groups == 1:
            return x

        batch_size, num_kv_heads, seq_len, head_dim = x.shape

        # KV stays compact until the eager reference path actually needs
        # expanded heads. The SDPA path can consume grouped KV directly.
        return (
            x[:, :, None, :, :]
            .expand(
                batch_size,
                num_kv_heads,
                self.num_kv_groups,
                seq_len,
                head_dim,
            )
            .reshape(
                batch_size,
                self.num_heads,
                seq_len,
                head_dim,
            )
        )

    def _build_causal_mask(
        self,
        query_len: int,
        key_len: int,
        past_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        # Absolute positions keep this mask valid for both full-sequence
        # training and cached decoding without separate masking code paths.
        query_positions = (
            torch.arange(query_len, device=device)
            + past_len
        )

        key_positions = torch.arange(
            key_len,
            device=device,
        )

        return (
            key_positions.unsqueeze(0)
            <= query_positions.unsqueeze(1)
        )

    def _eager_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_len: int,
    ) -> torch.Tensor:
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        )

        scores = scores / math.sqrt(self.head_dim)

        query_len = q.size(-2)
        key_len = k.size(-2)

        causal_mask = self._build_causal_mask(
            query_len=query_len,
            key_len=key_len,
            past_len=past_len,
            device=q.device,
        )

        scores = scores.masked_fill(
            ~causal_mask.unsqueeze(0).unsqueeze(0),
            torch.finfo(scores.dtype).min,
        )

        if attention_mask is not None:
            if attention_mask.size(-1) != key_len:
                raise ValueError(
                    "attention_mask length must match total key length, "
                    f"got {attention_mask.size(-1)} and {key_len}."
                )

            padding_mask = attention_mask[
                :,
                None,
                None,
                :
            ].bool()

            scores = scores.masked_fill(
                ~padding_mask,
                torch.finfo(scores.dtype).min,
            )

        # Softmax is intentionally evaluated in fp32. Unlike projections,
        # attention probabilities are particularly sensitive to reduced
        # precision when score ranges become large.
        probs = F.softmax(
            scores.float(),
            dim=-1,
        ).to(q.dtype)

        probs = F.dropout(
            probs,
            p=self.dropout,
            training=self.training,
        )

        return torch.matmul(probs, v)

    def _sdpa_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_len: int,
    ) -> torch.Tensor:
        query_len = q.size(-2)
        key_len = k.size(-2)

        dropout_p = (
            self.dropout
            if self.training
            else 0.0
        )

        # Avoid constructing a mask for the overwhelmingly common training
        # case. This leaves PyTorch free to select its fastest causal kernel.
        if attention_mask is None and past_len == 0:
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=dropout_p,
                is_causal=True,
                enable_gqa=(
                    self.num_heads
                    != self.num_kv_heads
                ),
            )

        causal_mask = self._build_causal_mask(
            query_len=query_len,
            key_len=key_len,
            past_len=past_len,
            device=q.device,
        )

        attn_mask = causal_mask[
            None,
            None,
            :,
            :
        ]

        if attention_mask is not None:
            if attention_mask.size(-1) != key_len:
                raise ValueError(
                    "attention_mask length must match total key length, "
                    f"got {attention_mask.size(-1)} and {key_len}."
                )

            padding_mask = attention_mask[
                :,
                None,
                None,
                :
            ].bool()

            attn_mask = attn_mask & padding_mask

        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,
            enable_gqa=(
                self.num_heads
                != self.num_kv_heads
            ),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_key_value: Optional[PastKeyValue] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[
        torch.Tensor,
        Optional[PastKeyValue],
    ]:
        batch_size, seq_len, _ = hidden_states.shape

        q = self._shape_q(
            self.q_proj(hidden_states),
            batch_size,
            seq_len,
        )

        k = self._shape_kv(
            self.k_proj(hidden_states),
            batch_size,
            seq_len,
        )

        v = self._shape_kv(
            self.v_proj(hidden_states),
            batch_size,
            seq_len,
        )

        q, k = apply_rotary_pos_emb(
            q,
            k,
            cos,
            sin,
        )

        past_len = 0

        if past_key_value is not None:
            past_k, past_v = past_key_value

            past_len = past_k.size(-2)

            if past_k.size(0) != batch_size:
                raise ValueError(
                    "Cached batch size does not match current batch size."
                )

            k = torch.cat(
                (past_k, k),
                dim=-2,
            )

            v = torch.cat(
                (past_v, v),
                dim=-2,
            )

        present_key_value = (
            (k, v)
            if use_cache
            else None
        )

        if self.attention_impl == "sdpa":
            output = self._sdpa_attention(
                q=q,
                k=k,
                v=v,
                attention_mask=attention_mask,
                past_len=past_len,
            )

        elif self.attention_impl == "eager":
            output = self._eager_attention(
                q=q,
                k=k,
                v=v,
                attention_mask=attention_mask,
                past_len=past_len,
            )

        else:
            raise ValueError(
                f"Unsupported attention_impl: {self.attention_impl}"
            )

        output = (
            output
            .transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                seq_len,
                self.hidden_size,
            )
        )

        return (
            self.o_proj(output),
            present_key_value,
        )