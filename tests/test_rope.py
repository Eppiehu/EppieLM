import torch

from eppielm.models.rope import (
    EppieRotaryEmbedding,
    apply_rotary_pos_emb,
)


def test_rope_preserves_shape():
    batch_size = 2
    seq_len = 16
    head_dim = 64

    rope = EppieRotaryEmbedding(head_dim=head_dim)

    q = torch.randn(batch_size, 12, seq_len, head_dim)
    k = torch.randn(batch_size, 4, seq_len, head_dim)

    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    cos, sin = rope(position_ids, dtype=q.dtype)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


def test_rope_position_zero_is_identity():
    head_dim = 64

    rope = EppieRotaryEmbedding(head_dim=head_dim)

    q = torch.randn(1, 12, 1, head_dim)
    k = torch.randn(1, 4, 1, head_dim)

    position_ids = torch.zeros((1, 1), dtype=torch.long)

    cos, sin = rope(position_ids, dtype=q.dtype)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

    assert torch.allclose(q_rot, q, atol=1e-6)
    assert torch.allclose(k_rot, k, atol=1e-6)


def test_rope_preserves_vector_norm():
    batch_size = 2
    seq_len = 32
    head_dim = 64

    rope = EppieRotaryEmbedding(head_dim=head_dim)

    q = torch.randn(batch_size, 12, seq_len, head_dim)
    k = torch.randn(batch_size, 4, seq_len, head_dim)

    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    cos, sin = rope(position_ids, dtype=q.dtype)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

    q_norm = q.float().norm(dim=-1)
    q_rot_norm = q_rot.float().norm(dim=-1)

    k_norm = k.float().norm(dim=-1)
    k_rot_norm = k_rot.float().norm(dim=-1)

    assert torch.allclose(q_norm, q_rot_norm, atol=1e-5, rtol=1e-5)
    assert torch.allclose(k_norm, k_rot_norm, atol=1e-5, rtol=1e-5)


def test_rope_rejects_odd_head_dim():
    try:
        EppieRotaryEmbedding(head_dim=63)
    except ValueError:
        return

    raise AssertionError("Expected ValueError for odd head_dim.")