import torch

from eppielm.models.attention import EppieAttention
from eppielm.models.configuration_eppielm import EppieLMConfig
from eppielm.models.rope import EppieRotaryEmbedding


def build_inputs(seq_len: int = 16):
    config = EppieLMConfig(
        hidden_size=768,
        num_attention_heads=12,
        num_key_value_heads=4,
        attention_dropout=0.0,
    )

    x = torch.randn(
        2,
        seq_len,
        config.hidden_size,
    )

    position_ids = (
        torch.arange(seq_len)
        .unsqueeze(0)
        .expand(2, -1)
    )

    rope = EppieRotaryEmbedding(
        head_dim=config.head_dim,
        max_position_embeddings=config.max_position_embeddings,
    )

    cos, sin = rope(
        position_ids,
        dtype=x.dtype,
    )

    return (
        config,
        x,
        position_ids,
        rope,
        cos,
        sin,
    )


def test_attention_preserves_shape():
    config, x, _, _, cos, sin = build_inputs()

    attention = EppieAttention(config)
    attention.eval()

    y, cache = attention(
        x,
        cos,
        sin,
    )

    assert y.shape == x.shape
    assert cache is None


def test_attention_output_is_finite():
    config, x, _, _, cos, sin = build_inputs()

    attention = EppieAttention(config)
    attention.eval()

    y, _ = attention(
        x,
        cos,
        sin,
    )

    assert torch.isfinite(y).all()


def test_attention_cache_shapes():
    config, x, _, _, cos, sin = build_inputs()

    attention = EppieAttention(config)
    attention.eval()

    _, cache = attention(
        x,
        cos,
        sin,
        use_cache=True,
    )

    assert cache is not None

    k, v = cache

    expected_shape = (
        2,
        config.num_key_value_heads,
        16,
        config.head_dim,
    )

    assert k.shape == expected_shape
    assert v.shape == expected_shape


def test_eager_and_sdpa_are_close():
    config, x, _, _, cos, sin = build_inputs()

    sdpa_config = EppieLMConfig(
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        attention_dropout=0.0,
        attention_impl="sdpa",
    )

    eager_config = EppieLMConfig(
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        attention_dropout=0.0,
        attention_impl="eager",
    )

    sdpa_attention = EppieAttention(sdpa_config)
    eager_attention = EppieAttention(eager_config)

    eager_attention.load_state_dict(
        sdpa_attention.state_dict()
    )

    sdpa_attention.eval()
    eager_attention.eval()

    y_sdpa, _ = sdpa_attention(
        x,
        cos,
        sin,
    )

    y_eager, _ = eager_attention(
        x,
        cos,
        sin,
    )

    assert torch.allclose(
        y_sdpa,
        y_eager,
        atol=1e-5,
        rtol=1e-4,
    )


def test_cached_decoding_matches_full_sequence():
    (
        config,
        x,
        _,
        rope,
        _,
        _,
    ) = build_inputs(seq_len=8)

    config.attention_impl = "eager"

    attention = EppieAttention(config)
    attention.eval()

    full_positions = (
        torch.arange(8)
        .unsqueeze(0)
        .expand(2, -1)
    )

    full_cos, full_sin = rope(
        full_positions,
        dtype=x.dtype,
    )

    full_output, _ = attention(
        x,
        full_cos,
        full_sin,
    )

    first_x = x[:, :7]
    last_x = x[:, 7:]

    first_positions = (
        torch.arange(7)
        .unsqueeze(0)
        .expand(2, -1)
    )

    last_positions = torch.full(
        (2, 1),
        7,
        dtype=torch.long,
    )

    first_cos, first_sin = rope(
        first_positions,
        dtype=x.dtype,
    )

    last_cos, last_sin = rope(
        last_positions,
        dtype=x.dtype,
    )

    _, cache = attention(
        first_x,
        first_cos,
        first_sin,
        use_cache=True,
    )

    cached_output, _ = attention(
        last_x,
        last_cos,
        last_sin,
        past_key_value=cache,
        use_cache=True,
    )

    assert torch.allclose(
        full_output[:, -1:],
        cached_output,
        atol=1e-5,
        rtol=1e-4,
    )


def test_attention_rejects_bad_mask_length():
    config, x, _, _, cos, sin = build_inputs()

    config.attention_impl = "eager"

    attention = EppieAttention(config)
    attention.eval()

    bad_mask = torch.ones(
        2,
        15,
        dtype=torch.long,
    )

    try:
        attention(
            x,
            cos,
            sin,
            attention_mask=bad_mask,
        )

    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError for mismatched attention_mask length."
    )