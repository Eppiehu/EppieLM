import torch

from eppielm.models.configuration_eppielm import EppieLMConfig
from eppielm.models.modeling_eppielm import (
    EppieLMModel,
    EppieLMForCausalLM,
)


def tiny_config(
    attention_impl: str = "eager",
) -> EppieLMConfig:
    # Keep the architecture small enough for fast CI while preserving the
    # same structural choices as the real EppieLM-150M configuration.
    return EppieLMConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=352,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        attention_impl=attention_impl,
        use_cache=True,
        tie_word_embeddings=True,
    )


def test_base_model_output_shape():
    config = tiny_config()

    model = EppieLMModel(config)
    model.eval()

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (2, 16),
    )

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            use_cache=False,
            return_dict=True,
        )

    assert outputs.last_hidden_state.shape == (
        2,
        16,
        config.hidden_size,
    )

    assert torch.isfinite(
        outputs.last_hidden_state
    ).all()


def test_causal_lm_logits_shape():
    config = tiny_config()

    model = EppieLMForCausalLM(config)
    model.eval()

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (2, 16),
    )

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            use_cache=False,
            return_dict=True,
        )

    assert outputs.logits.shape == (
        2,
        16,
        config.vocab_size,
    )

    assert torch.isfinite(
        outputs.logits
    ).all()


def test_causal_lm_loss_is_finite():
    config = tiny_config()

    model = EppieLMForCausalLM(config)
    model.eval()

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (2, 16),
    )

    labels = input_ids.clone()

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )

    assert outputs.loss is not None
    assert outputs.loss.ndim == 0
    assert torch.isfinite(outputs.loss)


def test_ignore_index_in_language_model_loss():
    config = tiny_config()

    model = EppieLMForCausalLM(config)
    model.eval()

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (2, 16),
    )

    labels = input_ids.clone()
    labels[:, :4] = -100

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )

    assert outputs.loss is not None
    assert torch.isfinite(outputs.loss)


def test_weight_tying():
    config = tiny_config()

    model = EppieLMForCausalLM(config)

    assert (
        model.model.embed_tokens.weight.data_ptr()
        ==
        model.lm_head.weight.data_ptr()
    )


def test_model_cache_has_one_entry_per_layer():
    config = tiny_config()

    model = EppieLMForCausalLM(config)
    model.eval()

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (2, 8),
    )

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
        )

    assert outputs.past_key_values is not None

    assert len(outputs.past_key_values) == (
        config.num_hidden_layers
    )

    expected_shape = (
        2,
        config.num_key_value_heads,
        8,
        config.head_dim,
    )

    for key_states, value_states in (
        outputs.past_key_values
    ):
        assert key_states.shape == expected_shape
        assert value_states.shape == expected_shape


def test_cached_decoding_matches_full_sequence():
    config = tiny_config(
        attention_impl="eager",
    )

    model = EppieLMForCausalLM(config)
    model.eval()

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (2, 8),
    )

    with torch.no_grad():
        full_outputs = model(
            input_ids=input_ids,
            use_cache=False,
            return_dict=True,
        )

        prefix_outputs = model(
            input_ids=input_ids[:, :7],
            use_cache=True,
            return_dict=True,
        )

        cached_outputs = model(
            input_ids=input_ids[:, 7:],
            past_key_values=(
                prefix_outputs.past_key_values
            ),
            use_cache=True,
            return_dict=True,
        )

    assert cached_outputs.logits.shape == (
        2,
        1,
        config.vocab_size,
    )

    assert torch.allclose(
        full_outputs.logits[:, -1:],
        cached_outputs.logits,
        atol=1e-4,
        rtol=1e-4,
    )


def test_sdpa_model_forward():
    config = tiny_config(
        attention_impl="sdpa",
    )

    model = EppieLMForCausalLM(config)
    model.eval()

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (2, 16),
    )

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            use_cache=False,
            return_dict=True,
        )

    assert outputs.logits.shape == (
        2,
        16,
        config.vocab_size,
    )

    assert torch.isfinite(
        outputs.logits
    ).all()


def test_eager_and_sdpa_models_match_with_same_weights():
    eager_config = tiny_config(
        attention_impl="eager",
    )

    sdpa_config = tiny_config(
        attention_impl="sdpa",
    )

    eager_model = EppieLMForCausalLM(
        eager_config
    )

    sdpa_model = EppieLMForCausalLM(
        sdpa_config
    )

    sdpa_model.load_state_dict(
        eager_model.state_dict()
    )

    eager_model.eval()
    sdpa_model.eval()

    input_ids = torch.randint(
        0,
        eager_config.vocab_size,
        (2, 12),
    )

    with torch.no_grad():
        eager_outputs = eager_model(
            input_ids=input_ids,
            use_cache=False,
            return_dict=True,
        )

        sdpa_outputs = sdpa_model(
            input_ids=input_ids,
            use_cache=False,
            return_dict=True,
        )

    assert torch.allclose(
        eager_outputs.logits,
        sdpa_outputs.logits,
        atol=1e-4,
        rtol=1e-4,
    )


def test_logits_to_keep():
    config = tiny_config()

    model = EppieLMForCausalLM(config)
    model.eval()

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (2, 16),
    )

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            logits_to_keep=1,
            use_cache=False,
            return_dict=True,
        )

    assert outputs.logits.shape == (
        2,
        1,
        config.vocab_size,
    )


def test_default_eppielm_parameter_count():
    config = EppieLMConfig()

    model = EppieLMForCausalLM(config)

    params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"\nEppieLM parameters: "
        f"{params / 1_000_000:.2f}M"
    )

    # "150M" is a scale label rather than a promise of an exact count.
    assert 140_000_000 <= params <= 160_000_000