import torch

from eppielm.models.mlp import EppieMLP


def test_mlp_preserves_outer_shape():
    mlp = EppieMLP(
        hidden_size=768,
        intermediate_size=1792,
    )

    x = torch.randn(2, 16, 768)
    y = mlp(x)

    assert y.shape == x.shape


def test_mlp_output_is_finite():
    mlp = EppieMLP(
        hidden_size=768,
        intermediate_size=1792,
    )

    x = torch.randn(2, 16, 768)
    y = mlp(x)

    assert torch.isfinite(y).all()


def test_mlp_has_expected_parameter_shapes():
    mlp = EppieMLP(
        hidden_size=768,
        intermediate_size=1792,
    )

    assert mlp.gate_proj.weight.shape == (1792, 768)
    assert mlp.up_proj.weight.shape == (1792, 768)
    assert mlp.down_proj.weight.shape == (768, 1792)