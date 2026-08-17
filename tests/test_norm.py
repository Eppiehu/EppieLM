import torch

from eppielm.models.norm import EppieRMSNorm


def test_rms_norm_preserves_shape():
    norm = EppieRMSNorm(hidden_size=768)

    x = torch.randn(2, 16, 768)
    y = norm(x)

    assert y.shape == x.shape


def test_rms_norm_output_is_finite():
    norm = EppieRMSNorm(hidden_size=768)

    x = torch.randn(2, 16, 768)
    y = norm(x)

    assert torch.isfinite(y).all()


def test_rms_norm_has_unit_rms_at_initialization():
    norm = EppieRMSNorm(hidden_size=768)

    x = torch.randn(4, 8, 768)
    y = norm(x)

    rms = y.float().pow(2).mean(dim=-1).sqrt()

    assert torch.allclose(
        rms,
        torch.ones_like(rms),
        atol=1e-4,
        rtol=1e-4,
    )