import torch
import pytest

from src.dataset.dummy_loader import (
    get_dummy_dataloader,
    DEFAULT_T,
    DEFAULT_L_TXT,
    DEFAULT_D_VIS,
    DEFAULT_D_AUD,
    DEFAULT_D_TXT,
)
from src.models.encoders import (
    Masked3DPooling,
    TemporalConvEncoder,
    TextEncoder,
    TemporalEncoderBank,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_pooling_isolated():
    """Tests pooling.py in isolation, including an all-False empty mask."""
    B, T, D = 4, 90, 128
    pool = Masked3DPooling(d_model=D).to(DEVICE)

    x = torch.randn(B, T, D, device=DEVICE, requires_grad=True)
    mask = torch.ones(B, T, dtype=torch.bool, device=DEVICE)
    mask[0, :] = False  # Simulate 100% face dropout on sample 0
    mask[1, 45:] = False  # Simulate half-padded sequence on sample 1

    out = pool(x, mask)
    loss = out.sum()
    loss.backward()

    assert out.shape == (B, D)
    assert not torch.isnan(out).any(), "NaN in pooled output!"
    assert not torch.isnan(x.grad).any(), "NaN in backward gradients!"
    assert (out[0] == 0).all(), "Fully masked sample must be strictly zero!"


def test_conv_encoder_isolated():
    """Tests conv_encoder.py in isolation."""
    B, T, D_in, D_model = 4, 90, 35, 128
    enc = TemporalConvEncoder(d_in=D_in, d_model=D_model, kernel_size=3).to(DEVICE)

    x = torch.randn(B, T, D_in, device=DEVICE)
    mask = torch.ones(B, T, dtype=torch.bool, device=DEVICE)
    mask[:, 60:] = False  # Last 30 frames are padding

    out = enc(x, mask)

    assert out.shape == (B, T, D_model)
    assert not torch.isnan(out).any()
    assert (out[:, 60:, :] == 0).all(), "Padded frames leaked non-zero values!"


def test_text_encoder_isolated():
    """Tests text_encoder.py in isolation."""
    B, L, D_txt, D_model = 4, 15, 768, 128
    enc = TextEncoder(d_in=D_txt, d_model=D_model).to(DEVICE)

    x = torch.randn(B, L, D_txt, device=DEVICE)
    mask = torch.ones(B, L, dtype=torch.bool, device=DEVICE)
    mask[:, 10:] = False

    out = enc(x, mask)

    assert out.shape == (B, L, D_model)
    assert (out[:, 10:, :] == 0).all(), "Padded tokens leaked non-zero values!"


def test_encoder_bank_with_dummy_loader():
    """Tests bank.py end-to-end with dummy_loader.py."""
    loader = get_dummy_dataloader(batch_size=8, num_samples=24, allow_empty_modalities=True)
    batch = next(iter(loader)).to(DEVICE)

    bank = TemporalEncoderBank(
        d_vis=DEFAULT_D_VIS,
        d_aud=DEFAULT_D_AUD,
        d_txt=DEFAULT_D_TXT,
        d_model=128,
    ).to(DEVICE)

    out = bank(batch)

    assert out.H_v.shape == (8, DEFAULT_T, 128)
    assert out.H_a.shape == (8, DEFAULT_T, 128)
    assert out.H_t.shape == (8, DEFAULT_L_TXT, 128)
    assert out.P_v_raw.shape == (8, 128)
    assert not torch.isnan(out.P_v_raw).any()