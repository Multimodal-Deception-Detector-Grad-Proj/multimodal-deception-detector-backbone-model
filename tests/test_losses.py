import math
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
from src.models.fusion_net import MultimodalFusionNet
from src.evaluation.losses import MultimodalDeceptionLoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_loss_end_to_end_with_dummy_loader():
    loader = get_dummy_dataloader(batch_size=8, num_samples=16, allow_empty_modalities=True)
    batch = next(iter(loader)).to(DEVICE)

    model = MultimodalFusionNet(
        d_vis=DEFAULT_D_VIS,
        d_aud=DEFAULT_D_AUD,
        d_txt=DEFAULT_D_TXT,
        d_model=64,
    ).to(DEVICE)
    criterion = MultimodalDeceptionLoss(lambda_aux=0.3, label_smoothing=0.05).to(DEVICE)

    out = model(batch)
    losses = criterion(out, batch.label)
    losses.total_loss.backward()

    assert losses.total_loss.ndim == 0
    assert not torch.isnan(losses.total_loss)
    assert model.classifier[1].weight.grad is not None
    assert model.aux_head_v.weight.grad is not None


def test_missing_modality_produces_zero_aux_bias_gradient():
    """
    If visual is naturally missing for ALL samples in a batch, aux_head_v
    must receive strictly zero gradient (no bias memorization bug).
    """
    B = 4
    model = MultimodalFusionNet(
        d_vis=DEFAULT_D_VIS,
        d_aud=DEFAULT_D_AUD,
        d_txt=DEFAULT_D_TXT,
        d_model=64,
        p_drop=0.5,
    ).to(DEVICE)
    criterion = MultimodalDeceptionLoss(lambda_aux=0.5, label_smoothing=0.05).to(DEVICE)

    batch = {
        "visual_feats": torch.zeros(B, DEFAULT_T, DEFAULT_D_VIS, device=DEVICE),
        "visual_mask": torch.zeros(B, DEFAULT_T, dtype=torch.bool, device=DEVICE),  # All missing
        "audio_feats": torch.randn(B, DEFAULT_T, DEFAULT_D_AUD, device=DEVICE),
        "audio_mask": torch.ones(B, DEFAULT_T, dtype=torch.bool, device=DEVICE),
        "text_feats": torch.randn(B, DEFAULT_L_TXT, DEFAULT_D_TXT, device=DEVICE),
        "text_mask": torch.ones(B, DEFAULT_L_TXT, dtype=torch.bool, device=DEVICE),
        "alignment_indices": torch.zeros(B, DEFAULT_L_TXT, 2, dtype=torch.long, device=DEVICE),
        "label": torch.ones(B, dtype=torch.long, device=DEVICE),  # All positive class
    }

    out = model(batch)
    losses = criterion(out, batch["label"])
    losses.total_loss.backward()

    assert losses.aux_visual.item() == 0.0
    assert model.aux_head_v.bias.grad is not None
    assert model.aux_head_v.bias.grad.item() == 0.0
    assert model.aux_head_v.weight.grad.abs().sum().item() == 0.0
    assert model.aux_head_a.bias.grad.abs().item() > 0.0


def test_label_smoothing_floor():
    """Verifies exact binary label smoothing floor when logits match smoothed targets."""
    eps = 0.1
    criterion = MultimodalDeceptionLoss(lambda_aux=0.0, label_smoothing=eps)

    # Optimal logits for smoothed targets: p = 1 - 0.5*eps for y=1, p = 0.5*eps for y=0
    p1 = 1.0 - 0.5 * eps
    p0 = 0.5 * eps
    opt_logits = torch.tensor([math.log(p1 / (1.0 - p1)), math.log(p0 / (1.0 - p0))])
    labels = torch.tensor([1, 0])

    # Minimal mock matching FusionNetOutput fields used by criterion
    from types import SimpleNamespace
    mock_out = SimpleNamespace(
        logits=opt_logits,
        aux_logits={"visual": opt_logits, "audio": opt_logits, "text": opt_logits},
        natural_mask=torch.ones(2, 3, dtype=torch.bool),
    )

    loss_val = criterion(mock_out, labels).main_loss.item()
    expected_entropy = -(p1 * math.log(p1) + (1.0 - p1) * math.log(1.0 - p1))
    assert math.isclose(loss_val, expected_entropy, rel_tol=1e-5)