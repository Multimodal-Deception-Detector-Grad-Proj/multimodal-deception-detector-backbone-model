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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("use_interaction", [True, False])
@pytest.mark.parametrize("use_softmax_gate", [True, False])
@pytest.mark.parametrize("batch_size", [1, 8])
def test_fusion_net_forward_and_backward(
    use_interaction: bool,
    use_softmax_gate: bool,
    batch_size: int,
):
    loader = get_dummy_dataloader(
        batch_size=batch_size,
        num_samples=max(8, batch_size),
        allow_empty_modalities=True,
    )
    batch = next(iter(loader)).to(DEVICE)

    model = MultimodalFusionNet(
        d_vis=DEFAULT_D_VIS,
        d_aud=DEFAULT_D_AUD,
        d_txt=DEFAULT_D_TXT,
        d_model=128,
        p_drop=0.3,
        use_softmax_gate=use_softmax_gate,
        use_interaction=use_interaction,
        classifier_dropout=0.0,
    ).to(DEVICE)
    model.train()

    out = model(batch)

    # 1. Output shape assertions (including B=1 edge case)
    assert out.logits.shape == (batch_size,)
    assert out.Z.shape == (batch_size, 128)
    assert out.P.shape == (batch_size, 128)
    assert out.S.shape == (batch_size, 128)
    assert out.natural_mask.shape == (batch_size, 3)
    assert out.active_mask.shape == (batch_size, 3)
    assert out.modality_weights.shape == (batch_size, 3)
    for mod in ("visual", "audio", "text"):
        assert out.aux_logits[mod].shape == (batch_size,)

    # 2. Mask invariants: active_mask must never activate a naturally missing modality
    assert not (out.active_mask & ~out.natural_mask).any(), (
        "active_mask turned on a modality that was naturally missing!"
    )

    # 3. Residual fusion invariants
    if not use_interaction:
        assert out.gate_g is None
        assert torch.equal(out.Z, out.P), "Z must equal P when interaction flag is off!"
    else:
        assert out.gate_g is not None and out.gate_g.shape == (batch_size, 128)
        # At step 0, Block 5 out_proj is zero-initialized -> S == 0 -> Z == P
        assert torch.allclose(out.Z, out.P, atol=1e-6), "Z must equal P at initialization!"

    # 4. Block 10 conflict stub check
    assert out.conflict.validated is False

    # 5. Joint backward pass (main + aux heads)
    total_loss = out.logits.sum() + sum(v.sum() for v in out.aux_logits.values())
    total_loss.backward()

    assert not torch.isnan(out.logits).any()
    assert model.classifier[1].weight.grad is not None
    assert model.aux_head_v.weight.grad is not None
    assert model.aux_head_a.weight.grad is not None
    assert model.aux_head_t.weight.grad is not None

    if use_interaction:
        assert model.span_interaction.out_proj.weight.grad is not None
        assert model.span_interaction.out_proj.weight.grad.abs().sum() > 0


def test_fusion_net_natural_vs_active_mask_separation():
    """
    Verifies that natural_mask tracks true input availability (for Block 8 aux losses)
    independently of stochastic modality dropout in active_mask (for Blocks 4 & 5).
    """
    B = 4
    model = MultimodalFusionNet(
        d_vis=DEFAULT_D_VIS,
        d_aud=DEFAULT_D_AUD,
        d_txt=DEFAULT_D_TXT,
        d_model=64,
        p_drop=0.99,  # Force aggressive modality dropout during train()
        use_interaction=True,
    ).to(DEVICE)
    model.train()

    batch = {
        "visual_feats": torch.randn(B, DEFAULT_T, DEFAULT_D_VIS, device=DEVICE),
        "visual_mask": torch.ones(B, DEFAULT_T, dtype=torch.bool, device=DEVICE),
        "audio_feats": torch.randn(B, DEFAULT_T, DEFAULT_D_AUD, device=DEVICE),
        "audio_mask": torch.ones(B, DEFAULT_T, dtype=torch.bool, device=DEVICE),
        "text_feats": torch.randn(B, DEFAULT_L_TXT, DEFAULT_D_TXT, device=DEVICE),
        "text_mask": torch.ones(B, DEFAULT_L_TXT, dtype=torch.bool, device=DEVICE),
        "alignment_indices": torch.zeros(B, DEFAULT_L_TXT, 2, dtype=torch.long, device=DEVICE),
        "label": torch.randint(0, 2, (B,), device=DEVICE),
    }

    # Simulate natural visual missingness (e.g., face occlusion) on sample 0 only
    batch["visual_mask"][0, :] = False
    batch["visual_feats"][0, :] = 0.0

    out = model(batch)

    # Sample 0 has V=False, A=True, T=True naturally; samples 1..3 have all True
    expected_natural = torch.tensor(
        [
            [False, True, True],
            [True, True, True],
            [True, True, True],
            [True, True, True],
        ],
        device=DEVICE,
    )
    assert torch.equal(out.natural_mask, expected_natural)

    # Despite p_drop=0.99, every sample must keep >= 1 active modality in active_mask
    assert out.active_mask.any(dim=-1).all()
    assert not out.active_mask[0, 0].item()