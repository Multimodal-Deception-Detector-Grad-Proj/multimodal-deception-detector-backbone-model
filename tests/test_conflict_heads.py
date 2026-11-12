import torch
from src.dataset.dummy_loader import (
    get_dummy_dataloader,
    DEFAULT_D_VIS,
    DEFAULT_D_AUD,
    DEFAULT_D_TXT,
)
from src.models.fusion_net import MultimodalFusionNet
from src.evaluation.losses import MultimodalDeceptionLoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_conflict_head_detach_and_sync_gap():
    loader = get_dummy_dataloader(batch_size=4, num_samples=8, allow_empty_modalities=True)
    batch = next(iter(loader)).to(DEVICE)

    model = MultimodalFusionNet(
        d_vis=DEFAULT_D_VIS,
        d_aud=DEFAULT_D_AUD,
        d_txt=DEFAULT_D_TXT,
        d_model=64,
        conflict_detach_inputs=True,
        conflict_validated=False,
    ).to(DEVICE)
    criterion = MultimodalDeceptionLoss(lambda_aux=0.0, lambda_conflict=0.5).to(DEVICE)

    model.train()
    out = model(batch)
    losses = criterion(out, batch.label)

    # Verify LossOutput combined main_loss + lambda_conflict * align_loss properly
    expected_total = losses.main_loss + 0.5 * out.conflict.align_loss.detach()
    assert torch.allclose(losses.total_loss.detach(), expected_total, atol=1e-6)

    # Backprop ONLY align_loss to verify detach_inputs=True isolates encoders
    out.conflict.align_loss.backward()

    assert out.conflict.validated is False
    assert out.conflict.c_trace.shape == (4, batch.visual_feats.shape[1])
    assert out.conflict.word_conflict.shape == (4, batch.text_feats.shape[1])
    assert model.conflict_hook.proj_v[0].weight.grad is not None
    assert model.encoder_bank.vis_encoder.input_proj.weight.grad is None

    # Verify sync_gap runs cleanly in eval mode
    model.eval()
    enc = model.encoder_bank(batch)
    natural_mask = torch.stack(
        [batch.visual_mask.any(-1), batch.audio_mask.any(-1), batch.text_mask.any(-1)],
        dim=-1,
    )
    gap = model.conflict_hook.sync_gap(enc, natural_mask, shift_frames=10)
    assert isinstance(gap, float)