import math
import torch
import pytest

from src.dataset.dummy_loader import (
    DummyMultimodalDataset,
    collate_multimodal_batch,
    get_dummy_dataloader,
    DEFAULT_D_VIS,
    DEFAULT_D_AUD,
    DEFAULT_D_TXT,
)
from src.models.fusion_net import MultimodalFusionNet
from src.evaluation.losses import MultimodalDeceptionLoss
from src.evaluation.metrics import DeceptionEvaluator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _grad_norm(module: torch.nn.Module) -> float:
    total_sq = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total_sq += float(p.grad.detach().pow(2).sum().item())
    return math.sqrt(total_sq)


def test_overfit_sanity_check_gate():
    """
    Section 5.3 Pre-Training Overfit Sanity Check:
      - 8 balanced clips (4 truthful, 4 deceptive)
      - Dropout, weight decay, and label smoothing disabled
      - Trained with AdamW for 80 epochs
      - Asserts loss < 0.05, accuracy == 100%, and non-zero gradient norms
        across all active encoders and fusion blocks.
    """
    torch.manual_seed(42)

    # Build a deterministic 8-clip balanced subset
    raw_ds = DummyMultimodalDataset(num_samples=8, allow_empty_modalities=False)
    samples = []
    for idx in range(8):
        item = raw_ds[idx]
        item["label"] = torch.tensor(idx % 2, dtype=torch.long)
        samples.append(item)
    batch = collate_multimodal_batch(samples).to(DEVICE)

    model = MultimodalFusionNet(
        d_vis=DEFAULT_D_VIS,
        d_aud=DEFAULT_D_AUD,
        d_txt=DEFAULT_D_TXT,
        d_model=128,
        p_drop=0.0,               # Disable modality dropout
        use_softmax_gate=True,
        use_interaction=True,
        use_span_qkv=True,
        classifier_dropout=0.0,   # Disable classifier dropout
        conflict_detach_inputs=True,
        conflict_validated=False,
    ).to(DEVICE)

    criterion = MultimodalDeceptionLoss(
        lambda_aux=0.3,
        lambda_conflict=0.2,
        label_smoothing=0.0,      # Disable smoothing so CE loss can approach 0
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)

    model.train()
    final_loss = float("inf")

    for epoch in range(80):
        optimizer.zero_grad(set_to_none=True)
        out = model(batch)
        losses = criterion(out, batch.label)
        losses.total_loss.backward()

        # Verify non-zero gradient norms across all active encoders and blocks
        if epoch == 1:
            assert _grad_norm(model.encoder_bank.vis_encoder) > 0.0
            assert _grad_norm(model.encoder_bank.aud_encoder) > 0.0
            assert _grad_norm(model.encoder_bank.txt_encoder) > 0.0
            assert _grad_norm(model.span_interaction) > 0.0
            assert _grad_norm(model.residual_gate) > 0.0
            assert _grad_norm(model.classifier) > 0.0
            assert _grad_norm(model.conflict_hook) > 0.0

        optimizer.step()
        final_loss = float(losses.total_loss.item())

    # Evaluate accuracy on the memorized 8-clip subset
    model.eval()
    with torch.no_grad():
        eval_out = model(batch)
        preds = (torch.sigmoid(eval_out.logits) >= 0.5).long()
        train_acc = float((preds == batch.label).float().mean().item())

    assert final_loss < 0.05, f"Expected overfit loss < 0.05, got {final_loss:.4f}"
    assert train_acc == 1.0, f"Expected 100% overfit accuracy, got {train_acc * 100:.1f}%"


def _build_multi_window_batches(num_clips: int, windows_per_clip: int, batch_size: int):
    """Creates deterministic multi-window clips where windows of a clip share subject & label."""
    total_windows = num_clips * windows_per_clip
    raw_ds = DummyMultimodalDataset(num_samples=total_windows, allow_empty_modalities=True)

    items = []
    for c in range(num_clips):
        sub_id = f"sub_{(c % 6) + 1:03d}"
        clip_id = f"clip_{c:03d}"
        clip_label = c % 2
        for w in range(windows_per_clip):
            sample = raw_ds[c * windows_per_clip + w]
            sample["subject_id"] = sub_id
            sample["clip_id"] = clip_id
            sample["label"] = torch.tensor(clip_label, dtype=torch.long)
            items.append(sample)

    batches = []
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        batch = collate_multimodal_batch(chunk)
        clip_ids = [x["clip_id"] for x in chunk]
        batches.append((batch, clip_ids))
    return batches


def test_full_train_val_test_pipeline():
    """
    End-to-end integration test across Blocks 1–10:
      1. Training steps with modality dropout, missing modalities, aux losses, and conflict loss.
      2. Validation pass with multi-window clip aggregation and threshold tuning.
      3. Test pass with locked validation threshold, ECE calibration, and subject bootstrap CIs.
    """
    torch.manual_seed(123)

    train_loader = get_dummy_dataloader(batch_size=8, num_samples=24, allow_empty_modalities=True)
    val_batches = _build_multi_window_batches(num_clips=12, windows_per_clip=2, batch_size=8)
    test_batches = _build_multi_window_batches(num_clips=12, windows_per_clip=2, batch_size=8)

    model = MultimodalFusionNet(
        d_vis=DEFAULT_D_VIS,
        d_aud=DEFAULT_D_AUD,
        d_txt=DEFAULT_D_TXT,
        d_model=64,
        p_drop=0.2,
        use_softmax_gate=True,
        use_interaction=True,
        classifier_dropout=0.2,
        conflict_detach_inputs=True,
        conflict_validated=False,
    ).to(DEVICE)

    criterion = MultimodalDeceptionLoss(
        lambda_aux=0.3,
        lambda_conflict=0.2,
        label_smoothing=0.05,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    # 1. Train Pass
    model.train()
    for batch in train_loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        out = model(batch)
        losses = criterion(out, batch.label)
        assert not torch.isnan(losses.total_loss)
        losses.total_loss.backward()
        optimizer.step()

    # 2. Validation Pass (multi-window clip aggregation + validation-only threshold tuning)
    model.eval()
    val_evaluator = DeceptionEvaluator(aggregation_mode="prob_mean", n_bootstraps=100)
    with torch.no_grad():
        for batch, clip_ids in val_batches:
            batch = batch.to(DEVICE)
            out = model(batch)
            val_evaluator.update(
                logits=out.logits,
                labels=batch.label,
                subject_ids=batch.subject_id,
                clip_ids=clip_ids,
            )

    val_report = val_evaluator.compute(tune_threshold=True, threshold_criterion="macro_f1")
    assert 0.0 < val_report.threshold < 1.0
    assert val_report.num_windows == 24
    assert val_report.num_clips == 12

    # 3. Test Pass (locked validation threshold + subject-level bootstrap CIs)
    test_evaluator = DeceptionEvaluator(aggregation_mode="prob_mean", n_bootstraps=100)
    with torch.no_grad():
        for batch_idx, (batch, clip_ids) in enumerate(test_batches):
            batch = batch.to(DEVICE)
            out = model(batch)
            test_evaluator.update(
                logits=out.logits,
                labels=batch.label,
                subject_ids=batch.subject_id,
                clip_ids=clip_ids,
            )

            # Verify Block 10 sync_gap runs cleanly on held-out batches
            if batch_idx == 0:
                enc = model.encoder_bank(batch)
                gap = model.conflict_hook.sync_gap(enc, out.natural_mask, shift_frames=15)
                assert isinstance(gap, float)

    test_report = test_evaluator.compute(
        threshold=val_report.threshold,
        run_bootstrap=True,
    )
    report_dict = test_report.to_dict()

    assert test_report.threshold == val_report.threshold
    assert test_report.num_clips == 12
    assert 0.0 <= test_report.ece <= 1.0
    assert "auc_roc_ci_lower" in report_dict
    assert "macro_f1_ci_upper" in report_dict