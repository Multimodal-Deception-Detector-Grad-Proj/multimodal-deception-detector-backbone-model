"""Guideline 5.3: the audio model must overfit 8 real clips (4 lie / 4 truth). Skipped if no extracted data."""
import glob, os
import torch
import pytest
from src.models.audio.audio_cnn import AudioClassifier

ROOT = os.environ.get("AUDIO_PT_DIR", "data/processed/audio/train")


def test_overfit_8_clips():
    files = sorted(glob.glob(f"{ROOT}/*.pt"))
    samples = [torch.load(f) for f in files]
    lies = [s for s in samples if float(s["label"]) == 1][:4]
    truths = [s for s in samples if float(s["label"]) == 0][:4]
    if len(lies) < 4 or len(truths) < 4:
        pytest.skip("no extracted audio .pt files")
    batch = lies + truths
    x = torch.stack([s["audio_feats"] for s in batch]).float()
    y = torch.tensor([float(s["label"]) for s in batch])
    torch.manual_seed(0)
    model = AudioClassifier(p_drop=0.0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(300):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.05, f"cannot overfit 8 clips (loss {loss.item():.3f})"