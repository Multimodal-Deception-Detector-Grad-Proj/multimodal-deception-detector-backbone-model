import pytest
import torch
from src.common.constants import L_TXT, TEXT_ENCODER
from src.extractors.text.text_aligner import TokenAligner

@pytest.fixture(scope="module")
def aligner():
    return TokenAligner(model_name=TEXT_ENCODER, max_len=L_TXT)

def test_aligner_tensor_shapes_and_types(aligner):
    sample_words = [
        {"word": "I", "start": 0.1, "end": 0.3},
        {"word": "testified", "start": 0.4, "end": 0.9},
    ]
    out = aligner.align_clip(sample_words, clip_fps=25.0)
    
    assert out["input_ids"].shape == (L_TXT,)
    assert out["input_ids"].dtype == torch.long
    assert out["text_mask"].shape == (L_TXT,)
    assert out["text_mask"].dtype == torch.bool
    assert out["alignment_indices"].shape == (L_TXT, 2)
    assert out["alignment_indices"].dtype == torch.long

def test_aligner_handles_none_timestamps(aligner):
    # Tests tolerance for unaligned words from transcribe.py
    words_with_nones = [
        {"word": "In", "start": 0.0, "end": 0.2},
        {"word": "2019", "start": None, "end": None},
    ]
    # Should complete without raising TypeError
    out = aligner.align_clip(words_with_nones, clip_fps=25.0)
    assert out["input_ids"].shape == (L_TXT,)

def test_aligner_empty_speech(aligner):
    out = aligner.align_clip([], clip_fps=30.0)
    assert out["input_ids"].shape == (L_TXT,)
    assert out["text_mask"].sum().item() == 2  # Only [CLS] and [SEP] active