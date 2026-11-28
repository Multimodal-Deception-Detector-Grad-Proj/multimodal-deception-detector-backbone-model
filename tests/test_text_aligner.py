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
    
    
def test_aligner_indices_within_T(aligner):
    from src.common.constants import TARGET_T
    words = [{"word": "edge", "start": 7.5, "end": 8.0}]   # ceil(8.0*25) would be 200
    out = aligner.align_clip(words, clip_fps=25.0)
    assert out["alignment_indices"].max().item() <= TARGET_T - 1


def test_aligner_drops_words_after_window(aligner):
    out = aligner.align_clip([{"word": "late", "start": 9.0, "end": 9.5}], clip_fps=25.0)
    assert out["text_mask"].sum().item() == 2          # only CLS + SEP


def test_aligner_untimed_word_not_at_frame_zero(aligner):
    words = [
        {"word": "In", "start": 1.0, "end": 1.2},
        {"word": "2019", "start": None, "end": None},
        {"word": "was", "start": 2.0, "end": 2.2},
    ]
    out = aligner.align_clip(words, clip_fps=25.0)
    assert out["alignment_indices"][2, 0].item() > 0   # first token of "2019"


def test_aligner_padding_and_truncation(aligner):
    words = [{"word": f"word{i}", "start": i * 0.05, "end": i * 0.05 + 0.04} for i in range(200)]
    out = aligner.align_clip(words, clip_fps=25.0)
    assert out["text_mask"].all()
    assert out["input_ids"][-1].item() == aligner.tokenizer.sep_token_id

    short = aligner.align_clip([{"word": "hi", "start": 0.0, "end": 0.2}], clip_fps=25.0)
    pad = ~short["text_mask"]
    assert (short["alignment_indices"][pad] == 0).all()