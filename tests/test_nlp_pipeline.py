import sys
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import torch

if "whisperx" not in sys.modules:
    mock_whisperx = MagicMock()
    sys.modules["whisperx"] = mock_whisperx

from src.common.contracts import check_text_sample, verify_text_tensors
from src.extractors.text.text_aligner import TokenAligner
from src.extractors.text.transcribe import Transcriber


@pytest.fixture
def transcribe_config(tmp_path):
    return {
        "model_size": "medium.en",
        "language": "en",
        "device": "cpu",
        "compute_type": "int8",
        "batch_size": 1,
        "beam_size": 1,
        "condition_on_previous_text": False,
        "no_speech_threshold": 0.6,
        "initial_prompt": None,
        "align_model": None,
        "cache_dir": str(tmp_path / "cache"),
    }


@patch("src.extractors.text.transcribe.whisperx.align")
@patch("src.extractors.text.transcribe.whisperx.load_audio")
@patch("src.extractors.text.transcribe.whisperx.load_align_model")
@patch("src.extractors.text.transcribe.whisperx.load_model")
def test_full_text_pipeline_mocked(
    mock_load_asr,
    mock_load_align,
    mock_load_audio,
    mock_align,
    transcribe_config,
):
    # 1. Setup WhisperX mocks for audio loading and alignment outputs
    mock_load_audio.return_value = np.zeros(16000 * 2, dtype=np.float32)  # 2-second dummy audio
    mock_asr = MagicMock()
    mock_asr.transcribe.return_value = {
        "segments": [{"text": "I was there", "start": 0.0, "end": 1.5}]
    }
    mock_load_asr.return_value = mock_asr

    mock_align_model = MagicMock()
    mock_load_align.return_value = (mock_align_model, MagicMock())

    mock_align.return_value = {
        "segments": [
            {
                "words": [
                    {"word": "I", "start": 0.1, "end": 0.3},
                    {"word": "was", "start": 0.4, "end": 0.7},
                    {"word": "there", "start": 0.8, "end": 1.2},
                ]
            }
        ]
    }

    # 2. Step A: Transcribe media clip and clean timestamps
    transcriber = Transcriber(transcribe_config)
    clip_data = transcriber.transcribe_clip("test_clip_001", "dummy_video.mp4")
    assert clip_data["status"] == "ok"
    assert len(clip_data["words"]) == 3

    # 3. Step B: Align word tokens and compute frame spans
    aligner = TokenAligner()
    aligned = aligner.align_clip(clip_data["words"], clip_fps=25.0)

    # 4. Step C: Generate dummy DeBERTa feature embedding tensor [56, 768]
    dummy_text_feats = torch.randn(56, 768, dtype=torch.float32)

    sample = {
        "input_ids": aligned["input_ids"],
        "text_feats": dummy_text_feats,
        "text_mask": aligned["text_mask"],
        "alignment_indices": aligned["alignment_indices"],
    }

    # 5. Step D: Validate final sample against system contracts
    check_text_sample(sample)
    verify_text_tensors(sample["text_feats"], sample["alignment_indices"])