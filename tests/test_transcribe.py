import sys
from unittest.mock import MagicMock

# mock whisperX to not install it
if "whisperx" not in sys.modules:
    mock_whisperx = MagicMock()
    sys.modules["whisperx"] = mock_whisperx

import json
from unittest.mock import patch
import pytest

from src.extractors.text.transcribe import Transcriber, clean_words


def test_clean_words_sanitization():
    raw = [
        {"word": " Hello ", "start": 0.1, "end": 0.5},
        {"word": "", "start": 0.5, "end": 0.6},
        {"word": "2019", "start": None, "end": None},
        {"word": "world", "start": 0.4, "end": 2.5},  # Overlap repair test
    ]
    cleaned = clean_words(raw, duration=2.0)

    assert len(cleaned) == 3
    assert cleaned[0] == {"word": "Hello", "start": 0.1, "end": 0.5}
    assert cleaned[1] == {"word": "2019", "start": None, "end": None}
    assert cleaned[2] == {"word": "world", "start": 0.5, "end": 2.0}


@patch("src.extractors.text.transcribe.whisperx.load_align_model")
@patch("src.extractors.text.transcribe.whisperx.load_model")
def test_transcriber_caching(mock_load_asr, mock_load_align, tmp_path):
    mock_load_asr.return_value = MagicMock()
    mock_load_align.return_value = (MagicMock(), MagicMock())

    cfg = {
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

    transcriber = Transcriber(cfg)
    cache_file = transcriber.cache_dir / "clip_001.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    mock_payload = {
        "clip_id": "clip_001",
        "status": "ok",
        "duration": 2.5,
        "words": [{"word": "test", "start": 0.1, "end": 0.5}],
        "model_size": "medium.en",
        "align_model": None,
        "language": "en",
    }
    cache_file.write_text(json.dumps(mock_payload), encoding="utf-8")

    result = transcriber.transcribe_clip("clip_001", "dummy_path.mp4")
    assert result["status"] == "ok"
    assert result["words"][0]["word"] == "test"
    mock_load_asr.return_value.transcribe.assert_not_called()