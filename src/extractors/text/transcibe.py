"""
Output per clip: {"clip_id", "status", "duration", "words": [{"word", "start", "end"}]}
Words WhisperX cannot align (e.g. "2019", "$5") get start = end = None.
"""
import json
import logging
import math
from pathlib import Path
import torch
import whisperx

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
SAMPLE_RATE = 16000  # whisperx.load_audio always resamples to 16 kHz mono


def clean_words(raw_words: list, duration: float) -> list:
    """Drop empty words and repair bad timestamps (pure function, easy to unit test)."""
    out, last_end = [], 0.0
    for w in raw_words:
        text = (w.get("word") or "").strip()
        if not text:
            continue
        start, end = w.get("start"), w.get("end")
        if start is None or end is None or not (math.isfinite(start) and math.isfinite(end)):
            out.append({"word": text, "start": None, "end": None})
            continue
        start = max(min(start, duration), last_end)  # clamp to clip, no overlaps
        end = max(min(end, duration), start)
        last_end = end
        out.append({"word": text, "start": round(start, 3), "end": round(end, 3)})
    return out


class Transcriber:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = cfg["device"]
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = cfg["compute_type"] if self.device == "cuda" else "int8"  # no fp16 on CPU

        self.asr = whisperx.load_model(
            cfg["model_size"],
            self.device,
            compute_type=compute_type,
            language=cfg["language"],
            asr_options={
                "beam_size": cfg["beam_size"],
                "condition_on_previous_text": cfg["condition_on_previous_text"],
                "no_speech_threshold": cfg["no_speech_threshold"],
                "initial_prompt": cfg["initial_prompt"],
            },
        )
        self.align_model, self.align_meta = whisperx.load_align_model(
            language_code=cfg["language"], device=self.device, model_name=cfg["align_model"]
        )
        self.cache_dir = ROOT / cfg["cache_dir"]

    def _read_cache(self, path: Path):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            return None
        same = (
            data.get("model_size") == self.cfg["model_size"]
            and data.get("align_model") == self.cfg["align_model"]
            and data.get("language") == self.cfg["language"]
        )
        return data if same and data.get("status") in ("ok", "no_speech") else None

    def transcribe_clip(self, clip_id: str, media_path) -> dict:
        """
        status "ok"         words found
               "no_speech"  valid audio but no words (cached; downstream must handle it)
               "failed"     see "error"; not cached, so the next run retries it
        """
        cache_path = self.cache_dir / f"{clip_id}.json"
        cached = self._read_cache(cache_path)
        if cached is not None:
            return cached

        try:
            audio = whisperx.load_audio(str(media_path))
            duration = len(audio) / SAMPLE_RATE
            result = self.asr.transcribe(
                audio, batch_size=self.cfg["batch_size"], language=self.cfg["language"]
            )

            raw_words = []
            if result["segments"]:
                aligned = whisperx.align(
                    result["segments"], self.align_model, self.align_meta, audio, self.device,
                    return_char_alignments=False,
                )
                for seg in aligned["segments"]:
                    raw_words.extend(seg.get("words", []))
            words = clean_words(raw_words, duration)
        except Exception as exc:
            logger.error("[%s] transcription failed: %s: %s", clip_id, type(exc).__name__, exc)
            return {"clip_id": clip_id, "status": "failed", "error": str(exc), "words": []}

        out = {
            "clip_id": clip_id,
            "status": "ok" if words else "no_speech",
            "duration": duration,
            "words": words,
            "model_size": self.cfg["model_size"],
            "align_model": self.cfg["align_model"],
            "language": self.cfg["language"],
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        return out