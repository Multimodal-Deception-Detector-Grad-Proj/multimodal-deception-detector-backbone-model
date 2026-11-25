"""Audio extraction sub-pipeline.

    video clip --ffmpeg-->  16 kHz mono waveform (first TARGET_T/TARGET_FPS seconds, t = 0 is the container's t = 0)
               --> log-mel (AUDIO_N_MELS bins) + frame-level channels, one frame per video frame (hop = SR / FPS)
               --> normalised with TRAIN-ONLY statistics (configs/audio_normalization_stats.json)
               --> (TARGET_T, D_AUD) FloatTensor + label, saved as .pt

Channel layout (constants.py):  [ mel bins | valid | f0_semitone | voiced | loudness_db | hnr_db ]
Padding frames are all zeros, including `valid`.

Modality flag. Every sample carries `audio_present` (scalar bool tensor). It is True when the audio is usable for fusion.
It is False when the clip has no audio stream, is too short to analyse, or fails a quality check (non-finite values, too
few frames, near-silence, heavy clipping, and the optional noise / voicing checks). Unusable audio is saved as exact
zeros with `n_valid_frames = 0`, and `audio_issues` says why, so the fusion model can treat audio as missing. The
thresholds live in configs/base_config.yaml (audio.quality); `quality-report` shows how many clips each one would flag.

Usage (from the repository root):
    python -m src.extractors.audio.extract_audio fit-stats      --manifest data/splits/subject_isolated_splits.csv
    python -m src.extractors.audio.extract_audio quality-report --manifest data/splits/subject_isolated_splits.csv
    python -m src.extractors.audio.extract_audio extract        --manifest data/splits/subject_isolated_splits.csv

The manifest is a CSV with columns  clip_id, relative_path (video, relative to the repo root), split (train|val|test)
and optionally label (0/1 or truth/lie). If `label` is absent it is parsed from the clip id.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np

from src.common.constants import (
    AUDIO_CHANNEL_INDEX, AUDIO_CLIP_SAMPLES, AUDIO_EXTRA_CHANNELS, AUDIO_F_MAX, AUDIO_F_MIN, AUDIO_HOP_LENGTH,
    AUDIO_LOG_EPS, AUDIO_N_FFT, AUDIO_N_MELS, AUDIO_SAMPLE_RATE, AUDIO_WIN_LENGTH, D_AUD, TARGET_FPS, TARGET_T,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "splits" / "subject_isolated_splits.csv"
DEFAULT_STATS = PROJECT_ROOT / "configs" / "audio_normalization_stats.json"
DEFAULT_RAW_CACHE = PROJECT_ROOT / "data" / "processed" / "audio_raw"
DEFAULT_OUT = PROJECT_ROOT / "data" / "processed" / "audio"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "base_config.yaml"

# Quality gate (audio_present). Defaults only catch clear failures; override in configs/base_config.yaml -> audio.quality.
# A value of None switches a check off. The two noise-related checks are OFF by default: choose their thresholds after
# looking at the distributions printed by `quality-report` on the TRAIN split.
QUALITY_DEFAULTS = {
    "enabled": True,
    "min_valid_frames": 12,            # fewer real frames than this (12 = 0.48 s at 25 fps) -> "too_few_frames"
    "min_rms_db": -55.0,               # clip RMS (dBFS) below this -> "near_silent"
    "max_clipped_fraction": 0.01,      # share of samples with |x| >= 0.999 above this -> "clipped"
    "min_level_range_db": None,        # p90 - p10 of frame loudness below this -> "low_level_range" (crude noise proxy)
    "min_voiced_ratio": None,          # voiced frames / valid frames below this -> "low_voiced_ratio"
}
QUALITY_METRICS = ("n_valid_frames", "rms_db", "peak", "clipped_fraction", "level_range_db", "voiced_ratio", "hnr_mean_db")

assert AUDIO_WIN_LENGTH == AUDIO_N_FFT, "this implementation assumes win_length == n_fft"
SPLITS = ("train", "val", "test")
SCALAR_CHANNELS = ("f0_semitone", "loudness_db", "hnr_db")      # z-scored with train statistics (voiced frames only for f0/hnr)
MIN_SAMPLES = AUDIO_N_FFT                                        # shorter audio cannot be reflect-padded for the STFT


class NoAudioStream(RuntimeError):
    """The video file has no audio stream."""


class AudioTooShort(ValueError):
    """Fewer than MIN_SAMPLES decoded samples."""


# ---------------------------------------------------------------------------
# 1. Decoding
# ---------------------------------------------------------------------------
def decode_audio(video_path: str | Path) -> np.ndarray:
    """Decode the first audio stream to 16 kHz mono float32, at most AUDIO_CLIP_SAMPLES long.

    `aresample=async=1:first_pts=0` places the audio on the container timeline: if the audio stream starts later than
    the video (e.g. at 0.5 s) the gap is filled with silence, so sample 0 always corresponds to video time 0.
    """
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(video_path), "-map", "0:a:0", "-vn", "-ac", "1",
           "-ar", str(AUDIO_SAMPLE_RATE), "-af", "aresample=async=1:first_pts=0",
           "-t", f"{AUDIO_CLIP_SAMPLES / AUDIO_SAMPLE_RATE:.6f}", "-f", "f32le", "-"]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        err = res.stderr.decode(errors="replace").strip()
        if "matches no streams" in err:
            raise NoAudioStream(f"{video_path}: no audio stream")
        raise RuntimeError(f"ffmpeg failed on {video_path}: {err.splitlines()[0] if err else res.returncode}")
    y = np.frombuffer(res.stdout, dtype="<f4").astype(np.float32)[:AUDIO_CLIP_SAMPLES]
    if y.size < MIN_SAMPLES:
        raise AudioTooShort(f"{video_path}: only {y.size} samples decoded")
    return y


# ---------------------------------------------------------------------------
# 2. Log-mel (numerically equivalent to torchaudio.transforms.MelSpectrogram with power=2, center=True,
#    pad_mode="reflect", periodic Hann window, mel_scale="htk", norm=None; see test_mel_matches_torchaudio)
# ---------------------------------------------------------------------------
def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


@lru_cache(maxsize=None)
def mel_filterbank(sr=AUDIO_SAMPLE_RATE, n_fft=AUDIO_N_FFT, n_mels=AUDIO_N_MELS,
                   f_min=AUDIO_F_MIN, f_max=AUDIO_F_MAX) -> np.ndarray:
    """(n_mels, n_fft // 2 + 1) triangular filterbank."""
    all_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    f_pts = _mel_to_hz(np.linspace(_hz_to_mel(f_min), _hz_to_mel(f_max), n_mels + 2))
    f_diff = np.diff(f_pts)
    slopes = f_pts[None, :] - all_freqs[:, None]
    down = -slopes[:, :-2] / f_diff[:-1]
    up = slopes[:, 2:] / f_diff[1:]
    fb = np.maximum(0.0, np.minimum(down, up))
    if (fb.sum(axis=0) == 0).any():
        raise ValueError("mel filterbank has empty filters: lower AUDIO_N_MELS or raise AUDIO_N_FFT")
    return fb.T


@lru_cache(maxsize=None)
def _window(n_fft=AUDIO_N_FFT) -> np.ndarray:
    return np.hanning(n_fft + 1)[:-1]                 # periodic Hann (== torch.hann_window)


def frame_signal(y: np.ndarray, n_fft=AUDIO_N_FFT, hop=AUDIO_HOP_LENGTH) -> np.ndarray:
    """Centered frames: frame k is centered on sample k * hop. Returns (1 + len(y) // hop, n_fft)."""
    n_frames = 1 + len(y) // hop
    yp = np.pad(y.astype(np.float64), (n_fft // 2, n_fft // 2), mode="reflect")
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    return yp[idx]


def log_mel_and_loudness(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frames = frame_signal(y)
    power = np.abs(np.fft.rfft(frames * _window(), axis=1)) ** 2
    log_mel = np.log(power @ mel_filterbank().T + AUDIO_LOG_EPS)               # (frames, n_mels)
    loudness_db = 10.0 * np.log10(np.mean(frames ** 2, axis=1) + 1e-10)        # energy of the same 64 ms window
    return log_mel.astype(np.float32), np.maximum(loudness_db, -100.0).astype(np.float32)


# ---------------------------------------------------------------------------
# 3. Pitch and HNR (Praat). Injectable so the rest of the pipeline can be tested without Praat.
# ---------------------------------------------------------------------------
ProsodyFn = Callable[[np.ndarray, int, np.ndarray], tuple[np.ndarray, np.ndarray]]


def praat_prosody(y: np.ndarray, sr: int, centers_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """F0 (Hz, 0 = unvoiced) and HNR (dB, NaN where unvoiced) sampled at the frame centers."""
    import parselmouth                                           # lazy import: only needed when extracting

    snd = parselmouth.Sound(np.asarray(y, dtype=np.float64), sampling_frequency=float(sr))
    pitch = snd.to_pitch(pitch_floor=75, pitch_ceiling=400)      # same settings as the EDA
    harm = snd.to_harmonicity_cc()
    t_p, f0_p = np.asarray(pitch.xs()), np.asarray(pitch.selected_array["frequency"])
    t_h, h_p = np.asarray(harm.xs()), np.asarray(harm.values[0])

    def nearest(t_src, t_q, tol=0.02):
        d = np.abs(t_src[None, :] - t_q[:, None])
        j = d.argmin(axis=1)
        return j, d[np.arange(len(t_q)), j] <= tol

    jp, okp = nearest(t_p, centers_s)
    jh, okh = nearest(t_h, centers_s)
    f0 = np.where(okp, f0_p[jp], 0.0).astype(np.float32)
    hnr = np.where(okh & (f0 > 0) & (h_p[jh] > -100.0), h_p[jh], np.nan).astype(np.float32)
    return f0, hnr


# ---------------------------------------------------------------------------
# 4. Raw (un-normalised) features
# ---------------------------------------------------------------------------
@dataclass
class RawAudio:
    log_mel: np.ndarray       # (n_valid, AUDIO_N_MELS)
    f0_semitone: np.ndarray   # (n_valid,)  NaN where unvoiced
    voiced: np.ndarray        # (n_valid,)  0/1
    loudness_db: np.ndarray   # (n_valid,)
    hnr_db: np.ndarray        # (n_valid,)  NaN where unvoiced
    n_samples: int

    @property
    def n_valid(self) -> int:
        return self.log_mel.shape[0]


def compute_raw(y: np.ndarray, prosody_fn: ProsodyFn | None = None) -> RawAudio:
    prosody_fn = prosody_fn or praat_prosody                           # resolved at call time (tests can swap it)
    y = np.asarray(y, dtype=np.float32)[:AUDIO_CLIP_SAMPLES]
    if y.size < MIN_SAMPLES:
        raise AudioTooShort(f"{y.size} samples")
    n_valid = min(TARGET_T, math.ceil(y.size / AUDIO_HOP_LENGTH))      # frames whose center lies inside the audio
    log_mel, loudness = log_mel_and_loudness(y)
    log_mel, loudness = log_mel[:n_valid], loudness[:n_valid]
    centers = np.arange(n_valid) * AUDIO_HOP_LENGTH / AUDIO_SAMPLE_RATE
    try:
        f0, hnr = prosody_fn(y, AUDIO_SAMPLE_RATE, centers)
    except Exception as e:                                             # Praat can fail on odd signals: keep the clip, mark unvoiced
        print(f"  warning: prosody failed ({e}); channels set to unvoiced", file=sys.stderr)
        f0, hnr = np.zeros(n_valid, np.float32), np.full(n_valid, np.nan, np.float32)
    voiced = (f0 > 0).astype(np.float32)
    f0_st = np.where(f0 > 0, 12.0 * np.log2(np.maximum(f0, 1e-6) / 27.5), np.nan).astype(np.float32)
    hnr = np.where(voiced > 0, hnr, np.nan).astype(np.float32)
    return RawAudio(log_mel, f0_st, voiced, loudness, hnr, int(y.size))


# ---------------------------------------------------------------------------
# 5. Train-only normalisation statistics
# ---------------------------------------------------------------------------
def _config() -> dict:
    cfg = dict(sample_rate=AUDIO_SAMPLE_RATE, n_fft=AUDIO_N_FFT, win_length=AUDIO_WIN_LENGTH, hop_length=AUDIO_HOP_LENGTH,
               n_mels=AUDIO_N_MELS, f_min=AUDIO_F_MIN, f_max=AUDIO_F_MAX, log_eps=AUDIO_LOG_EPS,
               target_t=TARGET_T, target_fps=TARGET_FPS, extra_channels=list(AUDIO_EXTRA_CHANNELS), d_aud=D_AUD,
               window="first TARGET_T/TARGET_FPS seconds from container t=0",
               mel="htk, norm=None, power=2, periodic hann, center=True reflect",
               loudness="10*log10(mean(x^2)) over the 1024-sample frame, floor -100 dB",
               pitch="praat to_pitch floor 75 / ceiling 400 Hz; f0_semitone = 12*log2(f0/27.5)",
               hnr="praat to_harmonicity_cc, voiced frames only")
    return json.loads(json.dumps(cfg, sort_keys=True))


class _Acc:
    """Streaming per-channel mean/std/min/max in float64; NaNs are ignored."""

    def __init__(self, shape):
        self.n, self.s, self.ss = np.zeros(shape), np.zeros(shape), np.zeros(shape)
        self.lo, self.hi = np.full(shape, np.inf), np.full(shape, -np.inf)

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        m = ~np.isnan(x)
        z = np.where(m, x, 0.0)
        self.n += m.sum(axis=0)
        self.s += z.sum(axis=0)
        self.ss += (z ** 2).sum(axis=0)
        self.lo = np.minimum(self.lo, np.where(m, x, np.inf).min(axis=0))
        self.hi = np.maximum(self.hi, np.where(m, x, -np.inf).max(axis=0))

    def finalize(self):
        mean = self.s / np.maximum(self.n, 1)
        var = (self.ss - self.n * mean ** 2) / np.maximum(self.n - 1, 1)
        return mean, np.maximum(np.sqrt(np.maximum(var, 0.0)), 1e-6)


@dataclass
class AudioStats:
    mel_mean: np.ndarray
    mel_std: np.ndarray
    scalars: dict             # channel -> {"mean", "std", "min", "max", "n"}
    n_clips: int
    n_frames: int
    meta: dict

    def to_json(self) -> dict:
        return {"config": _config(), "mel_mean": self.mel_mean.tolist(), "mel_std": self.mel_std.tolist(),
                "scalars": self.scalars, "n_clips": self.n_clips, "n_frames": self.n_frames, "meta": self.meta}


def fit_stats(raws, meta: dict | None = None) -> AudioStats:
    """`raws` must come from TRAINING clips only (guidelines 2.1). Statistics use valid (non-padded) frames."""
    mel_acc = _Acc((AUDIO_N_MELS,))
    sc_acc = {name: _Acc(()) for name in SCALAR_CHANNELS}
    n_clips = n_frames = 0
    for raw in raws:
        mel_acc.update(raw.log_mel)
        sc_acc["f0_semitone"].update(raw.f0_semitone)
        sc_acc["loudness_db"].update(raw.loudness_db)
        sc_acc["hnr_db"].update(raw.hnr_db)
        n_clips += 1
        n_frames += raw.n_valid
    if n_clips == 0:
        raise ValueError("no training clips to fit statistics on")
    mean, std = mel_acc.finalize()
    scalars = {}
    for name, acc in sc_acc.items():
        m, s = acc.finalize()
        scalars[name] = dict(mean=float(m), std=float(s), min=float(acc.lo), max=float(acc.hi), n=int(acc.n))
    return AudioStats(mean, std, scalars, n_clips, n_frames, meta or {})


def save_stats(stats: AudioStats, path: str | Path = DEFAULT_STATS):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats.to_json(), indent=2))


def load_stats(path: str | Path = DEFAULT_STATS) -> AudioStats:
    d = json.loads(Path(path).read_text())
    if d["config"] != _config():
        diff = sorted(k for k in set(d["config"]) | set(_config()) if d["config"].get(k) != _config().get(k))
        raise ValueError(f"{path} was fitted with a different audio configuration (differs in {diff}). "
                         "Re-run `fit-stats` -- applying stale statistics silently corrupts the features.")
    return AudioStats(np.array(d["mel_mean"]), np.array(d["mel_std"]), d["scalars"], d["n_clips"], d["n_frames"], d["meta"])


# ---------------------------------------------------------------------------
# 6. Raw -> contract tensor (T, D_AUD)
# ---------------------------------------------------------------------------
def to_array(raw: RawAudio, stats: AudioStats) -> np.ndarray:
    """Normalise and pad to (TARGET_T, D_AUD) float32. Padding rows (and unvoiced f0/hnr) are 0 = the train mean."""
    n, idx = raw.n_valid, AUDIO_CHANNEL_INDEX
    out = np.zeros((TARGET_T, D_AUD), dtype=np.float32)
    out[:n, :AUDIO_N_MELS] = (raw.log_mel - stats.mel_mean.astype(np.float32)) / stats.mel_std.astype(np.float32)
    out[:n, idx["valid"]] = 1.0
    out[:n, idx["voiced"]] = raw.voiced
    for name in SCALAR_CHANNELS:
        s = stats.scalars[name]
        z = (getattr(raw, name) - s["mean"]) / s["std"]
        out[:n, idx[name]] = np.nan_to_num(z, nan=0.0)
    return out


def make_sample(arr: np.ndarray, label: int, clip_id: str, n_valid: int,
                audio_present: bool = True, audio_issues: str = "") -> dict:
    import torch
    from src.common.contracts import check_audio_sample      # contract lives in common/, shared by all modalities
    sample = {"audio_feats": torch.from_numpy(arr).float(), "label": torch.tensor(int(label), dtype=torch.long),
              "audio_present": torch.tensor(bool(audio_present), dtype=torch.bool),
              "audio_issues": audio_issues,                    # "" when usable, else e.g. "near_silent;clipped"
              "clip_id": clip_id, "n_valid_frames": int(n_valid)}
    check_audio_sample(sample)
    return sample


# ---------------------------------------------------------------------------
# 7. Settings from configs/base_config.yaml and the quality gate (audio_present)
# ---------------------------------------------------------------------------
def load_audio_settings(path: str | Path | None = None) -> dict:
    """The `audio:` section of configs/base_config.yaml ({} when the file or the section is absent).
    Read here: stats_path, data_dir (output folder), optional manifest / raw_cache_dir, and the `quality:` block."""
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            f"PyYAML is required to read {path}; install it in your venv with `pip install pyyaml`."
        ) from e
    return dict((yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("audio") or {})


def resolve_quality(overrides: dict | None = None) -> dict:
    q = dict(QUALITY_DEFAULTS)
    unknown = set(overrides or {}) - set(q)
    if unknown:
        raise ValueError(f"unknown audio.quality settings {sorted(unknown)}; allowed: {sorted(q)}")
    q.update(overrides or {})
    return q


def _cfg_path(settings: dict, key: str, default: Path) -> Path:
    if settings.get(key):
        p = Path(settings[key])
        return p if p.is_absolute() else PROJECT_ROOT / p
    return default


def audio_metrics(y: np.ndarray, raw: RawAudio) -> dict:
    """Per-clip quality measurements from the decoded waveform and the raw features."""
    y64 = np.asarray(y, dtype=np.float64)
    with np.errstate(all="ignore"):
        rms_db = 20.0 * np.log10(np.sqrt(np.mean(y64 ** 2)) + 1e-10)
        loud = raw.loudness_db
        level_range = float(np.subtract(*np.percentile(loud, [90, 10]))) if loud.size >= 2 else 0.0
        hnr = raw.hnr_db[np.isfinite(raw.hnr_db)]
        return {
            "n_valid_frames": int(raw.n_valid),
            "rms_db": float(rms_db),
            "peak": float(np.max(np.abs(y64))),
            "clipped_fraction": float(np.mean(np.abs(y64) >= 0.999)),
            "level_range_db": level_range,
            "voiced_ratio": float(np.mean(raw.voiced)) if raw.voiced.size else 0.0,
            "hnr_mean_db": float(hnr.mean()) if hnr.size else float("nan"),
            "finite": bool(np.isfinite(y64).all() and np.isfinite(raw.log_mel).all() and np.isfinite(loud).all()),
        }


def judge_quality(m: dict, q: dict) -> list[str]:
    """Names of the failed checks (empty list = usable)."""
    issues = []
    if not m["finite"]:
        issues.append("non_finite")
    if m["n_valid_frames"] < q["min_valid_frames"]:
        issues.append("too_few_frames")
    if m["rms_db"] < q["min_rms_db"]:
        issues.append("near_silent")
    if m["clipped_fraction"] > q["max_clipped_fraction"]:
        issues.append("clipped")
    if q["min_level_range_db"] is not None and m["level_range_db"] < q["min_level_range_db"]:
        issues.append("low_level_range")
    if q["min_voiced_ratio"] is not None and m["voiced_ratio"] < q["min_voiced_ratio"]:
        issues.append("low_voiced_ratio")
    return issues


def _decode_or_issue(video_path) -> tuple[np.ndarray | None, str | None]:
    """(waveform, None), or (None, issue) for a media problem. A missing FILE is a pipeline error and raises, so a wrong
    path in the manifest can never turn the whole dataset into 'audio absent'."""
    if not Path(video_path).exists():
        raise FileNotFoundError(f"video not found: {video_path}")
    try:
        return decode_audio(video_path), None
    except NoAudioStream:
        return None, "no_audio_stream"
    except AudioTooShort:
        return None, "audio_too_short"
    except RuntimeError:
        return None, "decode_error"


# ---------------------------------------------------------------------------
# 8. Manifest, caching and the CLI
# ---------------------------------------------------------------------------
_LABEL_RE = re.compile(r"(?i)_(lie|deception|deceptive|truth|true)_?\d+$")
_LIE_WORDS = ("lie", "deception", "deceptive")          # some sources (e.g. BRI) write "deception" instead of "lie"


def parse_label(clip_id: str, value=None) -> int:
    if value is not None and not (isinstance(value, float) and math.isnan(value)):
        v = str(value).strip().lower()
        if v in ("1", "lie", "deception", "deceptive"):
            return 1
        if v in ("0", "truth", "true", "truthful"):
            return 0
        raise ValueError(f"{clip_id}: unknown label {value!r}")
    m = _LABEL_RE.search(re.sub(r"\s+", "", clip_id))
    if not m:
        raise ValueError(f"cannot parse a label from clip id {clip_id!r}")
    return 1 if m.group(1).lower() in _LIE_WORDS else 0


def read_manifest(path: str | Path, root: Path = PROJECT_ROOT):
    import pandas as pd
    df = pd.read_csv(path)
    missing = {"clip_id", "relative_path", "split"} - set(df.columns)
    if missing:
        raise ValueError(f"manifest {path} lacks columns {sorted(missing)}")
    df["split"] = df["split"].astype(str).str.lower().str.strip()
    bad = set(df["split"]) - set(SPLITS)
    if bad:
        raise ValueError(f"unknown split values {bad}")
    raw_labels = df["label"] if "label" in df else [None] * len(df)
    labels, bad_ids = [], []
    for c, v in zip(df["clip_id"], raw_labels):
        try:
            labels.append(parse_label(c, v))
        except ValueError:
            labels.append(None); bad_ids.append(c)
    if bad_ids:                                                  # report ALL of them, not just the first
        raise ValueError(f"{len(bad_ids)} clip ids have no parseable label, e.g. {bad_ids[:10]}")
    df["label"] = labels
    df["video_path"] = [root / str(p).replace("\\", "/") for p in df["relative_path"]]   # tolerate Windows-style separators
    return df


def _signature() -> str:
    return hashlib.md5(json.dumps(_config(), sort_keys=True).encode()).hexdigest()


def get_raw(clip_id: str, video_path: Path, cache_dir: Path | None, prosody_fn: ProsodyFn | None = None,
            y: np.ndarray | None = None) -> RawAudio:
    """Raw features for one clip, cached as .npz (invalidated whenever the audio configuration changes)."""
    cache = None if cache_dir is None else Path(cache_dir) / f"{clip_id}.npz"
    if cache is not None and cache.exists():
        z = np.load(cache, allow_pickle=False)
        if str(z["signature"]) == _signature():
            return RawAudio(z["log_mel"], z["f0_semitone"], z["voiced"], z["loudness_db"], z["hnr_db"], int(z["n_samples"]))
    raw = compute_raw(y if y is not None else decode_audio(video_path), prosody_fn)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, log_mel=raw.log_mel, f0_semitone=raw.f0_semitone, voiced=raw.voiced,
                            loudness_db=raw.loudness_db, hnr_db=raw.hnr_db, n_samples=raw.n_samples,
                            signature=_signature())
    return raw


def cmd_fit_stats(args) -> None:
    df = read_manifest(args.manifest)
    train = df[df["split"] == "train"]                      # guidelines 2.1: statistics from the training partition only
    print(f"Fitting audio statistics on {len(train)} TRAIN clips (val/test are never read here).")
    raws, failures = [], []
    for i, r in enumerate(train.itertuples(), 1):
        try:
            raws.append(get_raw(r.clip_id, r.video_path, args.raw_cache))
        except Exception as e:
            failures.append((r.clip_id, str(e)))
        if i % 100 == 0:
            print(f"  {i}/{len(train)}")
    stats = fit_stats(raws, meta={"manifest": str(Path(args.manifest).name), "split": "train",
                                  "n_failed": len(failures)})
    save_stats(stats, args.stats)
    print(f"Saved {args.stats}  ({stats.n_clips} clips, {stats.n_frames} frames, {len(failures)} failed)")
    for name, s in stats.scalars.items():
        print(f"  {name:12s} mean={s['mean']:.3f} std={s['std']:.3f} range=[{s['min']:.2f}, {s['max']:.2f}] n={s['n']}")
    for cid, msg in failures[:20]:
        print(f"  FAILED {cid}: {msg}")


def assess_clip(clip_id: str, video_path, cache_dir, quality: dict):
    """-> (raw | None, metrics | None, issues). Media problems become issues; only pipeline errors raise."""
    y, issue = _decode_or_issue(video_path)
    if issue:
        return None, None, [issue]
    raw = get_raw(clip_id, video_path, cache_dir, y=y)
    metrics = audio_metrics(y, raw)
    return raw, metrics, (judge_quality(metrics, quality) if quality["enabled"] else [])


def cmd_extract(args) -> None:
    import pandas as pd
    import torch
    quality = getattr(args, "quality", None) or resolve_quality(None)
    stats = load_stats(args.stats)                          # raises if missing or fitted with another configuration
    df = read_manifest(args.manifest)
    out_root, failures, rows = Path(args.out), [], []
    for i, r in enumerate(df.itertuples(), 1):
        try:
            raw, metrics, issues = assess_clip(r.clip_id, r.video_path, args.raw_cache, quality)
            present = not issues
            if present:
                arr, n_valid = to_array(raw, stats), raw.n_valid
            else:                                           # unusable audio: exact zeros + audio_present = False
                arr, n_valid = np.zeros((TARGET_T, D_AUD), dtype=np.float32), 0
            sample = make_sample(arr, r.label, r.clip_id, n_valid, audio_present=present, audio_issues=";".join(issues))
            dest = out_root / r.split / f"{r.clip_id}.pt"
            dest.parent.mkdir(parents=True, exist_ok=True)
            torch.save(sample, dest)
            rows.append({"clip_id": r.clip_id, "split": r.split, "label": r.label, "audio_present": present,
                         "issues": ";".join(issues), **(metrics or {})})
        except Exception as e:
            failures.append({"clip_id": r.clip_id, "split": r.split, "reason": str(e)})
        if i % 100 == 0:
            print(f"  {i}/{len(df)}")
    print(f"Wrote {len(df) - len(failures)} / {len(df)} samples to {out_root}")
    if rows:
        rep = pd.DataFrame(rows)
        out_root.mkdir(parents=True, exist_ok=True)
        rep.to_csv(out_root / "audio_quality.csv", index=False)
        n_bad = int((~rep["audio_present"]).sum())
        print(f"audio_present = False for {n_bad} / {len(rep)} clips ({100 * n_bad / len(rep):.1f}%); "
              f"details in {out_root / 'audio_quality.csv'}")
        counts = Counter(x for s_ in rep["issues"] for x in s_.split(";") if x)
        for k, v in counts.most_common():
            print(f"  {k}: {v}")
        if n_bad:
            print("  flagged share by label:", {("lie" if k else "truth"): f"{100 * g.mean():.1f}%"
                                              for k, g in (~rep["audio_present"]).groupby(rep["label"])})
        if n_bad > 0.10 * len(rep):
            print("  WARNING: more than 10% of clips are flagged. Check the thresholds with `quality-report`.")
    if failures:
        out_root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(failures).to_csv(out_root / "extraction_failures.csv", index=False)
        print(f"{len(failures)} failures listed in {out_root / 'extraction_failures.csv'}")


def cmd_quality_report(args) -> None:
    """Read-only: quality metrics of the chosen splits (default train), how many clips each check would flag, and the
    flagged share per label (a gate that flags one label more than the other would leak the label)."""
    import pandas as pd
    quality = getattr(args, "quality", None) or resolve_quality(None)
    df = read_manifest(args.manifest)
    df = df[df["split"].isin(args.splits or ["train"])]
    print(f"Quality report on {len(df)} clips, splits {args.splits or ['train']}; thresholds: {quality}")
    rows = []
    for i, r in enumerate(df.itertuples(), 1):
        try:
            _, metrics, issues = assess_clip(r.clip_id, r.video_path, args.raw_cache, quality)
        except Exception as e:
            print(f"  FAILED {r.clip_id}: {e}")
            continue
        rows.append({"clip_id": r.clip_id, "split": r.split, "label": r.label, "issues": ";".join(issues),
                     **(metrics or {})})
        if i % 100 == 0:
            print(f"  {i}/{len(df)}")
    rep = pd.DataFrame(rows)
    if rep.empty:
        print("no clips assessed"); return
    pd.set_option("display.width", 200)
    cols = [c for c in QUALITY_METRICS if c in rep]
    print("\nDistribution of the measurements (p1 ... p99):")
    print(rep[cols].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).round(3).T[["min", "1%", "5%", "50%", "95%", "99%", "max"]])
    flagged = rep["issues"] != ""
    print(f"\nFlagged with the current thresholds: {int(flagged.sum())} / {len(rep)}")
    for k, v in Counter(x for s_ in rep["issues"] for x in s_.split(";") if x).most_common():
        print(f"  {k}: {v}")
    print("Flagged share by label:", {("lie" if k else "truth"): f"{100 * g.mean():.1f}% of {len(g)}"
                                       for k, g in flagged.groupby(rep["label"])})
    if args.csv:
        rep.to_csv(args.csv, index=False)
        print(f"saved {args.csv}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("fit-stats", cmd_fit_stats), ("quality-report", cmd_quality_report), ("extract", cmd_extract)):
        sp = sub.add_parser(name)
        sp.add_argument("--manifest", type=Path, default=None)
        sp.add_argument("--stats", type=Path, default=None)
        sp.add_argument("--raw-cache", type=Path, default=None)
        sp.add_argument("--config", type=Path, default=None, help="default: configs/base_config.yaml")
        if name == "extract":
            sp.add_argument("--out", type=Path, default=None)
        if name == "quality-report":
            sp.add_argument("--splits", nargs="+", default=None, choices=SPLITS)
            sp.add_argument("--csv", type=Path, default=None)
        sp.set_defaults(func=fn)
    args = p.parse_args(argv)
    settings = load_audio_settings(args.config)                  # command-line paths win over the YAML, the YAML over the defaults
    args.manifest = args.manifest or _cfg_path(settings, "manifest", DEFAULT_MANIFEST)
    args.stats = args.stats or _cfg_path(settings, "stats_path", DEFAULT_STATS)
    args.raw_cache = args.raw_cache or _cfg_path(settings, "raw_cache_dir", DEFAULT_RAW_CACHE)
    if args.cmd == "extract":
        args.out = args.out or _cfg_path(settings, "data_dir", DEFAULT_OUT)
    args.quality = resolve_quality(settings.get("quality"))
    args.func(args)


if __name__ == "__main__":
    main()