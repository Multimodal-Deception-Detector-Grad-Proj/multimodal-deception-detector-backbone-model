"""Audio extraction sub-pipeline.

    video clip --ffmpeg-->  16 kHz mono waveform (first TARGET_T/TARGET_FPS seconds, t = 0 is the container's t = 0)
               --> log-mel (AUDIO_N_MELS bins) + frame-level channels, one frame per video frame (hop = SR / FPS)
               --> normalised with TRAIN-ONLY statistics (configs/audio_normalization_stats.json)
               --> (TARGET_T, D_AUD) FloatTensor + label, saved as .pt

Channel layout (constants.py):  [ mel bins | valid | f0_semitone | voiced | loudness_db | hnr_db ]
Padding frames are all zeros, including `valid`.

Usage (from the repository root):
    python -m src.extractors.audio.extract_audio fit-stats --manifest data/splits/subject_isolated_splits.csv
    python -m src.extractors.audio.extract_audio extract   --manifest data/splits/subject_isolated_splits.csv

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


def validate_audio_tensor(t) -> None:
    """Runtime assertions from guidelines 4.2, for a sample (T, D) or a batch (B, T, D)."""
    import torch
    assert not torch.isnan(t).any(), "NaN values detected in feature extraction!"
    assert not torch.isinf(t).any(), "Inf values detected in feature extraction!"
    assert t.ndim in (2, 3), f"expected (T, D) or (B, T, D), got {tuple(t.shape)}"
    assert t.shape[-2] == TARGET_T, f"Expected T={TARGET_T}, got {t.shape[-2]}"
    assert t.shape[-1] == D_AUD, f"Expected D_AUD={D_AUD}, got {t.shape[-1]}"


def make_sample(arr: np.ndarray, label: int, clip_id: str, n_valid: int) -> dict:
    import torch
    feats = torch.from_numpy(arr).float()
    validate_audio_tensor(feats)
    return {"audio_feats": feats, "label": torch.tensor(int(label), dtype=torch.long),
            "clip_id": clip_id, "n_valid_frames": int(n_valid)}


# ---------------------------------------------------------------------------
# 7. Manifest, caching and the CLI
# ---------------------------------------------------------------------------
_LABEL_RE = re.compile(r"(?i)_(lie|truth|true|deception)_?\d+$")


def parse_label(clip_id: str, value=None) -> int:
    if value is not None and not (isinstance(value, float) and math.isnan(value)):
        v = str(value).strip().lower()
        if v in ("1", "lie", "deceptive"):
            return 1
        if v in ("0", "truth", "true", "truthful"):
            return 0
        raise ValueError(f"{clip_id}: unknown label {value!r}")
    m = _LABEL_RE.search(re.sub(r"\s+", "", clip_id))
    if not m:
        raise ValueError(f"cannot parse a label from clip id {clip_id!r}")
    return 1 if m.group(1).lower() == "lie" else 0


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
    df["label"] = [parse_label(c, v) for c, v in zip(df["clip_id"], df["label"] if "label" in df else [None] * len(df))]
    df["video_path"] = [root / str(p).replace("\\", "/") for p in df["relative_path"]]   # tolerate Windows-style separators
    return df


def _signature() -> str:
    return hashlib.md5(json.dumps(_config(), sort_keys=True).encode()).hexdigest()


def get_raw(clip_id: str, video_path: Path, cache_dir: Path | None, prosody_fn: ProsodyFn | None = None) -> RawAudio:
    """Raw features for one clip, cached as .npz (invalidated whenever the audio configuration changes)."""
    cache = None if cache_dir is None else Path(cache_dir) / f"{clip_id}.npz"
    if cache is not None and cache.exists():
        z = np.load(cache, allow_pickle=False)
        if str(z["signature"]) == _signature():
            return RawAudio(z["log_mel"], z["f0_semitone"], z["voiced"], z["loudness_db"], z["hnr_db"], int(z["n_samples"]))
    raw = compute_raw(decode_audio(video_path), prosody_fn)
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


def cmd_extract(args) -> None:
    import torch
    stats = load_stats(args.stats)                          # raises if missing or fitted with another configuration
    df = read_manifest(args.manifest)
    out_root, failures = Path(args.out), []
    for i, r in enumerate(df.itertuples(), 1):
        try:
            raw = get_raw(r.clip_id, r.video_path, args.raw_cache)
            sample = make_sample(to_array(raw, stats), r.label, r.clip_id, raw.n_valid)
            dest = out_root / r.split / f"{r.clip_id}.pt"
            dest.parent.mkdir(parents=True, exist_ok=True)
            torch.save(sample, dest)
        except Exception as e:
            failures.append({"clip_id": r.clip_id, "split": r.split, "reason": str(e)})
        if i % 100 == 0:
            print(f"  {i}/{len(df)}")
    print(f"Wrote {len(df) - len(failures)} / {len(df)} samples to {out_root}")
    if failures:
        import pandas as pd
        rep = out_root / "extraction_failures.csv"
        pd.DataFrame(failures).to_csv(rep, index=False)
        print(f"{len(failures)} failures listed in {rep}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("fit-stats", cmd_fit_stats), ("extract", cmd_extract)):
        sp = sub.add_parser(name)
        sp.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
        sp.add_argument("--stats", type=Path, default=DEFAULT_STATS)
        sp.add_argument("--raw-cache", type=Path, default=DEFAULT_RAW_CACHE)
        if name == "extract":
            sp.add_argument("--out", type=Path, default=DEFAULT_OUT)
        sp.set_defaults(func=fn)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()