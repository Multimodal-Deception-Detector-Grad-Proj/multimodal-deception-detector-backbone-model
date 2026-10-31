"""Audio pipeline verification (guidelines 5.2).

Run from the repository root:   python -m pytest tests/test_audio_pipeline.py -v
or without pytest:              python tests/test_audio_pipeline.py

Most tests are hermetic: they synthesise small mp4 clips with ffmpeg and replace Praat by a stub, so they run anywhere.
Tests that need parselmouth, torch, torchaudio or real clips under data/raw are skipped when those are unavailable.
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.constants import (  # noqa: E402
    AUDIO_CHANNEL_INDEX, AUDIO_CLIP_SAMPLES, AUDIO_HOP_LENGTH, AUDIO_LOG_EPS, AUDIO_N_FFT, AUDIO_N_MELS,
    AUDIO_SAMPLE_RATE, D_AUD, TARGET_T,
)
import src.extractors.audio.extract_audio as ea  # noqa: E402


def skip(reason):
    raise unittest.SkipTest(reason)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def stub_prosody(y, sr, centers_s):
    """Praat stand-in: 'voiced' (150-170 Hz, HNR 12 dB) wherever the local signal is loud."""
    idx = np.clip((centers_s * sr).astype(int), 0, len(y) - 1)
    rms = np.array([np.sqrt(np.mean(y[max(0, i - 256): i + 256] ** 2)) for i in idx])
    voiced = rms > 0.02
    f0 = np.where(voiced, 150.0 + 20.0 * np.sin(centers_s * 3.0), 0.0).astype(np.float32)
    hnr = np.where(voiced, 12.0, np.nan).astype(np.float32)
    return f0, hnr


def make_clip(path: Path, dur: float, audio_delay: float = 0.0, with_audio: bool = True, freq: int = 180):
    """25 fps test video; the audio is a gated sine (0.4 s on / 0.2 s off) so loudness and voicing vary over time."""
    cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"testsrc=size=64x64:rate=25:duration={dur}"]
    if with_audio:
        if audio_delay:
            cmd += ["-itsoffset", str(audio_delay)]
        cmd += ["-f", "lavfi", "-i", f"sine=frequency={freq}:duration={dur - audio_delay}"]
        cmd += ["-filter_complex", "[1:a]volume='if(lt(mod(t,0.6),0.4),1,0.02)':eval=frame[a]", "-map", "0:v", "-map", "[a]",
                "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True)


_CACHE = {}


def clips() -> dict:
    """Synthetic clips, created once per run."""
    if not _CACHE:
        if shutil.which("ffmpeg") is None:
            skip("ffmpeg not installed")
        d = Path(tempfile.mkdtemp(prefix="audio_pipeline_test_"))
        spec = {"A_WILTY_EP1_lie1": dict(dur=5.0), "A_WILTY_EP1_truth2": dict(dur=3.0, freq=220),
                "B_WILTY_EP2_lie_3": dict(dur=12.0, freq=150), "B_WILTY_EP2_truth4": dict(dur=4.0, freq=260),
                "C_WILTY_EP3_lie5": dict(dur=5.0, freq=200), "late": dict(dur=5.0, audio_delay=0.5),
                "noaudio": dict(dur=3.0, with_audio=False)}
        for name, kw in spec.items():
            make_clip(d / f"{name}.mp4", **kw)
        _CACHE.update({"dir": d, **{n: d / f"{n}.mp4" for n in spec}})
    return _CACHE


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        skip("torch not installed")


# ---------------------------------------------------------------------------
# decoding and alignment
# ---------------------------------------------------------------------------
def test_decode_lengths():
    c = clips()
    y5 = ea.decode_audio(c["A_WILTY_EP1_lie1"])
    assert y5.dtype == np.float32 and abs(len(y5) - 5 * AUDIO_SAMPLE_RATE) < 0.1 * AUDIO_SAMPLE_RATE, len(y5)
    y12 = ea.decode_audio(c["B_WILTY_EP2_lie_3"])
    assert len(y12) == AUDIO_CLIP_SAMPLES, f"12 s clip must be cut to the 8 s window, got {len(y12)}"


def test_late_audio_is_aligned_to_video_time_zero():
    """Audio starting 0.5 s into the container must be preceded by ~0.5 s of silence (sample 0 == video time 0)."""
    y = ea.decode_audio(clips()["late"])
    n = AUDIO_SAMPLE_RATE // 10
    assert np.abs(y[: 4 * n]).max() < 1e-2, "first 0.4 s should be silent"
    assert np.abs(y[6 * n: 12 * n]).max() > 0.05, "audio should be present after ~0.5 s"


def test_missing_audio_stream_raises():
    try:
        ea.decode_audio(clips()["noaudio"])
    except ea.NoAudioStream:
        return
    raise AssertionError("expected NoAudioStream")


# ---------------------------------------------------------------------------
# log-mel
# ---------------------------------------------------------------------------
def test_mel_filterbank_has_no_empty_filters():
    fb = ea.mel_filterbank()
    assert fb.shape == (AUDIO_N_MELS, AUDIO_N_FFT // 2 + 1)
    assert (fb.sum(axis=1) > 0).all()


def test_frame_count_and_centering():
    y = np.random.default_rng(0).standard_normal(AUDIO_CLIP_SAMPLES).astype(np.float32)
    assert ea.frame_signal(y).shape == (TARGET_T + 1, AUDIO_N_FFT)       # 201 frames for exactly 8 s -> the extractor keeps 200
    # frame k is centred on sample k * hop
    f = ea.frame_signal(y)
    k = 7
    assert np.allclose(f[k, AUDIO_N_FFT // 2], y[k * AUDIO_HOP_LENGTH])


def test_mel_matches_torchaudio():
    try:
        import torch
        import torchaudio.transforms as T
    except ImportError:
        skip("torchaudio not installed")
    from src.common.constants import AUDIO_F_MAX, AUDIO_F_MIN, AUDIO_WIN_LENGTH
    y = np.random.default_rng(1).standard_normal(3 * AUDIO_SAMPLE_RATE).astype(np.float32) * 0.1
    ref = T.MelSpectrogram(sample_rate=AUDIO_SAMPLE_RATE, n_fft=AUDIO_N_FFT, win_length=AUDIO_WIN_LENGTH,
                           hop_length=AUDIO_HOP_LENGTH, f_min=AUDIO_F_MIN, f_max=AUDIO_F_MAX, n_mels=AUDIO_N_MELS,
                           power=2.0, center=True, pad_mode="reflect", norm=None, mel_scale="htk")
    ref = torch.log(ref(torch.from_numpy(y).unsqueeze(0)) + AUDIO_LOG_EPS)[0].numpy().T
    mine, _ = ea.log_mel_and_loudness(y)
    assert mine.shape == ref.shape, (mine.shape, ref.shape)
    assert np.abs(mine - ref).max() < 5e-3, np.abs(mine - ref).max()


# ---------------------------------------------------------------------------
# raw features -> contract tensor
# ---------------------------------------------------------------------------
def _fit_on(names):
    raws = [ea.compute_raw(ea.decode_audio(clips()[n]), stub_prosody) for n in names]
    return raws, ea.fit_stats(raws)


def test_shape_padding_and_valid_channel():
    raws, stats = _fit_on(["A_WILTY_EP1_lie1", "A_WILTY_EP1_truth2", "B_WILTY_EP2_lie_3"])
    idx = AUDIO_CHANNEL_INDEX
    short = raws[1]                                                     # 3 s clip
    arr = ea.to_array(short, stats)
    assert arr.shape == (TARGET_T, D_AUD) and arr.dtype == np.float32
    n = short.n_valid
    assert 74 <= n <= 77, n
    assert arr[:n, idx["valid"]].min() == 1.0 and arr[n:].any() == False, "padding must be all zeros incl. valid"
    assert int(arr[:, idx["valid"]].sum()) == n
    long = ea.to_array(raws[2], stats)                                  # 12 s clip -> exactly T valid frames
    assert raws[2].n_valid == TARGET_T and long[:, idx["valid"]].min() == 1.0


def test_unvoiced_frames_are_neutral_and_voiced_flag_matches():
    raws, stats = _fit_on(["A_WILTY_EP1_lie1", "A_WILTY_EP1_truth2"])
    arr, raw = ea.to_array(raws[0], stats), raws[0]
    idx, n = AUDIO_CHANNEL_INDEX, raw.n_valid
    unv = raw.voiced == 0
    assert unv.any() and (~unv).any(), "gated test signal should contain voiced and unvoiced frames"
    assert np.allclose(arr[:n][unv, idx["f0_semitone"]], 0.0) and np.allclose(arr[:n][unv, idx["hnr_db"]], 0.0)
    assert np.array_equal(arr[:n, idx["voiced"]], raw.voiced)


def test_train_statistics_standardise_the_training_set():
    raws, stats = _fit_on(["A_WILTY_EP1_lie1", "A_WILTY_EP1_truth2", "B_WILTY_EP2_lie_3", "B_WILTY_EP2_truth4"])
    mel = np.concatenate([ea.to_array(r, stats)[: r.n_valid, :AUDIO_N_MELS] for r in raws])
    assert np.abs(mel.mean(axis=0)).max() < 1e-3 and np.abs(mel.std(axis=0, ddof=1) - 1).max() < 1e-3
    assert stats.n_clips == 4 and stats.n_frames == sum(r.n_valid for r in raws)


def test_stats_with_stale_configuration_are_rejected():
    _, stats = _fit_on(["A_WILTY_EP1_lie1"])
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "stats.json"
        ea.save_stats(stats, p)
        ea.load_stats(p)                                                # same configuration: fine
        d = json.loads(p.read_text())
        d["config"]["hop_length"] = 533                                 # e.g. statistics left over from the 30 FPS example
        p.write_text(json.dumps(d))
        try:
            ea.load_stats(p)
        except ValueError as e:
            assert "hop_length" in str(e)
            return
    raise AssertionError("stale statistics were accepted")


def test_contract_assertions_catch_bad_tensors():
    torch = _torch()
    good = torch.from_numpy(np.zeros((TARGET_T, D_AUD), np.float32))
    ea.validate_audio_tensor(good)
    for bad in (np.full((TARGET_T, D_AUD), np.nan, np.float32), np.full((TARGET_T, D_AUD), np.inf, np.float32),
                np.zeros((TARGET_T - 1, D_AUD), np.float32), np.zeros((TARGET_T, D_AUD + 1), np.float32)):
        try:
            ea.validate_audio_tensor(torch.from_numpy(bad))
        except AssertionError:
            continue
        raise AssertionError("bad tensor passed validation")


def test_label_parsing():
    assert ea.parse_label("AN_WILTY_EP15_lie10") == 1 and ea.parse_label("BRI_WILTY_EP61_lie_6") == 1
    assert ea.parse_label("LS_WILTY_EP73_true5") == 0 and ea.parse_label("SB_WILTY_EP34_truth 8") == 0
    assert ea.parse_label("x", "Lie") == 1 and ea.parse_label("x", 0) == 0
    try:
        ea.parse_label("AN_WILTY_EP15_maybe3")
    except ValueError:
        return
    raise AssertionError("unparseable id accepted")


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------
def _write_manifest(path: Path):
    c = clips()
    rows = [("A_WILTY_EP1_lie1", "train"), ("A_WILTY_EP1_truth2", "train"), ("B_WILTY_EP2_lie_3", "train"),
            ("B_WILTY_EP2_truth4", "val"), ("C_WILTY_EP3_lie5", "test")]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "relative_path", "split"])
        for cid, split in rows:
            w.writerow([cid, str(c[cid]), split])                       # absolute paths: root / abs == abs
    return rows


def test_cli_fits_on_train_only_and_exports_all_splits():
    torch = _torch()
    real, calls = ea.praat_prosody, []
    real_decode = ea.decode_audio
    ea.praat_prosody = stub_prosody
    ea.decode_audio = lambda p: (calls.append(Path(p).stem), real_decode(p))[1]
    try:
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            rows = _write_manifest(t / "manifest.csv")
            common = ["--manifest", str(t / "manifest.csv"), "--stats", str(t / "stats.json"), "--raw-cache", str(t / "raw")]
            ea.main(["fit-stats", *common])
            assert set(calls) == {c for c, s in rows if s == "train"}, f"fit-stats touched non-train clips: {set(calls)}"
            assert json.loads((t / "stats.json").read_text())["n_clips"] == 3
            ea.main(["extract", *common, "--out", str(t / "out")])
            for cid, split in rows:
                s = torch.load(t / "out" / split / f"{cid}.pt", weights_only=False)
                ea.validate_audio_tensor(s["audio_feats"])
                assert s["audio_feats"].shape == (TARGET_T, D_AUD)
                assert int(s["label"].a if hasattr(s["label"], "a") else s["label"]) == ea.parse_label(cid)
    finally:
        ea.praat_prosody, ea.decode_audio = real, real_decode


def test_three_clips_end_to_end_with_real_praat():
    """Guidelines 5.2: 3 clips end to end, outputs match (T, D). Needs parselmouth + torch."""
    torch = _torch()
    try:
        import parselmouth  # noqa: F401
    except ImportError:
        skip("parselmouth not installed")
    c = clips()
    raws = [ea.compute_raw(ea.decode_audio(c[n])) for n in ("A_WILTY_EP1_lie1", "A_WILTY_EP1_truth2", "B_WILTY_EP2_lie_3")]
    stats = ea.fit_stats(raws)
    for raw in raws:
        t = torch.from_numpy(ea.to_array(raw, stats))
        ea.validate_audio_tensor(t)
        assert t.shape == (TARGET_T, D_AUD)


def test_three_real_clips_from_data_raw():
    torch = _torch()
    try:
        import parselmouth  # noqa: F401
    except ImportError:
        skip("parselmouth not installed")
    found = sorted((ROOT / "data" / "raw").rglob("*.mp4"))[:3]
    if len(found) < 3:
        skip("fewer than 3 clips under data/raw")
    raws = [ea.compute_raw(ea.decode_audio(p)) for p in found]
    stats = ea.fit_stats(raws)
    for raw in raws:
        ea.validate_audio_tensor(torch.from_numpy(ea.to_array(raw, stats)))


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except unittest.SkipTest as e:
            print(f"SKIP  {name}  ({e})")
        except Exception as e:                                          # noqa: BLE001
            failed += 1
            import traceback
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
    if _CACHE:
        shutil.rmtree(_CACHE["dir"], ignore_errors=True)
    print(f"\n{len(tests) - failed} ran without failure, {failed} failed")
    sys.exit(1 if failed else 0)