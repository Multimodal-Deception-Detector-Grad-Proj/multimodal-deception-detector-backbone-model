"""audio_present: the quality gate, the settings read from configs/base_config.yaml, and the extraction of unusable clips."""
import csv
import shutil
import subprocess

import numpy as np
import pytest

import src.extractors.audio.extract_audio as ea
from src.common.constants import AUDIO_SAMPLE_RATE, D_AUD, TARGET_T

SR = AUDIO_SAMPLE_RATE


def _stub_prosody(y, sr, centers):                        # voiced, 220 Hz, HNR 10 dB on every frame (no Praat needed)
    n = len(centers)
    return np.full(n, 220.0, np.float32), np.full(n, 10.0, np.float32)


def _raw(y):
    return ea.compute_raw(np.asarray(y, np.float32), _stub_prosody)


def _sine(seconds, amp=0.3, f=220.0):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def _issues(y, **overrides):
    q = ea.resolve_quality(overrides)
    return ea.judge_quality(ea.audio_metrics(y, _raw(y)), q)


# ---------------------------------------------------------------------------------------------------------------
# the checks
# ---------------------------------------------------------------------------------------------------------------
def test_clean_clip_is_usable():
    assert _issues(_sine(2.0)) == []


def test_each_default_check_fires_on_its_own_problem():
    assert _issues(np.zeros(2 * SR, np.float32)) == ["near_silent"]
    assert _issues(_sine(2.0, amp=1e-4)) == ["near_silent"]                         # about -90 dBFS
    assert _issues(_sine(0.3)) == ["too_few_frames"]                                # 8 frames < 12
    square = np.where(np.arange(2 * SR) // 40 % 2 == 0, 1.0, -1.0).astype(np.float32)
    assert "clipped" in _issues(square)
    bad = _sine(2.0); bad[100] = np.nan
    assert "non_finite" in _issues(bad)


def test_optional_checks_are_off_by_default_and_work_when_enabled():
    steady = _sine(2.0)                                                              # constant level: range ~ 0 dB
    assert _issues(steady) == []
    assert _issues(steady, min_level_range_db=6.0) == ["low_level_range"]
    unvoiced = lambda y, sr, c: (np.zeros(len(c), np.float32), np.full(len(c), np.nan, np.float32))
    raw = ea.compute_raw(steady, unvoiced)
    m = ea.audio_metrics(steady, raw)
    assert ea.judge_quality(m, ea.resolve_quality({})) == []
    assert ea.judge_quality(m, ea.resolve_quality({"min_voiced_ratio": 0.1})) == ["low_voiced_ratio"]


def test_quality_gate_can_be_disabled_and_unknown_keys_are_rejected():
    assert ea.resolve_quality({"min_rms_db": -30.0})["min_rms_db"] == -30.0
    with pytest.raises(ValueError, match="min_rms"):
        ea.resolve_quality({"min_rms": -30.0})                                      # typo must not be ignored silently


def test_settings_come_from_the_yaml(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("audio:\n  stats_path: some/stats.json\n  data_dir: some/out\n  quality:\n    min_valid_frames: 3\n")
    s = ea.load_audio_settings(cfg)
    assert s["stats_path"] == "some/stats.json" and ea.resolve_quality(s["quality"])["min_valid_frames"] == 3
    assert ea.load_audio_settings(tmp_path / "missing.yaml") == {}                  # no file -> defaults


# ---------------------------------------------------------------------------------------------------------------
# end to end (needs ffmpeg and torch)
# ---------------------------------------------------------------------------------------------------------------
def _make_clip(path, audio=None, volume=1.0):
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:r=25:d=2"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=f={audio}:d=2", "-af", f"volume={volume}", "-c:a", "aac"]
    cmd += ["-c:v", "mpeg4", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True)


@pytest.fixture
def dataset(tmp_path):
    pytest.importorskip("torch")
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    clips = {"A_WILTY_EP1_lie1": ("train", 220, 1.0), "A_WILTY_EP1_truth2": ("train", 330, 1.0),
             "B_WILTY_EP2_lie_3": ("val", None, 1.0),       # no audio stream
             "B_WILTY_EP2_truth4": ("val", 220, 0.0),       # silent track
             "C_WILTY_EP3_lie5": ("test", 440, 1.0)}
    rows = []
    for cid, (split, f, vol) in clips.items():
        p = tmp_path / f"{cid}.mp4"
        _make_clip(p, f, vol)
        rows.append((cid, str(p), split))
    manifest = tmp_path / "m.csv"
    with open(manifest, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["clip_id", "relative_path", "split"]); w.writerows(rows)
    return tmp_path, manifest


def test_unusable_clips_are_saved_as_zeros_with_the_flag_false(dataset):
    import torch
    tmp, manifest = dataset
    cfg = tmp / "c.yaml"                                                              # paths come from the YAML, not flags
    cfg.write_text(f"audio:\n  stats_path: {(tmp / 'stats.json').as_posix()}\n  data_dir: {(tmp / 'out').as_posix()}\n"
                   f"  raw_cache_dir: {(tmp / 'raw').as_posix()}\n")
    common = ["--manifest", str(manifest), "--config", str(cfg)]
    ea.main(["fit-stats"] + common)
    ea.main(["extract"] + common)
    out = tmp / "out"

    good = torch.load(out / "train" / "A_WILTY_EP1_lie1.pt")
    assert bool(good["audio_present"]) and good["audio_issues"] == "" and good["n_valid_frames"] >= 12
    assert float(good["audio_feats"].sum()) != 0.0 and int(good["label"]) == 1

    for split, cid, why, lab in (("val", "B_WILTY_EP2_lie_3", "no_audio_stream", 1), ("val", "B_WILTY_EP2_truth4", "near_silent", 0)):
        s = torch.load(out / split / f"{cid}.pt")
        assert not bool(s["audio_present"]) and s["audio_issues"] == why, (cid, s["audio_issues"])
        assert tuple(s["audio_feats"].shape) == (TARGET_T, D_AUD) and float(abs(s["audio_feats"]).sum()) == 0.0
        assert s["n_valid_frames"] == 0 and int(s["label"]) == lab

    rows = list(csv.DictReader(open(out / "audio_quality.csv")))
    assert len(rows) == 5 and sum(r["audio_present"] == "False" for r in rows) == 2
    assert not (out / "extraction_failures.csv").exists()


def test_missing_video_file_is_a_failure_not_absent_audio(dataset):
    import torch
    tmp, manifest = dataset
    ea.main(["fit-stats", "--manifest", str(manifest), "--stats", str(tmp / "s.json"), "--raw-cache", str(tmp / "raw"),
             "--config", str(tmp / "none.yaml")])
    broken = tmp / "m2.csv"
    with open(broken, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["clip_id", "relative_path", "split"])
        w.writerow(["A_WILTY_EP9_lie1", str(tmp / "does_not_exist.mp4"), "train"])
    ea.main(["extract", "--manifest", str(broken), "--stats", str(tmp / "s.json"), "--raw-cache", str(tmp / "raw"),
             "--out", str(tmp / "out2"), "--config", str(tmp / "none.yaml")])
    assert (tmp / "out2" / "extraction_failures.csv").exists()
    assert not (tmp / "out2" / "train" / "A_WILTY_EP9_lie1.pt").exists()
