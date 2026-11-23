"""Dataset for the vision modality.

Reads the cached OpenFace tables (.pkl) written by the EDA notebook:
    {FEATURE_CACHE_DIR}/{video_stem}_{sha256(abspath(video))}.pkl
then slices them into fixed-length sliding-window chunks.

Splitting is done per source video (never per chunk) so overlapping chunks
cannot leak across train/val/test.
"""
import hashlib
import os
import glob

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

AU_R = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45]
AU_C = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 28, 45]

DEFAULT_FEATURES = (
    [f"gaze_{e}_{a}" for e in (0, 1) for a in "xyz"]
    + ["gaze_angle_x", "gaze_angle_y"]
    + [f"pose_{c}" for c in ("Tx", "Ty", "Tz", "Rx", "Ry", "Rz")]
    + [f"AU{i:02d}_r" for i in AU_R]
    + [f"AU{i:02d}_c" for i in AU_C]
)  # 49 dims

LIE_TOKENS = {"deception", "deceptive", "lie", "lies", "false"}
TRUTH_TOKENS = {"truth", "truthful", "true"}


def feature_cache_path_for(video_path, cache_dir):
    """Identical to the notebook's cache naming."""
    file_hash = hashlib.sha256(os.path.abspath(video_path).encode("utf-8")).hexdigest()
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(cache_dir, f"{stem}_{file_hash}.pkl")


def label_from_path(video_path):
    """1 = deceptive, 0 = truthful, from the folder names (then the filename)."""
    parts = [p.lower() for p in os.path.normpath(video_path).split(os.sep)[:-1]]
    parts.append(os.path.splitext(os.path.basename(video_path))[0].lower())
    for p in reversed(parts):
        tokens = set(p.replace("-", "_").split("_")) | {p}
        if tokens & LIE_TOKENS:
            return 1
        if tokens & TRUTH_TOKENS:
            return 0
    return None


def discover_videos(data_dir, cache_dir):
    """Return [(video_path, cache_path, label)] for videos that have a cache and a label."""
    paths = sorted(glob.glob(os.path.join(data_dir, "**", "*.mp4"), recursive=True))
    items, no_cache, no_label = [], 0, 0
    for p in paths:
        label = label_from_path(p)
        cache = feature_cache_path_for(p, cache_dir)
        if label is None:
            no_label += 1
        elif not os.path.exists(cache):
            no_cache += 1
        else:
            items.append((p, cache, label))
    if no_cache or no_label:
        print(f"[discover] skipped {no_cache} videos without cache, {no_label} without label")
    return items


def split_videos(items, val_frac=0.15, test_frac=0.15, seed=0):
    """Stratified video-level split."""
    rng = np.random.RandomState(seed)
    train, val, test = [], [], []
    for lab in sorted({it[2] for it in items}):
        grp = [it for it in items if it[2] == lab]
        rng.shuffle(grp)
        n_val = max(1, int(round(len(grp) * val_frac)))
        n_test = max(1, int(round(len(grp) * test_frac)))
        val += grp[:n_val]
        test += grp[n_val:n_val + n_test]
        train += grp[n_val + n_test:]
    return train, val, test


def load_feature_table(cache_path, feature_cols, conf_thresh=0.75):
    """-> feats (N, D) float32 with NaN -> 0, valid (N,) bool."""
    df = pd.read_pickle(cache_path)
    df.columns = df.columns.str.strip()
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"{os.path.basename(cache_path)} lacks columns: {missing[:5]}...")
    feats = np.nan_to_num(df[feature_cols].to_numpy(dtype=np.float32))
    conf = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0).to_numpy()
    valid = conf >= conf_thresh
    if "success" in df.columns:
        valid &= df["success"].to_numpy() == 1
    return feats, valid


def compute_stats(items, feature_cols, conf_thresh=0.75):
    """Per-feature mean/std over valid frames of the given (train) videos."""
    s, ss, n = 0.0, 0.0, 0
    for _, cache, _ in items:
        x, v = load_feature_table(cache, feature_cols, conf_thresh)
        x = x[v].astype(np.float64)
        s += x.sum(0)
        ss += (x ** 2).sum(0)
        n += len(x)
    mean = s / max(n, 1)
    std = np.sqrt(np.maximum(ss / max(n, 1) - mean ** 2, 1e-12))
    std[std < 1e-6] = 1.0  # constant columns
    return mean.astype(np.float32), std.astype(np.float32)


def chunk_starts(n, chunk_len, stride, min_len):
    if n <= chunk_len:
        return [0]
    starts = list(range(0, n - chunk_len + 1, stride))
    tail = starts[-1] + stride
    if tail < n and n - tail >= min_len:
        starts.append(tail)  # partial tail chunk, padded in __getitem__
    return starts


class VisionChunkDataset(Dataset):
    """Items: dict(x=(T, D) float, length=int, label=int, video_idx=int)."""

    def __init__(self, items, mean, std, feature_cols=DEFAULT_FEATURES, chunk_len=240,
                 stride=180, min_len=30, conf_thresh=0.75):
        self.items = items
        self.mean, self.std = mean, std
        self.chunk_len = chunk_len
        self.tables = []   # per video: (normalised feats with invalid frames zeroed)
        self.index = []    # (video_idx, start)
        for vi, (_, cache, _) in enumerate(items):
            x, valid = load_feature_table(cache, feature_cols, conf_thresh)
            if len(x) == 0:
                self.tables.append(None)
                continue
            x = (x - mean) / std
            x[~valid] = 0.0
            self.tables.append(x)
            for s in chunk_starts(len(x), chunk_len, stride, min_len):
                self.index.append((vi, s))
        self.feature_dim = len(feature_cols)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        vi, s = self.index[i]
        x = self.tables[vi][s:s + self.chunk_len]
        length = len(x)
        out = np.zeros((self.chunk_len, self.feature_dim), dtype=np.float32)
        out[:length] = x
        return {
            "x": torch.from_numpy(out),
            "length": torch.tensor(length, dtype=torch.long),
            "label": torch.tensor(self.items[vi][2], dtype=torch.long),
            "video_idx": torch.tensor(vi, dtype=torch.long),
        }