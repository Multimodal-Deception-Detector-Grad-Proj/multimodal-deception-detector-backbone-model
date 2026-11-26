"""Loading cached OpenFace-derived feature tensors and wrapping them for PyTorch.

Nothing here runs OpenFace or the emotion model itself: `FeatureStore.ensure` only
computes videos missing from `feature_cache`, everything else is read from disk.

Every sequence carries a (T,) bool validity mask (True = real face observation).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

LABEL_TO_ID = {"Truth": 0, "Lie": 1}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


# --------------------------------------------------------------------------- data container
@dataclass
class SequenceData:
    X: np.ndarray                 # (N, T, D) float32
    y: np.ndarray                 # (N,) int64, 0 = Truth, 1 = Lie
    video_ids: List[str]
    speakers: List[str]
    feature_names: List[str]
    mask: Optional[np.ndarray] = None   # (N, T) bool, True = valid face frame

    def __post_init__(self):
        if self.mask is None:
            self.mask = np.ones(self.X.shape[:2], dtype=bool)
        else:
            self.mask = np.asarray(self.mask, dtype=bool)
            if self.mask.shape != self.X.shape[:2]:
                raise ValueError(f"mask {self.mask.shape} != X[:2] {self.X.shape[:2]}")

    def __len__(self) -> int:
        return len(self.y)

    @property
    def input_dim(self) -> int:
        return self.X.shape[-1]

    def subset(self, idx: Sequence[int]) -> "SequenceData":
        idx = np.asarray(idx)
        return SequenceData(self.X[idx], self.y[idx], [self.video_ids[i] for i in idx],
                            [self.speakers[i] for i in idx], self.feature_names, self.mask[idx])

    def describe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "videos": [len(self)], "speakers": [len(set(self.speakers))],
            "truth": [int((self.y == 0).sum())], "lie": [int((self.y == 1).sum())],
            "T": [self.X.shape[1]], "D": [self.X.shape[2]],
            "valid_frac": [float(self.mask.mean()) if self.mask.size else 0.0],
            "no_valid_frames": [int((~self.mask.any(axis=1)).sum())],
        })


# --------------------------------------------------------------------------- helpers
class SpeakerParser:
    """Speaker identity from a `video_id` (POSIX path relative to dataset root).

    Default: the parent folder name, e.g. deception/DOLOS/Female/Catherine/clip.mp4 -> "Catherine".
    Pass a custom callable if the truth/ folder uses a different layout.
    """

    def __init__(self, fn: Optional[Callable[[str], str]] = None):
        self.fn = fn

    def __call__(self, video_id: str) -> str:
        if self.fn is not None:
            return self.fn(video_id)
        parts = PurePosixPath(video_id).parts
        return parts[-2] if len(parts) >= 2 else parts[0]


class FeatureSelector:
    """Pick feature columns by name prefix (for modality ablations).

    Column selection never affects the frame mask, which is stored separately.
    """

    def __init__(self, include: Optional[Tuple[str, ...]] = None, exclude: Optional[Tuple[str, ...]] = None):
        self.include, self.exclude = include, exclude

    def indices(self, names: Sequence[str]) -> List[int]:
        keep = []
        for i, n in enumerate(names):
            if self.include and not n.startswith(tuple(self.include)):
                continue
            if self.exclude and n.startswith(tuple(self.exclude)):
                continue
            keep.append(i)
        if not keep:
            raise ValueError("FeatureSelector removed every feature")
        return keep


# --------------------------------------------------------------------------- cache loader
class CachedFeatureLoader:
    """Reads the Stage-4 `(1, T, D)` feature cache (+ masks) into one `SequenceData`."""

    def __init__(self, feature_store, speaker_parser: Optional[SpeakerParser] = None):
        self.store = feature_store
        self.speaker_parser = speaker_parser or SpeakerParser()

    def load(self, meta_df: pd.DataFrame, quality_df: Optional[pd.DataFrame] = None,
             selector: Optional[FeatureSelector] = None) -> SequenceData:
        report = self.store.ensure(meta_df)        # no-op for cached videos
        print(report.summary())
        fs = self.store.load_all(meta_df)
        if fs.masks is None:
            raise RuntimeError("FeatureSet has no masks; use the mask-aware FeatureStore")

        X = fs.features.detach().cpu().numpy().astype(np.float32)
        M = fs.masks.detach().cpu().numpy().astype(bool)
        video_ids, names = list(fs.video_ids), list(fs.feature_names)

        # labels come from meta_df by video_id (never filename, never FeatureSet.labels encoding)
        label_map = meta_df.set_index("video_id")["label"]
        labels = [label_map.get(v, "Unknown") for v in video_ids]
        keep = np.array([l in LABEL_TO_ID for l in labels])
        print(f"dropping {int((~keep).sum())} videos with Unknown label")

        if quality_df is not None and not quality_df.empty:
            passing = quality_df.groupby("video_id" if "video_id" in quality_df else "video")["quality_gate_pass"].all()
            ok = np.array([bool(passing.get(v, False)) for v in video_ids])
            print(f"quality gate removes {int((keep & ~ok).sum())} videos")
            keep &= ok

        idx = np.where(keep)[0]
        X = np.nan_to_num(X[idx], nan=0.0, posinf=0.0, neginf=0.0)
        M = M[idx]
        video_ids = [video_ids[i] for i in idx]
        y = np.array([LABEL_TO_ID[labels[i]] for i in idx], dtype=np.int64)

        if selector is not None:
            cols = selector.indices(names)
            X, names = X[..., cols], [names[c] for c in cols]

        print(f"valid frames: {M.mean():.1%} overall; "
              f"{int((~M.any(axis=1)).sum())} videos have no valid frames")
        return SequenceData(X, y, video_ids, [self.speaker_parser(v) for v in video_ids], names, M)


# --------------------------------------------------------------------------- normalisation
class Standardizer:
    """Per-feature z-score, fitted on valid frames of the training split only.

    Invalid frames are set to 0 after transform (they are replaced by the model's
    "missing" embedding anyway, so their value only needs to be finite).
    """

    def fit(self, data: SequenceData) -> "Standardizer":
        flat = data.X[data.mask]                     # (n_valid_frames, D)
        if len(flat) == 0:                           # nothing valid: fall back to all frames
            flat = data.X.reshape(-1, data.X.shape[-1])
        self.mean = flat.mean(0, keepdims=True).astype(np.float32)
        self.std = (flat.std(0, keepdims=True) + 1e-6).astype(np.float32)
        return self

    def transform(self, data: SequenceData) -> SequenceData:
        out = data.subset(np.arange(len(data)))
        X = ((data.X - self.mean) / self.std).astype(np.float32)
        X[~data.mask] = 0.0
        out.X = X
        return out


# --------------------------------------------------------------------------- torch dataset
class VisualSequenceDataset(Dataset):
    """Yields (x, y, i, mask) when return_mask=True (default), else the original (x, y, i)."""

    def __init__(self, data: SequenceData, train: bool = False, noise_std: float = 0.0,
                 feature_dropout: float = 0.0, return_mask: bool = True):
        self.data = data
        self.X = torch.from_numpy(data.X)
        self.y = torch.from_numpy(data.y)
        self.M = torch.from_numpy(data.mask)
        self.train, self.noise_std, self.feature_dropout = train, noise_std, feature_dropout
        self.return_mask = return_mask

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        x, m = self.X[i], self.M[i]
        if self.train:
            if self.noise_std > 0:
                x = x + torch.randn_like(x) * self.noise_std * m.unsqueeze(-1)   # noise on valid frames only
            if self.feature_dropout > 0:
                x = x * (torch.rand(1, x.shape[-1]) > self.feature_dropout).float()
        if self.return_mask:
            return x, self.y[i], i, m
        return x, self.y[i], i