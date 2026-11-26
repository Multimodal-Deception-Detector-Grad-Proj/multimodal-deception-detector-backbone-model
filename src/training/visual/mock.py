"""Synthetic stand-in for `CachedFeatureLoader`, for pipeline smoke tests only (never for results).

Same `.load(...)` contract as the real loader, so the rest of the pipeline is untouched.
Lies get a weak shift on a few channels; each speaker also has a nuisance offset, so a model that
memorises speakers fails on the speaker-disjoint split, as it would on real data.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .dataset import FeatureSelector, SequenceData


class MockFeatureLoader:
    def __init__(self, n_speakers: int = 24, videos_per_speaker: Tuple[int, int] = (6, 12), T: int = 90,
                 n_au: int = 17, n_gaze: int = 6, n_pose: int = 6, n_emo: int = 7,
                 signal: float = 0.6, seed: int = 0):
        self.n_speakers, self.vps, self.T = n_speakers, videos_per_speaker, T
        self.counts = dict(au_=n_au, gaze_=n_gaze, pose_=n_pose, emo_=n_emo)
        self.signal, self.seed = signal, seed

    def feature_names(self) -> List[str]:
        return [f"{p}{i}" for p, n in self.counts.items() for i in range(n)]

    def load(self, meta_df=None, quality_df=None, selector: Optional[FeatureSelector] = None) -> SequenceData:
        rng = np.random.default_rng(self.seed)
        names = self.feature_names()
        D = len(names)
        informative = rng.choice(D, size=max(3, D // 6), replace=False)

        X, y, vids, spk = [], [], [], []
        for s in range(self.n_speakers):
            name = f"Speaker{s:02d}"
            offset = rng.normal(0, 0.5, size=D)                       # nuisance, speaker-specific
            for k in range(int(rng.integers(self.vps[0], self.vps[1] + 1))):
                label = int(rng.integers(0, 2))
                x = offset + rng.normal(0, 1, size=(self.T, D))
                x += np.cumsum(rng.normal(0, 0.05, size=(self.T, D)), axis=0)   # slow temporal drift
                if label:
                    x[:, informative] += self.signal * np.sin(np.linspace(0, 6, self.T))[:, None]
                X.append(x.astype(np.float32)); y.append(label)
                folder = "deception" if label else "truth"
                vids.append(f"{folder}/MOCK/{name}/clip{k:02d}.mp4"); spk.append(name)

        X, y = np.stack(X), np.array(y, dtype=np.int64)
        if selector is not None:
            cols = selector.indices(names)
            X, names = X[..., cols], [names[c] for c in cols]
        return SequenceData(X, y, vids, spk, names)
