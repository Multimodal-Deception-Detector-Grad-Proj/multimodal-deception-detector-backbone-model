"""Speaker-disjoint, label-stratified train/val/test split."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .dataset import SequenceData


class SpeakerSplitter:
    def __init__(self, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42):
        self.val_frac, self.test_frac, self.seed = val_frac, test_frac, seed

    @staticmethod
    def _carve(y, groups, frac, seed):
        k = max(2, int(round(1 / frac)))
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        rest, held = next(sgkf.split(np.zeros(len(y)), y, groups))
        return rest, held

    def split(self, data: SequenceData) -> Dict[str, np.ndarray]:
        y, groups = data.y, np.array(data.speakers)
        all_idx = np.arange(len(y))

        rest, test = self._carve(y, groups, self.test_frac, self.seed)
        val_rel = self.val_frac / (1 - self.test_frac)
        tr_rel, va_rel = self._carve(y[rest], groups[rest], val_rel, self.seed + 1)
        splits = {"train": rest[tr_rel], "val": rest[va_rel], "test": test}

        sp = {k: set(groups[v]) for k, v in splits.items()}
        assert not (sp["train"] & sp["val"] or sp["train"] & sp["test"] or sp["val"] & sp["test"]), "speaker leakage"
        assert sum(len(v) for v in splits.values()) == len(all_idx)
        return splits

    @staticmethod
    def summary(data: SequenceData, splits: Dict[str, np.ndarray]) -> pd.DataFrame:
        rows = []
        for name, idx in splits.items():
            y = data.y[idx]
            rows.append({"split": name, "videos": len(idx), "speakers": len(set(np.array(data.speakers)[idx])),
                         "truth": int((y == 0).sum()), "lie": int((y == 1).sum()),
                         "lie_ratio": round(float(y.mean()), 3)})
        return pd.DataFrame(rows).set_index("split")
