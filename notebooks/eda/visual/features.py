"""Stage 4: cached (T, D) feature tensors per video, and a stacked FeatureSet."""

import inspect
import logging
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from .caching import DiskCache, params_key
from .status import StageReport, progress, run_tasks

logger = logging.getLogger(__name__)


@dataclass
class FeatureSet:
    features: "torch.Tensor"        # (N, T, D)
    feature_names: List[str]
    video_ids: List[str]
    labels: np.ndarray              # (N,) strings, e.g. 'Lie' / 'Truth'

    @property
    def emotion_labels(self) -> List[str]:
        return [n[len("emo_"):] for n in self.feature_names if n.startswith("emo_")]

    def emotion_probs(self) -> np.ndarray:
        """(N, T, K) emotion probabilities, selected by name rather than by position."""
        idx = [i for i, n in enumerate(self.feature_names) if n.startswith("emo_")]
        return self.features[:, :, idx].numpy()


class FeatureStore:
    """Builds extractor features from the cached OpenFace tables (no second OpenFace run)
    and caches one tensor per video. The cache key includes the extractor settings and the
    emotion model, so changing either recomputes instead of reusing stale tensors."""

    def __init__(self, extractor, tables, cache_dir: str, target_t: int = 90):
        self.extractor, self.tables, self.target_t = extractor, tables, target_t
        self._sig = self._signature()
        self.cache = DiskCache.torch(cache_dir, key_fn=lambda p: params_key(p, self._sig))
        self._accepts_df = "openface_df" in inspect.signature(extractor.extract_features).parameters
        if not self._accepts_df:
            logger.warning("extractor.extract_features has no 'openface_df' argument: OpenFace "
                           "will be re-run for every video. Add it to reuse the cached tables.")

    def _signature(self) -> dict:
        ex = self.extractor
        return {
            "target_t": self.target_t,
            "feature_names": list(ex.feature_names),
            "emotion_model": getattr(getattr(ex, "emotion", None), "model_id", None),
            **{k: getattr(ex, k, None) for k in
               ("min_confidence", "crop_faces", "crop_margin", "scale_features")},
        }

    def _run_one(self, task):
        video_id, path = task
        t0 = time.perf_counter()
        try:
            df = self.tables.load(path)
            if df is None or df.empty:
                return video_id, "no usable OpenFace table", time.perf_counter() - t0
            kwargs = {"openface_df": df} if self._accepts_df else {}
            feats = self.extractor.extract_features(path, target_t=self.target_t, **kwargs)
            self.cache.save(path, feats.cpu())
            return video_id, None, time.perf_counter() - t0
        except Exception as exc:
            return video_id, f"{type(exc).__name__}: {exc}", time.perf_counter() - t0

    def ensure(self, catalog_df: pd.DataFrame) -> StageReport:
        report = StageReport("features", total=len(catalog_df)).start()
        todo = []
        for r in catalog_df.itertuples():
            if self.cache.has(r.path):
                report.cached += 1
            else:
                todo.append((r.video_id, r.path))
        # workers=1: the emotion model already uses the GPU; parallelism wouldn't help.
        for video_id, err, elapsed in run_tasks(self._run_one, todo, workers=1, desc="Features"):
            if err:
                report.fail(video_id, err)
            else:
                report.computed += 1
                report.timings[video_id] = elapsed
        return report.finish()

    def load_all(self, catalog_df: pd.DataFrame) -> FeatureSet:
        import torch

        tensors, ids, labels = [], [], []
        for r in progress(list(catalog_df.itertuples()), desc="Loading tensors"):
            t = self.cache.load(r.path) if self.cache.has(r.path) else None
            if t is None:
                continue
            tensors.append(t)
            ids.append(r.video_id)
            labels.append(r.label)
        if not tensors:
            raise RuntimeError("No cached feature tensors found; run FeatureStore.ensure() first")
        return FeatureSet(torch.cat(tensors, dim=0), list(self.extractor.feature_names),
                          ids, np.array(labels))
