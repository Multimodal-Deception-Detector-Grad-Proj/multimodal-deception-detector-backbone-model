"""Stage 4: cached (T, D) feature tensors + (T,) validity masks per video, and a stacked FeatureSet."""

import logging
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import os ,sys
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
from notebooks.eda.visual.caching import DiskCache, params_key
from notebooks.eda.visual.status import StageReport, progress, run_tasks

logger = logging.getLogger(__name__)


@dataclass
class FeatureSet:
    features: "torch.Tensor"        # (N, T, D)
    feature_names: List[str]
    video_ids: List[str]
    labels: np.ndarray              # (N,) strings, e.g. 'Lie' / 'Truth'
    masks: Optional["torch.Tensor"] = None   # (N, T) bool, True = valid face frame

    @property
    def emotion_labels(self) -> List[str]:
        return [n[len("emo_"):] for n in self.feature_names if n.startswith("emo_")]

    def emotion_probs(self) -> np.ndarray:
        """(N, T, K) emotion probabilities, selected by name rather than by position."""
        idx = [i for i, n in enumerate(self.feature_names) if n.startswith("emo_")]
        return self.features[:, :, idx].numpy()

    @property
    def present(self) -> "torch.Tensor":
        """(N,) bool: clip has at least one valid frame (use for modality dropout / gating)."""
        if self.masks is None:
            raise ValueError("FeatureSet has no masks")
        return self.masks.any(dim=1)

    def valid_fraction(self) -> np.ndarray:
        """(N,) fraction of valid frames per clip; handy for spotting bad videos."""
        if self.masks is None:
            raise ValueError("FeatureSet has no masks")
        return self.masks.float().mean(dim=1).numpy()


class FeatureStore:
    """Builds extractor features from the cached OpenFace tables (no second OpenFace run)
    and caches one feature tensor and one validity mask per video. The cache key includes the
    extractor settings and the emotion model, so changing either recomputes instead of reusing
    stale tensors. Masks live in a sibling cache dir (``<cache_dir>_masks``) under the same key."""

    def __init__(self, extractor, tables, cache_dir: str, target_t: int = 90):
        self.extractor, self.tables, self.target_t = extractor, tables, target_t
        if not hasattr(extractor, "extract_features_with_mask"):
            raise TypeError("extractor must provide extract_features_with_mask(video_path, target_t, "
                            "openface_df) -> (features, mask); update VisualFeatureExtractor.")
        self._sig = self._signature()
        key_fn = lambda p: params_key(p, self._sig)
        self.cache = DiskCache.torch(cache_dir, key_fn=key_fn)
        self.mask_cache = DiskCache.torch(str(cache_dir).rstrip("/\\") + "_masks", key_fn=key_fn)

    def _signature(self) -> dict:
        ex = self.extractor
        return {
            "target_t": self.target_t,
            "feature_names": list(ex.feature_names),
            "emotion_model": getattr(getattr(ex, "emotion", None), "model_id", None),
            **{k: getattr(ex, k, None) for k in
               ("min_confidence", "crop_faces", "crop_margin", "scale_features")},
        }

    def _is_cached(self, path) -> bool:
        return self.cache.has(path) and self.mask_cache.has(path)

    def _run_one(self, task):
        video_id, path = task
        t0 = time.perf_counter()
        try:
            df = self.tables.load(path)
            if df is None or df.empty:
                return video_id, "no usable OpenFace table", time.perf_counter() - t0
            feats, mask = self.extractor.extract_features_with_mask(
                path, target_t=self.target_t, openface_df=df)
            self.cache.save(path, feats.cpu())
            self.mask_cache.save(path, mask.cpu())
            return video_id, None, time.perf_counter() - t0
        except Exception as exc:
            return video_id, f"{type(exc).__name__}: {exc}", time.perf_counter() - t0

    def ensure(self, catalog_df: pd.DataFrame) -> StageReport:
        report = StageReport("features", total=len(catalog_df)).start()
        todo = []
        for r in catalog_df.itertuples():
            if self._is_cached(r.path):
                report.cached += 1
            else:
                # Includes videos cached before masks existed: they are recomputed once.
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

        tensors, masks, ids, labels = [], [], [], []
        for r in progress(list(catalog_df.itertuples()), desc="Loading tensors"):
            if not self._is_cached(r.path):
                continue
            t, m = self.cache.load(r.path), self.mask_cache.load(r.path)
            if t is None or m is None:
                continue
            tensors.append(t)
            masks.append(m.bool())
            ids.append(r.video_id)
            labels.append(r.label)
        if not tensors:
            raise RuntimeError("No cached feature tensors found; run FeatureStore.ensure() first")
        return FeatureSet(torch.cat(tensors, dim=0), list(self.extractor.feature_names),
                          ids, np.array(labels), masks=torch.cat(masks, dim=0))