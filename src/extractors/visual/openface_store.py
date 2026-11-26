"""Stage 1: run OpenFace once per video and cache the resulting table."""

import logging
import tempfile
import time
from typing import Optional
import os
import sys
import pandas as pd
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
from notebooks.eda.visual.caching import DiskCache, legacy_key
from notebooks.eda.visual.status import StageReport, run_tasks

logger = logging.getLogger(__name__)


class OpenFaceTableStore:
    """Cached OpenFace tables (with 2D landmarks).

    OpenFace is a subprocess, so several videos can run at once with threads
    (``workers``). Videos where OpenFace finds no face are cached as an empty
    table so they are not retried on every run.
    """

    def __init__(self, extractor, cache_dir: str, workers: int = 4, landmarks: bool = True):
        self.extractor, self.workers, self.landmarks = extractor, workers, landmarks
        self.cache = DiskCache.pickle(cache_dir, key_fn=legacy_key)

    def has(self, path: str) -> bool:
        return self.cache.has(path)

    def load(self, path: str) -> Optional[pd.DataFrame]:
        return self.cache.load(path)

    def _run_one(self, task):
        video_id, path = task
        t0 = time.perf_counter()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                df = self.extractor.run_openface(path, tmp, landmarks=self.landmarks)
            df = pd.DataFrame() if df is None else df.copy()
            if len(df.columns):
                df.columns = [str(c).strip() for c in df.columns]
            self.cache.save(path, df)
            return video_id, df, None, time.perf_counter() - t0
        except Exception as exc:
            return video_id, None, f"{type(exc).__name__}: {exc}", time.perf_counter() - t0

    def ensure(self, catalog_df: pd.DataFrame, verify: bool = False) -> StageReport:
        """Compute missing tables. Safe to interrupt and re-run: finished videos are kept.

        ``verify=True`` also opens every cached file and recomputes unreadable ones.
        """
        report = StageReport("openface", total=len(catalog_df)).start()
        todo = []
        for row in catalog_df.itertuples():
            if self.cache.has(row.path) and not (verify and self.cache.load(row.path) is None):
                report.cached += 1
            else:
                todo.append((row.video_id, row.path))

        for video_id, df, err, elapsed in run_tasks(
            self._run_one, todo, workers=self.workers, desc="OpenFace"
        ):
            if err:
                report.fail(video_id, err)
                continue
            report.computed += 1
            report.timings[video_id] = elapsed
            if df.empty:
                report.fail(video_id, "OpenFace returned no rows (cached as empty)")
        return report.finish()
