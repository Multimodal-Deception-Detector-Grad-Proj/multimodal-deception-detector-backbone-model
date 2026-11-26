"""Stage 2: scene boundaries per video (1-based inclusive frame ranges, like OpenFace)."""

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .caching import DiskCache, params_key
from .status import StageReport, run_tasks

logger = logging.getLogger(__name__)

Scenes = List[List[int]]


def _detect_content(task):
    """Worker: PySceneDetect ContentDetector. Module-level so spawn workers can import it."""
    video_id, path, threshold, min_scene_len, frame_skip, fallback_frames = task
    t0 = time.perf_counter()
    try:
        import cv2

        cv2.setNumThreads(1)  # one decoder thread per worker; avoids CPU oversubscription
        from scenedetect import ContentDetector, SceneManager, open_video

        manager = SceneManager()
        manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len))
        manager.detect_scenes(open_video(path), frame_skip=frame_skip)
        scenes = [[int(s.get_frames()) + 1, int(e.get_frames())] for s, e in manager.get_scene_list()]
        if not scenes:
            scenes = [[1, max(int(fallback_frames), 1)]]
        return video_id, path, scenes, None, time.perf_counter() - t0
    except Exception as exc:
        return video_id, path, None, f"{type(exc).__name__}: {exc}", time.perf_counter() - t0


def scenes_from_pose_jumps(
    df: Optional[pd.DataFrame], jump_mm: float, min_scene_len: int,
    total_frames: int, max_gap: int = 5,
) -> Scenes:
    """Cheap alternative: cut where head translation jumps between consecutive tracked
    frames (what a camera cut looks like to OpenFace). Needs no video decoding."""
    whole = [[1, max(int(total_frames), 1)]]
    cols = ["frame", "pose_Tx", "pose_Ty", "pose_Tz"]
    if df is None or df.empty or not set(cols) <= set(df.columns):
        return whole
    d = df.copy()
    if "success" in d.columns:
        d = d[pd.to_numeric(d["success"], errors="coerce") > 0.5]
    d = d[cols].apply(pd.to_numeric, errors="coerce").dropna().sort_values("frame")
    if len(d) < 2:
        return whole
    frames = d["frame"].to_numpy(int)
    step = np.linalg.norm(np.diff(d[cols[1:]].to_numpy(float), axis=0), axis=1)
    close = np.diff(frames) <= max_gap
    starts = [1]
    for i in np.where((step > jump_mm) & close)[0] + 1:
        if frames[i] - starts[-1] >= min_scene_len:
            starts.append(int(frames[i]))
    last = max(int(frames.max()), int(total_frames), 1)
    return [[s, (starts[i + 1] - 1) if i + 1 < len(starts) else last] for i, s in enumerate(starts)]


class SceneDetector:
    """``method``: 'content' (PySceneDetect, cached, parallel spawn workers),
    'pose_jump' (derived from the cached OpenFace tables; instant, experimental),
    or 'none' (one scene per video)."""

    METHODS = ("content", "pose_jump", "none")

    def __init__(
        self, cache_dir: str, method: str = "content", threshold: float = 27.0,
        min_scene_len: int = 15, frame_skip: int = 2, workers: int = 4, jump_mm: float = 50.0,
    ):
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}")
        self.method, self.threshold, self.min_scene_len = method, threshold, min_scene_len
        self.frame_skip, self.workers, self.jump_mm = frame_skip, workers, jump_mm
        params = {"method": "content", "threshold": threshold,
                  "min_scene_len": min_scene_len, "frame_skip": frame_skip}
        self.cache = DiskCache.json(cache_dir, key_fn=lambda p: params_key(p, params))

    def ensure(self, catalog_df: pd.DataFrame, tables=None) -> Tuple[Dict[str, Scenes], StageReport]:
        """Return ({video_id: scenes}, report). ``tables`` (OpenFaceTableStore) is only
        needed for method='pose_jump'."""
        report = StageReport(f"scenes[{self.method}]", total=len(catalog_df)).start()
        out: Dict[str, Scenes] = {}

        if self.method == "none":
            for r in catalog_df.itertuples():
                out[r.video_id] = [[1, max(int(r.frame_count), 1)]]
            report.computed = len(out)
            return out, report.finish()

        if self.method == "pose_jump":
            if tables is None:
                raise ValueError("method='pose_jump' needs the OpenFaceTableStore (tables=...)")
            for r in catalog_df.itertuples():
                out[r.video_id] = scenes_from_pose_jumps(
                    tables.load(r.path), self.jump_mm, self.min_scene_len, int(r.frame_count)
                )
            report.computed = len(out)
            return out, report.finish()

        todo = []
        for r in catalog_df.itertuples():
            cached = self.cache.load(r.path) if self.cache.has(r.path) else None
            if cached is not None:
                out[r.video_id] = cached["scenes"]
                report.cached += 1
            else:
                todo.append((r.video_id, r.path, self.threshold, self.min_scene_len,
                             self.frame_skip, int(r.frame_count)))

        def _crash(task, exc):
            return task[0], task[1], None, f"worker crashed: {type(exc).__name__}: {exc}", 0.0

        for video_id, path, scenes, err, elapsed in run_tasks(
            _detect_content, todo, workers=self.workers, desc="Scene detection",
            processes=True, on_error=_crash,
        ):
            if err:
                report.fail(video_id, err)
                continue
            self.cache.save(path, {"scenes": scenes, "elapsed_sec": round(elapsed, 3)})
            out[video_id] = scenes
            report.computed += 1
            report.timings[video_id] = elapsed
        return out, report.finish()
