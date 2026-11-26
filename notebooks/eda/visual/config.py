"""Single place for every knob the visual EDA uses."""

import os
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class EDAConfig:
    # data
    data_dir: str = "../data"
    hf_repo_id: str = "dandyidk/Multi-Modal-Lie-Detection"
    hf_folders: Tuple[str, ...] = ("deception", "truth")
    # models / hardware
    openface_bin: str = "FeatureExtraction"
    device: str = "cuda"
    # OpenFace stage
    openface_workers: int = min(4, os.cpu_count() or 1)
    # scene stage: "content" (PySceneDetect), "pose_jump" (from OpenFace table), "none"
    scene_method: str = "content"
    scene_threshold: float = 27.0
    scene_min_len: int = 30 #15
    scene_frame_skip: int = 2
    scene_workers: int = min(4, os.cpu_count() or 1)
    pose_jump_mm: float = 50.0
    # quality gate
    conf_thresh: float = 0.75
    jitter_thresh: float = 20.0
    max_dropout_pct: float = 15.0
    # chunking analysis
    chunk_length_sec: float = 8.0
    overlap_sec: float = 2.0
    # feature tensors
    target_t: int = 200 #90
    # None -> <data_dir>/<default name>
    openface_cache_dir: Optional[str] = None
    scene_cache_dir: Optional[str] = None
    feature_cache_dir: Optional[str] = None
    quality_report_path: Optional[str] = None

    def __post_init__(self):
        d = self.data_dir
        self.openface_cache_dir = self.openface_cache_dir or os.path.join(d, "openface_cache")
        self.scene_cache_dir = self.scene_cache_dir or os.path.join(d, "scene_cache_v2")
        self.feature_cache_dir = self.feature_cache_dir or os.path.join(d, "feature_cache")
        self.quality_report_path = self.quality_report_path or os.path.join(
            d, "openface_quality_report.csv"
        )
