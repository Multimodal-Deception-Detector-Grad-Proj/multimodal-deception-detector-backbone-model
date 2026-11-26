"""Single source of truth for the visual modality: EDA/extraction knobs + training knobs.

Replaces the separate EDAConfig and TrainConfig. Both old names are kept as aliases at the
bottom, so existing imports (`from ... import EDAConfig` / `TrainConfig`) keep working.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class VisualConfig:
    # ------------------------------------------------------------------ data
    data_dir: str = "../data"
    hf_repo_id: str = "dandyidk/Multi-Modal-Lie-Detection"
    hf_folders: Tuple[str, ...] = ("deception", "truth")

    # ------------------------------------------------------------------ models / hardware
    openface_bin: str = "FeatureExtraction"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------ OpenFace stage
    openface_workers: int = min(4, os.cpu_count() or 1)

    # ------------------------------------------------------------------ scene stage
    # scene_method: "content" (PySceneDetect), "pose_jump" (from OpenFace table), "none"
    scene_method: str = "content"
    scene_threshold: float = 27.0
    scene_min_len: int = 30
    scene_frame_skip: int = 2
    scene_workers: int = min(4, os.cpu_count() or 1)
    pose_jump_mm: float = 50.0

    # ------------------------------------------------------------------ quality gate
    conf_thresh: float = 0.75
    jitter_thresh: float = 20.0
    max_dropout_pct: float = 15.0

    # ------------------------------------------------------------------ chunking analysis
    chunk_length_sec: float = 8.0
    overlap_sec: float = 2.0

    # ------------------------------------------------------------------ feature tensors
    target_t: int = 200
    openface_cache_dir: Optional[str] = None    # None -> <data_dir>/openface_cache
    scene_cache_dir: Optional[str] = None       # None -> <data_dir>/scene_cache_v2
    feature_cache_dir: Optional[str] = None     # None -> <data_dir>/feature_cache
    quality_report_path: Optional[str] = None   # None -> <data_dir>/openface_quality_report.csv

    # ------------------------------------------------------------------ training: data
    use_quality_gate: bool = False              # drop videos with any scene failing the Stage-3 gate
    include_prefixes: Optional[Tuple[str, ...]] = None   # e.g. ("au_", "gaze_"); None = all features
    exclude_prefixes: Optional[Tuple[str, ...]] = None   # e.g. ("emo_",) for an ablation
    val_frac: float = 0.15
    test_frac: float = 0.15
    batch_size: int = 32

    # ------------------------------------------------------------------ training: masking
    use_mask: bool = True                       # True = model ignores invalid-face frames; False = old behaviour

    # ------------------------------------------------------------------ training: model
    rnn_type: str = "gru"                       # "lstm" | "gru"
    hidden_dim: int = 64
    num_layers: int = 1
    bidirectional: bool = False
    dropout: float = 0.3
    pooling: str = "attention"                  # "attention" | "mean" | "last"

    # ------------------------------------------------------------------ training: optimisation
    lr: float = 1e-3
    weight_decay: float = 1e-2
    epochs: int = 100
    patience: int = 14                          # early stopping on val macro-F1
    grad_clip: float = 1.0
    class_weighted_loss: bool = True
    noise_std: float = 0.05                     # train-time gaussian noise (standardised features, valid frames)
    feature_dropout: float = 0.1                # train-time random feature-channel dropout

    # ------------------------------------------------------------------ misc
    seed: int = 42
    output_dir: str = "../outputs/visual_training"
    num_workers: int = 0

    def __post_init__(self):
        d = self.data_dir
        self.openface_cache_dir = self.openface_cache_dir or os.path.join(d, "openface_cache")
        self.scene_cache_dir = self.scene_cache_dir or os.path.join(d, "scene_cache_v2")
        self.feature_cache_dir = self.feature_cache_dir or os.path.join(d, "feature_cache")
        self.quality_report_path = self.quality_report_path or os.path.join(
            d, "openface_quality_report.csv")


# Backward-compatible names: both point at the same class.
EDAConfig = VisualConfig
TrainConfig = VisualConfig
