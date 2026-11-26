"""Combine OpenFace AUs/pose/gaze with ViT emotion probabilities."""

import logging
import tempfile
from typing import List, Optional, Sequence

import numpy as np
import torch
import pandas as pd

from . import constants as C
from .emotion import DEFAULT_MODEL_ID, EmotionRecognizer
from .faces import face_boxes
from .openface import clean_openface, compute_validity, find_binary, run_openface
from .video import video_frame_count

logger = logging.getLogger(__name__)


class VisualFeatureExtractor:
    """Frame-level OpenFace features + facial emotion probabilities."""

    # Kept as class attributes for backward compatibility.
    POSE = C.POSE
    GAZE = C.GAZE
    AU_INTENSITY = C.AU_INTENSITY
    AU_PRESENCE = C.AU_PRESENCE
    OPENFACE_COLS = C.OPENFACE_COLS
    QUALITY_COLS = C.QUALITY_COLS
    LANDMARK_X = C.LANDMARK_X
    LANDMARK_Y = C.LANDMARK_Y

    def __init__(
        self,
        emotion_model_id=DEFAULT_MODEL_ID,
        openface_bin_path="FeatureExtraction",
        device=None,
        min_confidence=0.75,
        crop_faces=True,
        crop_margin=0.2,
        include_quality_flags=True,
        add_pose_delta=False,
        scale_features=True,
        batch_size=32,
        openface_timeout=300.0,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.min_confidence = min_confidence
        self.crop_faces = crop_faces
        self.crop_margin = crop_margin
        self.include_quality_flags = include_quality_flags
        self.add_pose_delta = add_pose_delta
        self.scale_features = scale_features
        self.batch_size = batch_size
        self.openface_timeout = openface_timeout

        self.openface_bin = find_binary(openface_bin_path)
        self.emotion = EmotionRecognizer(
            emotion_model_id, device=self.device, batch_size=batch_size
        )

    # -- properties --------------------------------------------------------

    @property
    def emotion_labels(self) -> List[str]:
        return self.emotion.labels

    @property
    def feature_names(self) -> List[str]:
        names = self.OPENFACE_COLS.copy()
        if self.include_quality_flags:
            names += self.QUALITY_COLS
        if self.add_pose_delta:
            names += [f"d_{c}" for c in self.POSE]
        return names + [f"emo_{label}" for label in self.emotion_labels]

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    # -- thin delegates (public API unchanged) -----------------------------

    def run_openface(self, video_path: str, output_dir: str, landmarks=False):
        return run_openface(
            self.openface_bin, video_path, output_dir, self.openface_timeout, landmarks
        )

    def extract_emotion_probs(
        self, video_path: str, frame_indices: Sequence[int],
        boxes: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        return self.emotion.predict(video_path, frame_indices, boxes)

    # -- main entry point --------------------------------------------------

    def extract_features(self, video_path: str, target_t: int = 90, openface_df: Optional[pd.DataFrame] = None,) -> torch.FloatTensor:
        """Return visual features with shape (1, target_t, feature_dim)."""
        if openface_df is not None:
            df = openface_df
        with tempfile.TemporaryDirectory() as tmp:
            df = self.run_openface(video_path, tmp, landmarks=self.crop_faces)

        if len(df):
            valid = compute_validity(df, self.min_confidence)
            if not valid.any():
                logger.warning("%s: no valid face tracks; OpenFace features are zero", video_path)

            cleaned = clean_openface(df, valid, self.scale_features)
            rows = np.linspace(0, len(df) - 1, target_t).round().astype(int)
            of_feats = cleaned[rows]
            valid_s = valid[rows].astype(np.float32)
            conf_s = np.clip(df["confidence"].to_numpy(np.float32)[rows], 0, 1)
            frame_idx = (
                df["frame"].to_numpy(dtype=int)[rows] - 1 if "frame" in df
                else rows.copy()
            )
            frame_idx = np.clip(frame_idx, 0, None)
            boxes = face_boxes(df, valid, self.crop_margin)
            boxes = boxes[rows] if self.crop_faces and boxes is not None else None
        else:
            logger.warning("%s: OpenFace returned no rows", video_path)
            of_feats = np.zeros((target_t, len(self.OPENFACE_COLS)), dtype=np.float32)
            valid_s = conf_s = np.zeros(target_t, dtype=np.float32)
            total = video_frame_count(video_path)
            frame_idx = (
                np.linspace(0, total - 1, target_t).round().astype(int)
                if total else np.zeros(target_t, dtype=int)
            )
            boxes = None

        parts = [of_feats]
        if self.include_quality_flags:
            parts.append(np.column_stack((valid_s, conf_s)))
        if self.add_pose_delta:
            start = self.OPENFACE_COLS.index(self.POSE[0])
            pose = of_feats[:, start:start + len(self.POSE)]
            parts.append(np.diff(pose, axis=0, prepend=pose[:1]).astype(np.float32))

        tabular = torch.from_numpy(np.concatenate(parts, axis=1).astype(np.float32))
        emotion = self.extract_emotion_probs(video_path, frame_idx, boxes)
        features = torch.cat((tabular, emotion), dim=-1).unsqueeze(0)

        if not torch.isfinite(features).all():
            logger.warning("%s: replacing non-finite features with zero", video_path)
            features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        expected = (1, target_t, self.feature_dim)
        if tuple(features.shape) != expected:
            raise ValueError(f"Expected shape {expected}, got {tuple(features.shape)}")
        return features