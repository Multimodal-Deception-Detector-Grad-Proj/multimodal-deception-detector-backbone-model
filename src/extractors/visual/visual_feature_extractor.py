"""Extract OpenFace visual features with a stable team tensor contract."""

import logging
import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np
import pandas as pd
import torch

from ...common.constants import TARGET_T

logger = logging.getLogger(__name__)


class VisualFeatureExtractor:
    """Extract OpenFace AUs, pose, gaze and tracking-quality features."""

    POSE = [
        "pose_Tx", "pose_Ty", "pose_Tz",
        "pose_Rx", "pose_Ry", "pose_Rz",
    ]
    GAZE = [
        "gaze_0_x", "gaze_0_y", "gaze_0_z",
        "gaze_1_x", "gaze_1_y", "gaze_1_z",
        "gaze_angle_x", "gaze_angle_y",
    ]
    AU_INTENSITY = [
        "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r",
        "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r",
        "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r",
    ]
    AU_PRESENCE = [
        "AU01_c", "AU02_c", "AU04_c", "AU05_c", "AU06_c", "AU07_c",
        "AU09_c", "AU10_c", "AU12_c", "AU14_c", "AU15_c", "AU17_c",
        "AU20_c", "AU23_c", "AU25_c", "AU26_c", "AU28_c", "AU45_c",
    ]
    OPENFACE_COLS = AU_INTENSITY + AU_PRESENCE + POSE + GAZE
    QUALITY_COLS = ["valid", "confidence"]

    def __init__(
        self,
        custom_weights_path=None,
        openface_bin_path="FeatureExtraction",
        device=None,
        min_confidence=0.75,
        crop_faces=False,
        crop_margin=0.2,
        include_quality_flags=True,
        add_pose_delta=False,
        scale_features=True,
        batch_size=32,
        openface_timeout=300.0,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.min_confidence = min_confidence
        self.device = device
        self.include_quality_flags = include_quality_flags
        self.add_pose_delta = add_pose_delta
        self.scale_features = scale_features
        self.openface_timeout = openface_timeout

        self.openface_bin = shutil.which(openface_bin_path)
        if not self.openface_bin and os.path.isfile(openface_bin_path):
            self.openface_bin = os.path.abspath(openface_bin_path)
        if not self.openface_bin:
            raise FileNotFoundError(
                f"OpenFace binary not found: {openface_bin_path!r}"
            )

        if custom_weights_path:
            logger.warning(
                "Emotion weights were supplied but are not used. "
                "Emotion features are disabled until a trained model is integrated."
            )

    @property
    def feature_names(self):
        names = self.OPENFACE_COLS.copy()
        if self.include_quality_flags:
            names += self.QUALITY_COLS
        if self.add_pose_delta:
            names += [f"d_{c}" for c in self.POSE]
        return names

    @property
    def feature_dim(self):
        return len(self.feature_names)

    def run_openface(self, video_path, output_dir, landmarks=False):
        cmd = [
            self.openface_bin, "-f", video_path, "-out_dir", output_dir,
            "-pose", "-gaze", "-aus", "-q",
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=self.openface_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"OpenFace timed out after {self.openface_timeout}s: {video_path}"
            ) from exc
        if result.returncode:
            raise RuntimeError(f"OpenFace failed: {result.stderr}")

        name = os.path.splitext(os.path.basename(video_path))[0]
        csv_path = os.path.join(output_dir, f"{name}.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"OpenFace CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        return df

    def _validity(self, df):
        required = ["success", "confidence"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"Missing OpenFace columns: {missing}")
        return (
            (df["success"].to_numpy(dtype=float) > 0.5)
            & (df["confidence"].to_numpy(dtype=float) >= self.min_confidence)
        )

    def _clean_openface(self, df, valid):
        missing = [c for c in self.OPENFACE_COLS if c not in df.columns]
        if missing:
            raise KeyError(f"Missing OpenFace columns: {missing}")

        feats = df[self.OPENFACE_COLS].apply(
            pd.to_numeric, errors="coerce"
        ).copy()
        feats.loc[~valid] = np.nan

        if valid.any():
            continuous = self.AU_INTENSITY + self.POSE + self.GAZE
            feats[continuous] = feats[continuous].interpolate(
                limit_direction="both"
            )
            feats[self.AU_PRESENCE] = feats[self.AU_PRESENCE].ffill().bfill()

        feats = feats.fillna(0.0)
        if self.scale_features:
            feats[self.AU_INTENSITY] /= 5.0
            feats[["pose_Tx", "pose_Ty", "pose_Tz"]] /= 1000.0
        return feats.to_numpy(dtype=np.float32)

    @staticmethod
    def _video_frame_count(video_path):
        cap = cv2.VideoCapture(video_path)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return max(count, 0)

    def extract_features(self, video_path, target_t=TARGET_T):
        """Return a finite FloatTensor with shape (1, target_t, D_VIS)."""
        with tempfile.TemporaryDirectory() as tmp:
            df = self.run_openface(video_path, tmp)

        if len(df):
            valid = self._validity(df)
            cleaned = self._clean_openface(df, valid)
            rows = np.linspace(0, len(df) - 1, target_t).round().astype(int)
            of_feats = cleaned[rows]
            valid_s = valid[rows].astype(np.float32)
            conf_s = np.clip(
                df["confidence"].to_numpy(dtype=np.float32)[rows], 0, 1
            )
        else:
            logger.warning("%s: OpenFace returned no rows", video_path)
            of_feats = np.zeros(
                (target_t, len(self.OPENFACE_COLS)), dtype=np.float32
            )
            valid_s = np.zeros(target_t, dtype=np.float32)
            conf_s = np.zeros(target_t, dtype=np.float32)

        parts = [of_feats]
        if self.include_quality_flags:
            parts.append(np.column_stack((valid_s, conf_s)))

        if self.add_pose_delta:
            start = self.OPENFACE_COLS.index(self.POSE[0])
            pose = of_feats[:, start:start + len(self.POSE)]
            parts.append(
                np.diff(pose, axis=0, prepend=pose[:1]).astype(np.float32)
            )

        features = torch.from_numpy(
            np.concatenate(parts, axis=1).astype(np.float32)
        ).unsqueeze(0)

        if not torch.isfinite(features).all():
            features = torch.nan_to_num(
                features, nan=0.0, posinf=0.0, neginf=0.0
            )

        expected = (1, target_t, self.feature_dim)
        if tuple(features.shape) != expected:
            raise ValueError(
                f"Expected shape {expected}, got {tuple(features.shape)}"
            )
        return features
