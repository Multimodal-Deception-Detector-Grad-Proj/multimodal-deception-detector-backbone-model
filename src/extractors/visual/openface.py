"""Run OpenFace and turn its CSV output into a clean feature matrix."""

import os
import shutil
import subprocess

import numpy as np
import pandas as pd

from .constants import AU_INTENSITY, AU_PRESENCE, GAZE, OPENFACE_COLS, POSE


def find_binary(openface_bin_path: str) -> str:
    """Resolve the OpenFace executable or raise."""
    path = shutil.which(openface_bin_path)
    if not path:
        raise FileNotFoundError(f"OpenFace binary not found: {openface_bin_path!r}")
    return path


def run_openface(
    openface_bin: str,
    video_path: str,
    output_dir: str,
    timeout: float,
    landmarks: bool = False,
) -> pd.DataFrame:
    """Run OpenFace and return its output CSV."""
    cmd = [
        openface_bin, "-f", video_path, "-out_dir", output_dir,
        "-pose", "-gaze", "-aus", "-q",
    ]
    if landmarks:
        cmd.append("-2Dfp")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"OpenFace timed out after {timeout}s: {video_path}") from exc
    if result.returncode:
        raise RuntimeError(f"OpenFace failed: {result.stderr}")

    name = os.path.splitext(os.path.basename(video_path))[0]
    csv_path = os.path.join(output_dir, f"{name}.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"OpenFace CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return df


def compute_validity(df: pd.DataFrame, min_confidence: float) -> np.ndarray:
    """Mark frames with a successful, sufficiently confident face track."""
    required = ["success", "confidence"]
    missing = [c for c in required if c not in df]
    if missing:
        raise KeyError(f"Missing OpenFace columns: {missing}")
    return (
        (df["success"].to_numpy(dtype=float) > 0.5)
        & (df["confidence"].to_numpy(dtype=float) >= min_confidence)
    )


def clean_openface(df: pd.DataFrame, valid: np.ndarray, scale_features: bool) -> np.ndarray:
    """Mask failed tracks, interpolate continuous features, and scale values."""
    missing = [c for c in OPENFACE_COLS if c not in df]
    if missing:
        raise KeyError(f"Missing OpenFace columns: {missing}")

    feats = df[OPENFACE_COLS].astype(float).copy()
    feats.loc[~valid] = np.nan

    if valid.any():
        continuous = AU_INTENSITY + POSE + GAZE
        feats[continuous] = feats[continuous].interpolate(limit_direction="both")
        feats[AU_PRESENCE] = feats[AU_PRESENCE].ffill().bfill()
    feats = feats.fillna(0.0)

    if scale_features:
        feats[AU_INTENSITY] /= 5.0
        feats[["pose_Tx", "pose_Ty", "pose_Tz"]] /= 1000.0
    return feats.to_numpy(dtype=np.float32)