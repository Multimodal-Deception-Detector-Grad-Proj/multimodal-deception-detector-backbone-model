"""Stage 3: per-scene OpenFace quality metrics and the pass/fail gate."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .status import StageReport, progress

POS = ["pose_Tx", "pose_Ty", "pose_Tz"]
ROT = ["pose_Rx", "pose_Ry", "pose_Rz"]
GAZE = ["gaze_angle_x", "gaze_angle_y"]


@dataclass
class QualityThresholds:
    conf_thresh: float = 0.75       # a frame is "valid" at or above this confidence
    jitter_thresh: float = 5.0      # max RMS frame-to-frame translation (mm) per axis
    max_dropout_pct: float = 15.0   # max % of frames below conf_thresh


class QualityAuditor:
    """Computes raw metrics only; the gate is applied afterwards from the thresholds, so
    changing a threshold never requires recomputing (and can never serve stale pass/fail)."""

    def __init__(self, thresholds: Optional[QualityThresholds] = None):
        self.th = thresholds or QualityThresholds()

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["frame"] = (
            pd.to_numeric(df["frame"], errors="coerce") if "frame" in df.columns
            else np.arange(1, len(df) + 1)
        )
        conf = df["confidence"] if "confidence" in df.columns else 0.0
        df["confidence"] = pd.to_numeric(conf, errors="coerce").fillna(0.0)
        for col in POS + ROT + GAZE:
            if col not in df.columns:
                df[col] = np.nan
        return df

    @staticmethod
    def _jitter(valid_scene: pd.DataFrame) -> np.ndarray:
        """RMS of frame-to-frame translation, using only truly consecutive frames."""
        if len(valid_scene) < 2:
            return np.zeros(3)
        diff = np.diff(valid_scene[POS].to_numpy(float), axis=0)
        diff = diff[np.diff(valid_scene["frame"].to_numpy(float)) == 1]
        if len(diff) == 0:
            return np.zeros(3)
        return np.nan_to_num(np.sqrt(np.nanmean(diff ** 2, axis=0)))

    def audit_video(self, df: pd.DataFrame, scenes: List[List[int]]) -> List[Dict]:
        df = self._prepare(df)
        frames = df["frame"].to_numpy(float)
        rows = []
        for scene_idx, (start, end) in enumerate(scenes, 1):
            scene = df[(frames >= start) & (frames <= end)]
            if scene.empty:
                continue
            valid = scene["confidence"] >= self.th.conf_thresh
            valid_pct = float(valid.mean() * 100.0)
            vs = scene.loc[valid].sort_values("frame")
            jit = self._jitter(vs)
            if vs.empty:
                head, gaze = np.zeros(3), np.zeros(2)
            else:
                head = (vs[ROT].std(ddof=0).fillna(0.0) * 180 / np.pi).to_numpy()
                gaze = (vs[GAZE].std(ddof=0).fillna(0.0) * 180 / np.pi).to_numpy()
            rows.append({
                "scene_index": scene_idx,
                "scene_start_frame": int(start),
                "scene_end_frame": int(end),
                "scene_frame_count": int(len(scene)),
                "valid_frame_fraction_pct": round(valid_pct, 2),
                "dropout_pct": round(100.0 - valid_pct, 2),
                "mean_confidence": round(float(scene["confidence"].mean()), 4),
                "j_tx": float(jit[0]), "j_ty": float(jit[1]), "j_tz": float(jit[2]),
                "head_pitch_std_deg": float(head[0]),
                "head_yaw_std_deg": float(head[1]),
                "head_roll_std_deg": float(head[2]),
                "gaze_x_std_deg": float(gaze[0]),
                "gaze_y_std_deg": float(gaze[1]),
            })
        return rows

    def apply_gate(self, quality_df: pd.DataFrame) -> pd.DataFrame:
        out = quality_df.copy()
        out["max_jitter"] = out[["j_tx", "j_ty", "j_tz"]].max(axis=1)
        out["quality_gate_pass"] = (
            (out["dropout_pct"] <= self.th.max_dropout_pct)
            & (out["max_jitter"] <= self.th.jitter_thresh)
        )
        return out

    def audit(
        self, catalog_df: pd.DataFrame, tables, scenes_by_video: Dict[str, List[List[int]]],
    ) -> Tuple[pd.DataFrame, StageReport]:
        """Audit every video. Videos that can't be audited are reported, never skipped silently."""
        report = StageReport("quality audit", total=len(catalog_df)).start()
        rows = []
        for r in progress(list(catalog_df.itertuples()), desc="Quality audit"):
            scenes = scenes_by_video.get(r.video_id)
            if not scenes:
                report.fail(r.video_id, "no scenes available")
                continue
            df = tables.load(r.path)
            if df is None:
                report.fail(r.video_id, "missing or unreadable OpenFace table (re-run stage 1)")
                continue
            if df.empty:
                report.fail(r.video_id, "OpenFace found no face")
                continue
            for scene_row in self.audit_video(df, scenes):
                rows.append({"video": r.video_id, "video_index": int(r.sample_index),
                             "label": r.label, **scene_row})
            report.computed += 1
        quality_df = self.apply_gate(pd.DataFrame(rows)) if rows else pd.DataFrame()
        return quality_df, report.finish()
