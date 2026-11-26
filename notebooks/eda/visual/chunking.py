"""Sliding-window chunking analysis."""

from typing import Dict, Tuple

import numpy as np
import pandas as pd


class ChunkPlanner:
    """How many fixed-length windows each video yields."""

    def __init__(self, chunk_length_sec: float = 8.0, overlap_sec: float = 2.0):
        if overlap_sec >= chunk_length_sec:
            raise ValueError("overlap_sec must be smaller than chunk_length_sec")
        self.chunk_length_sec, self.overlap_sec = chunk_length_sec, overlap_sec

    @property
    def stride_sec(self) -> float:
        return self.chunk_length_sec - self.overlap_sec

    def projected_chunks(self, duration_sec: float) -> int:
        if duration_sec <= self.chunk_length_sec:
            return 1
        return int(np.ceil((duration_sec - self.chunk_length_sec) / self.stride_sec)) + 1

    def plan(self, meta_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
        out = meta_df.copy()
        out["projected_chunks"] = out["duration_sec"].apply(self.projected_chunks)
        fps = int(out["fps"].mode()[0]) if len(out) else 0
        summary = {
            "target_fps": fps,
            "sequence_length_frames": int(self.chunk_length_sec * fps),
            "chunk_length_sec": self.chunk_length_sec,
            "overlap_sec": self.overlap_sec,
            "original_videos": len(out),
            "projected_sequences": int(out["projected_chunks"].sum()),
        }
        return out, summary
