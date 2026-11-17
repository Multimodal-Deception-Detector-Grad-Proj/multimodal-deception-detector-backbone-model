import numpy as np
import pandas as pd
import torch

from src.extractors.visual.visual_feature_extractor import VisualFeatureExtractor
from src.common.constants import TARGET_T, D_VIS


def test_visual_extractor_resamples_mock_openface_dataframe(monkeypatch):
    extractor = VisualFeatureExtractor.__new__(VisualFeatureExtractor)
    extractor.include_quality_flags = True
    extractor.add_pose_delta = False
    extractor.scale_features = False
    extractor.min_confidence = 0.75

    columns = extractor.OPENFACE_COLS
    rows = []
    for i in range(8):
        row = {name: float(i + 1) for name in columns}
        row["success"] = 1
        row["confidence"] = 0.95
        rows.append(row)
    sample = pd.DataFrame(rows)

    monkeypatch.setattr(extractor, "run_openface", lambda *_: sample)
    for sample_number in range(3):
        result = extractor.extract_features(f"sample_{sample_number}.mp4")
        assert result.shape == (1, TARGET_T, D_VIS)
        assert result.dtype == torch.float32
        assert torch.isfinite(result).all()
