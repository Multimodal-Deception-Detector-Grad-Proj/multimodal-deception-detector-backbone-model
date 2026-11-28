from pathlib import Path
import pytest
import yaml
from src.common.constants import D_TXT, L_TXT, TARGET_T

ROOT = Path(__file__).resolve().parents[1]
CLIPS = sorted((ROOT / "data/raw").rglob("*.mp4"))[:3]


@pytest.mark.skipif(len(CLIPS) < 3, reason="needs 3 real clips in data/raw")
def test_three_real_clips_end_to_end():
    from src.extractors.text.extract_text import TextFeatureExtractor
    cfg = yaml.safe_load((ROOT / "configs/base_config.yaml").read_text(encoding="utf-8"))
    ext = TextFeatureExtractor(cfg)
    for clip in CLIPS:
        s = ext.process_clip(clip.stem, clip)
        assert s["text_feats"].shape == (L_TXT, D_TXT)
        assert (s["alignment_indices"] < TARGET_T).all()