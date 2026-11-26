"""Visual-modality EDA: dataset access, cached OpenFace/scene stages, quality audit, plots."""
import os 
import sys
from .caching import DiskCache, legacy_key, params_key
from .chunking import ChunkPlanner
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
from src.dataset.visual_dataset import HFDatasetLoader, LabelResolver, VideoCatalog
from src.extractors.visual.features import FeatureSet, FeatureStore
from src.extractors.visual.openface_store import OpenFaceTableStore
from .plots import DatasetPlots, EmotionPlots, QualityPlots
from .quality import QualityAuditor, QualityThresholds
from .scenes import SceneDetector
from .status import StageReport

__all__ = [
    "ChunkPlanner", "DatasetPlots", "DiskCache", "EmotionPlots", "FeatureSet",
    "FeatureStore", "HFDatasetLoader", "LabelResolver", "OpenFaceTableStore", "QualityAuditor",
    "QualityPlots", "QualityThresholds", "SceneDetector", "StageReport", "VideoCatalog",
    "legacy_key", "params_key",
]
