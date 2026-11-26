"""Visual-modality EDA: dataset access, cached OpenFace/scene stages, quality audit, plots."""

from .caching import DiskCache, legacy_key, params_key
from .chunking import ChunkPlanner
from .config import EDAConfig
from .dataset import HFDatasetLoader, LabelResolver, VideoCatalog
from .features import FeatureSet, FeatureStore
from .openface_store import OpenFaceTableStore
from .plots import DatasetPlots, EmotionPlots, QualityPlots
from .quality import QualityAuditor, QualityThresholds
from .scenes import SceneDetector
from .status import StageReport

__all__ = [
    "ChunkPlanner", "DatasetPlots", "DiskCache", "EDAConfig", "EmotionPlots", "FeatureSet",
    "FeatureStore", "HFDatasetLoader", "LabelResolver", "OpenFaceTableStore", "QualityAuditor",
    "QualityPlots", "QualityThresholds", "SceneDetector", "StageReport", "VideoCatalog",
    "legacy_key", "params_key",
]
