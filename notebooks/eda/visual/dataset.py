"""Dataset access: Hugging Face download, label resolution, video catalog."""

import glob
import logging
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import pandas as pd

from .status import progress

logger = logging.getLogger(__name__)


class HFDatasetLoader:
    """Lists, downloads and locates the dataset's videos."""

    def __init__(
        self,
        repo_id: str,
        local_dir: str,
        folders: Optional[Sequence[str]] = ("deception", "truth"),
        allow_patterns: Optional[Sequence[str]] = None,
        extensions: Tuple[str, ...] = (".mp4",),
    ):
        self.repo_id, self.local_dir = repo_id, local_dir
        self.folders = tuple(folders) if folders else None
        self.allow_patterns = list(allow_patterns) if allow_patterns else None
        self.extensions = tuple(e.lower() for e in extensions)

    def list_remote_files(self) -> List[str]:
        from huggingface_hub import HfApi

        files = HfApi().list_repo_files(repo_id=self.repo_id, repo_type="dataset")
        if self.folders:
            files = [f for f in files if f.split("/", 1)[0] in self.folders]
        return files

    def local_video_paths(self) -> List[str]:
        paths = glob.glob(os.path.join(self.local_dir, "**", "*"), recursive=True)
        return sorted(p for p in paths if p.lower().endswith(self.extensions))

    def download(self, force: bool = False) -> str:
        """Download the dataset. Skipped when videos are already on disk unless ``force``
        (use ``force=True`` if a previous download was interrupted)."""
        os.makedirs(self.local_dir, exist_ok=True)
        if not force and self.local_video_paths():
            logger.info("Videos already present in %s; skipping download", self.local_dir)
            return self.local_dir
        from huggingface_hub import snapshot_download

        return snapshot_download(
            repo_id=self.repo_id, repo_type="dataset", local_dir=self.local_dir,
            allow_patterns=self.allow_patterns,
        )


class LabelResolver:
    """Resolve Lie/Truth from the folder name first, then from whole-word filename tokens
    (so 'believe' no longer matches 'lie')."""

    def __init__(
        self,
        folder_labels: Optional[Dict[str, str]] = None,
        token_labels: Optional[Dict[str, str]] = None,
    ):
        self.folder_labels = folder_labels or {
            "deception": "Lie", "lie": "Lie", "truth": "Truth", "truthful": "Truth",
        }
        self.token_labels = token_labels or {
            "lie": "Lie", "deception": "Lie", "deceptive": "Lie",
            "truth": "Truth", "true": "Truth", "truthful": "Truth",
        }

    def resolve(self, path: str, root: str) -> Tuple[str, str]:
        """Return (label, source) where source is 'folder', 'filename' or 'unknown'."""
        rel = os.path.relpath(path, root).replace(os.sep, "/")
        parts = rel.split("/")
        for folder in parts[:-1]:
            if folder.lower() in self.folder_labels:
                return self.folder_labels[folder.lower()], "folder"
        stem = os.path.splitext(parts[-1])[0].lower()
        for token in re.split(r"[^a-z]+", stem):
            if token in self.token_labels:
                return self.token_labels[token], "filename"
        return "Unknown", "unknown"


class VideoCatalog:
    """Builds the metadata table (``meta_df``) used by every later stage.

    ``video_id`` is the path relative to the dataset root and is the unique key
    everywhere (basenames can collide across folders)."""

    def __init__(self, root: str, resolver: Optional[LabelResolver] = None):
        self.root = root
        self.resolver = resolver or LabelResolver()
        self.failed: List[str] = []

    def build(self, paths: Iterable[str]) -> pd.DataFrame:
        rows, self.failed = [], []
        for idx, path in enumerate(progress(sorted(paths), desc="Reading video metadata")):
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                self.failed.append(path)
                continue
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            label, source = self.resolver.resolve(path, self.root)
            rows.append({
                "sample_index": idx,
                "video_id": os.path.relpath(path, self.root).replace(os.sep, "/"),
                "filename": os.path.basename(path),
                "path": path,
                "label": label,
                "label_source": source,
                "frame_count": frame_count,
                "fps": round(fps, 2),
                "width": width,
                "height": height,
                "duration_sec": round(frame_count / fps, 2) if fps > 0 else 0.0,
            })
        meta_df = pd.DataFrame(rows)
        if self.failed:
            logger.warning("Could not open %d videos (see catalog.failed)", len(self.failed))
        if not meta_df.empty and (meta_df["label"] == "Unknown").any():
            logger.warning("%d videos have no resolvable label", (meta_df["label"] == "Unknown").sum())
        return meta_df
