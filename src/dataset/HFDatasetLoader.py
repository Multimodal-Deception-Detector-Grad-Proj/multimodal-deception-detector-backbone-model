import glob
import logging
import os
from typing import List, Optional, Sequence, Tuple
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
