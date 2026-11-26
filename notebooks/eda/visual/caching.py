"""Atomic on-disk caches keyed by video path (+ optional parameters)."""

import hashlib
import json
import logging
import os
import threading
from typing import Any, Callable, Dict

import pandas as pd

logger = logging.getLogger(__name__)


def _stem(video_path: str) -> str:
    return os.path.splitext(os.path.basename(video_path))[0]


def legacy_key(video_path: str) -> str:
    """Same naming as the original notebook, so existing OpenFace pickles stay valid."""
    digest = hashlib.sha256(os.path.abspath(video_path).encode("utf-8")).hexdigest()
    return f"{_stem(video_path)}_{digest}"


def params_key(video_path: str, params: Dict[str, Any]) -> str:
    """Key that also depends on the parameters, so changing them can't serve stale results."""
    blob = json.dumps(
        {"path": os.path.abspath(video_path), "params": params}, sort_keys=True, default=str
    )
    return f"{_stem(video_path)}_{hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]}"


def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


class DiskCache:
    """One file per video. Writes are atomic (temp file + rename), so an interrupted
    run can never leave a half-written file that later looks like a valid cache hit.
    Unreadable files are deleted on load and reported as a miss."""

    def __init__(
        self,
        root: str,
        suffix: str,
        reader: Callable[[str], Any],
        writer: Callable[[Any, str], None],
        key_fn: Callable[[str], str] = legacy_key,
    ):
        self.root, self.suffix, self.key_fn = root, suffix, key_fn
        self._read, self._write = reader, writer
        os.makedirs(root, exist_ok=True)

    # -- constructors for the formats we use ------------------------------
    @classmethod
    def pickle(cls, root: str, key_fn=legacy_key) -> "DiskCache":
        return cls(root, ".pkl", pd.read_pickle,
                   lambda obj, p: obj.to_pickle(p, compression=None), key_fn)

    @classmethod
    def json(cls, root: str, key_fn=legacy_key) -> "DiskCache":
        return cls(root, ".json", _read_json, _write_json, key_fn)

    @classmethod
    def torch(cls, root: str, key_fn=legacy_key) -> "DiskCache":
        import torch  # lazy: only needed for feature tensors
        return cls(root, ".pt", lambda p: torch.load(p, weights_only=True),
                   lambda obj, p: torch.save(obj, p), key_fn)

    # -- API ---------------------------------------------------------------
    def path_for(self, video_path: str) -> str:
        return os.path.join(self.root, f"{self.key_fn(video_path)}{self.suffix}")

    def has(self, video_path: str) -> bool:
        p = self.path_for(video_path)
        return os.path.isfile(p) and os.path.getsize(p) > 0

    def load(self, video_path: str):
        p = self.path_for(video_path)
        if not os.path.isfile(p):
            return None
        try:
            return self._read(p)
        except Exception as exc:
            logger.warning("Removing unreadable cache file %s (%s)", p, exc)
            try:
                os.remove(p)
            except OSError:
                pass
            return None

    def save(self, video_path: str, obj) -> None:
        final = self.path_for(video_path)
        tmp = f"{final}.tmp{os.getpid()}_{threading.get_ident()}"
        try:
            self._write(obj, tmp)
            os.replace(tmp, final)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
