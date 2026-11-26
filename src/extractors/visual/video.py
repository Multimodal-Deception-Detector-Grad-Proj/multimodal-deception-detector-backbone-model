"""Video frame reading helpers."""

import logging
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
from PIL import Image

from .faces import crop_face

logger = logging.getLogger(__name__)


def video_frame_count(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(count, 0)


def read_face_crops(
    video_path: str,
    frame_indices: Sequence[int],
    boxes: Optional[np.ndarray] = None,
) -> List[Image.Image]:
    """Return one PIL crop per requested zero-based frame (same order/length)."""
    n = len(frame_indices)

    # Keep duplicate frame requests mapped to all corresponding output positions.
    wanted: Dict[int, List[int]] = {}
    for pos, frame_idx in enumerate(frame_indices):
        wanted.setdefault(int(frame_idx), []).append(pos)

    images = [None] * n
    cap = cv2.VideoCapture(video_path)
    idx, last = 0, max(wanted)
    while idx <= last and cap.grab():
        if idx in wanted:
            ok, frame = cap.retrieve()
            if ok:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                for pos in wanted[idx]:
                    images[pos] = crop_face(rgb, None if boxes is None else boxes[pos])
        idx += 1
    cap.release()

    # Fill unreadable frames from the nearest readable neighbour.
    if any(im is None for im in images):
        logger.warning("%s: some frames could not be read; filling from neighbours", video_path)
        previous = None
        for i, im in enumerate(images):
            if im is None:
                images[i] = previous
            else:
                previous = im
        following = None
        for i in reversed(range(n)):
            if images[i] is None:
                images[i] = following
            else:
                following = images[i]
        blank = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        images = [im or blank for im in images]
    return images