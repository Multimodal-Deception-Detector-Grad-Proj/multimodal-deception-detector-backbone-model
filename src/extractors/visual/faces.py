"""Landmark-based face boxes and cropping."""

from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

from .constants import LANDMARK_X, LANDMARK_Y


def face_boxes(
    df: pd.DataFrame, valid: np.ndarray, crop_margin: float
) -> Optional[np.ndarray]:
    """Build square landmark-based crop boxes, reusing nearest valid boxes."""
    cols = LANDMARK_X + LANDMARK_Y
    if not all(c in df for c in cols) or not valid.any():
        return None

    xs, ys = df[LANDMARK_X].to_numpy(float), df[LANDMARK_Y].to_numpy(float)
    x0, x1, y0, y1 = xs.min(axis=1), xs.max(axis=1), ys.min(axis=1), ys.max(axis=1)
    side = np.maximum(x1 - x0, y1 - y0) * (1 + 2 * crop_margin)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    boxes = np.column_stack((cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2))
    boxes[~valid] = np.nan
    boxes = pd.DataFrame(boxes).ffill().bfill().to_numpy()
    return None if np.isnan(boxes).any() else boxes


def crop_face(rgb: np.ndarray, box: Optional[np.ndarray]) -> Image.Image:
    """Crop an RGB frame to a box, falling back to the full frame if too small."""
    if box is not None:
        h, w = rgb.shape[:2]
        x0, y0 = max(round(box[0]), 0), max(round(box[1]), 0)
        x1, y1 = min(round(box[2]), w), min(round(box[3]), h)
        if x1 - x0 >= 8 and y1 - y0 >= 8:
            return Image.fromarray(rgb[y0:y1, x0:x1])
    return Image.fromarray(rgb)