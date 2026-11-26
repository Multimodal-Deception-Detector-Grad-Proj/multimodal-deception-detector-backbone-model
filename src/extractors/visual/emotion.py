"""Facial emotion recognition with a Hugging Face ViT classifier.

Default model: https://huggingface.co/jihedjabnoun/vit-face-emotion-recognition
(ViT-base fine-tuned on EmoSet, FER2013, RAF-DB and RAVDESS; 7 classes).
"""

import logging
import re
from typing import List, Optional, Sequence

import numpy as np
import torch
from transformers import ViTForImageClassification, ViTImageProcessor

from .video import read_face_crops

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "jihedjabnoun/vit-face-emotion-recognition"


CARD_LABELS = ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]


class EmotionRecognizer:
    """Loads the ViT emotion model and predicts per-frame class probabilities."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cpu",
        batch_size: int = 32,
    ):
        self.device = str(device)
        self.batch_size = batch_size

        logger.info("Loading emotion model: %s", model_id)
        self.processor = ViTImageProcessor.from_pretrained(model_id)
        self.model = ViTForImageClassification.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

        self.labels: List[str] = self._resolve_labels()

    def _resolve_labels(self) -> List[str]:
        """Prefer the checkpoint's own label names; fall back to the model card's."""
        cfg = self.model.config
        n = cfg.num_labels
        names = [str(cfg.id2label.get(i, f"LABEL_{i}")) for i in range(n)]
        if all(re.fullmatch(r"LABEL_\d+", s) for s in names):
            if n != len(CARD_LABELS):
                raise ValueError(
                    f"Model has {n} classes with no label names; expected {len(CARD_LABELS)}"
                )
            logger.info("Checkpoint has generic labels; using model-card order: %s", CARD_LABELS)
            return list(CARD_LABELS)
        logger.info("Emotion labels (from checkpoint config): %s", names)
        return names

    @torch.no_grad()
    def predict(
        self,
        video_path: str,
        frame_indices: Sequence[int],
        boxes: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """Return emotion probabilities for the requested zero-based frames."""
        n = len(frame_indices)
        if not n:
            return torch.empty((0, len(self.labels)))

        images = read_face_crops(video_path, frame_indices, boxes)

        use_amp = self.device.startswith("cuda")
        outputs = []
        for start in range(0, n, self.batch_size):
            # The HF processor does the resize + normalization the model was trained with.
            pixel_values = self.processor(
                images=images[start:start + self.batch_size], return_tensors="pt"
            )["pixel_values"].to(self.device)

            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = self.model(pixel_values=pixel_values).logits

            outputs.append(torch.softmax(logits.float(), dim=-1).cpu())
        return torch.cat(outputs)