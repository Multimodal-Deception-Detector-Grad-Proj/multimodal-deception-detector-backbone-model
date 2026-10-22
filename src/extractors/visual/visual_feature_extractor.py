"""Extract frame-level OpenFace features and Custom Emotion probabilities."""

import logging
import os
import shutil
import subprocess
import tempfile
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

logger = logging.getLogger(__name__)

# --- CUSTOM MODEL EDIT HERE (1/3): Import your custom model architecture ---
# from my_custom_models import MyEmotionRecognizer
# -------------------------------------------------------------------------

class VisualFeatureExtractor:
    """Combine OpenFace AUs/pose/gaze with custom emotion probabilities."""

    POSE = ["pose_Tx", "pose_Ty", "pose_Tz", "pose_Rx", "pose_Ry", "pose_Rz"]
    GAZE = [
        "gaze_0_x", "gaze_0_y", "gaze_0_z",
        "gaze_1_x", "gaze_1_y", "gaze_1_z",
        "gaze_angle_x", "gaze_angle_y",
    ]
    AU_INTENSITY = [
        "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r",
        "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r",
        "AU25_r", "AU26_r", "AU45_r",
    ]
    AU_PRESENCE = [
        "AU01_c", "AU02_c", "AU04_c", "AU05_c", "AU06_c", "AU07_c", "AU09_c",
        "AU10_c", "AU12_c", "AU14_c", "AU15_c", "AU17_c", "AU20_c", "AU23_c",
        "AU25_c", "AU26_c", "AU28_c", "AU45_c",
    ]
    OPENFACE_COLS = AU_INTENSITY + AU_PRESENCE + POSE + GAZE
    QUALITY_COLS = ["valid", "confidence"]
    LANDMARK_X = [f"x_{i}" for i in range(68)]
    LANDMARK_Y = [f"y_{i}" for i in range(68)]

    def __init__(
        self,
        custom_weights_path="path/to/your/custom_weights.pth",
        openface_bin_path="FeatureExtraction",
        device=None,
        min_confidence=0.75,
        crop_faces=True,
        crop_margin=0.2,
        include_quality_flags=True,
        add_pose_delta=False,
        scale_features=True,
        batch_size=32,
        openface_timeout=300.0,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.min_confidence = min_confidence
        self.crop_faces = crop_faces
        self.crop_margin = crop_margin
        self.include_quality_flags = include_quality_flags
        self.add_pose_delta = add_pose_delta
        self.scale_features = scale_features
        self.batch_size = batch_size
        self.openface_timeout = openface_timeout

        self.openface_bin = shutil.which(openface_bin_path)
        if not self.openface_bin:
            raise FileNotFoundError(f"OpenFace binary not found: {openface_bin_path!r}")

        logger.info("Loading custom emotion model from: %s", custom_weights_path)
        
        # --- CUSTOM MODEL EDIT HERE (2/3): Initialize and load your model ---
        # Replace the labels with your model's exact output classes
        self.emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
        
        # Initialize your custom architecture and load weights
        # self.emotion_model = MyEmotionRecognizer(num_classes=len(self.emotion_labels))
        # self.emotion_model.load_state_dict(torch.load(custom_weights_path))
        
        # Placeholder dummy model for now (remove this and uncomment above)
        self.emotion_model = torch.nn.Linear(224*224*3, len(self.emotion_labels)) 
        
        self.emotion_model = self.emotion_model.to(self.device)
        self.emotion_model.eval()

        # Define the image transforms your custom model expects
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # -------------------------------------------------------------------------

    @property
    def feature_names(self) -> List[str]:
        names = self.OPENFACE_COLS.copy()
        if self.include_quality_flags:
            names += self.QUALITY_COLS
        if self.add_pose_delta:
            names += [f"d_{c}" for c in self.POSE]
        return names + [f"emo_{label}" for label in self.emotion_labels]

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    def run_openface(self, video_path: str, output_dir: str, landmarks=False) -> pd.DataFrame:
        """Run OpenFace and return its output CSV."""
        cmd = [
            self.openface_bin, "-f", video_path, "-out_dir", output_dir,
            "-pose", "-gaze", "-aus", "-q",
        ]
        if landmarks:
            cmd.append("-2Dfp")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.openface_timeout
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"OpenFace timed out after {self.openface_timeout}s: {video_path}"
            ) from exc
        if result.returncode:
            raise RuntimeError(f"OpenFace failed: {result.stderr}")

        name = os.path.splitext(os.path.basename(video_path))[0]
        csv_path = os.path.join(output_dir, f"{name}.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"OpenFace CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        return df

    def _validity(self, df: pd.DataFrame) -> np.ndarray:
        """Mark frames with a successful, sufficiently confident face track."""
        required = ["success", "confidence"]
        missing = [c for c in required if c not in df]
        if missing:
            raise KeyError(f"Missing OpenFace columns: {missing}")
        return (
            (df["success"].to_numpy(dtype=float) > 0.5)
            & (df["confidence"].to_numpy(dtype=float) >= self.min_confidence)
        )

    def _clean_openface(self, df: pd.DataFrame, valid: np.ndarray) -> np.ndarray:
        """Mask failed tracks, interpolate continuous features, and scale values."""
        missing = [c for c in self.OPENFACE_COLS if c not in df]
        if missing:
            raise KeyError(f"Missing OpenFace columns: {missing}")

        feats = df[self.OPENFACE_COLS].astype(float).copy()
        feats.loc[~valid] = np.nan

        if valid.any():
            continuous = self.AU_INTENSITY + self.POSE + self.GAZE
            feats[continuous] = feats[continuous].interpolate(limit_direction="both")
            feats[self.AU_PRESENCE] = feats[self.AU_PRESENCE].ffill().bfill()
        feats = feats.fillna(0.0)

        if self.scale_features:
            feats[self.AU_INTENSITY] /= 5.0
            feats[["pose_Tx", "pose_Ty", "pose_Tz"]] /= 1000.0
        return feats.to_numpy(dtype=np.float32)

    def _face_boxes(self, df: pd.DataFrame, valid: np.ndarray) -> Optional[np.ndarray]:
        """Build square landmark-based crop boxes, reusing nearest valid boxes."""
        cols = self.LANDMARK_X + self.LANDMARK_Y
        if not all(c in df for c in cols) or not valid.any():
            return None

        xs, ys = df[self.LANDMARK_X].to_numpy(float), df[self.LANDMARK_Y].to_numpy(float)
        x0, x1, y0, y1 = xs.min(axis=1), xs.max(axis=1), ys.min(axis=1), ys.max(axis=1)
        side = np.maximum(x1 - x0, y1 - y0) * (1 + 2 * self.crop_margin)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        boxes = np.column_stack((cx-side/2, cy-side/2, cx+side/2, cy+side/2))
        boxes[~valid] = np.nan
        boxes = pd.DataFrame(boxes).ffill().bfill().to_numpy()
        return None if np.isnan(boxes).any() else boxes

    @staticmethod
    def _crop(rgb: np.ndarray, box: Optional[np.ndarray]) -> Image.Image:
        """Crop an RGB frame to a box, falling back to the full frame if too small."""
        if box is not None:
            h, w = rgb.shape[:2]
            x0, y0 = max(round(box[0]), 0), max(round(box[1]), 0)
            x1, y1 = min(round(box[2]), w), min(round(box[3]), h)
            if x1 - x0 >= 8 and y1 - y0 >= 8:
                return Image.fromarray(rgb[y0:y1, x0:x1])
        return Image.fromarray(rgb)

    @torch.no_grad()
    def extract_emotion_probs(
        self, video_path: str, frame_indices: Sequence[int],
        boxes: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """Return custom emotion probabilities for the requested zero-based frames."""
        n = len(frame_indices)
        if not n:
            return torch.empty((0, len(self.emotion_labels)))

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
                        images[pos] = self._crop(rgb, None if boxes is None else boxes[pos])
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

        use_amp = self.device.startswith("cuda")
        outputs = []
        for start in range(0, n, self.batch_size):
            # Apply standard torchvision transforms to the PIL crops
            batch_imgs = [self.transform(img) for img in images[start:start + self.batch_size]]
            batch_tensor = torch.stack(batch_imgs).to(self.device)
            
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                # --- CUSTOM MODEL EDIT HERE (3/3): Model inference step ---
                # Adjust this if your model returns a tuple or specific dictionary
                # e.g., logits = self.emotion_model(batch_tensor)[0]
                logits = self.emotion_model(batch_tensor.view(batch_tensor.size(0), -1)) # Dummy reshape for the placeholder Linear layer
                # ----------------------------------------------------------
                
            outputs.append(torch.softmax(logits.float(), dim=-1).cpu())
        return torch.cat(outputs)

    @staticmethod
    def _video_frame_count(video_path: str) -> int:
        cap = cv2.VideoCapture(video_path)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return max(count, 0)

    def extract_features(self, video_path: str, target_t: int = 90) -> torch.FloatTensor:
        """Return visual features with shape (1, target_t, feature_dim)."""
        with tempfile.TemporaryDirectory() as tmp:
            df = self.run_openface(video_path, tmp, landmarks=self.crop_faces)

        if len(df):
            valid = self._validity(df)
            if not valid.any():
                logger.warning("%s: no valid face tracks; OpenFace features are zero", video_path)

            cleaned = self._clean_openface(df, valid)
            rows = np.linspace(0, len(df) - 1, target_t).round().astype(int)
            of_feats = cleaned[rows]
            valid_s = valid[rows].astype(np.float32)
            conf_s = np.clip(df["confidence"].to_numpy(np.float32)[rows], 0, 1)
            frame_idx = (
                df["frame"].to_numpy(dtype=int)[rows] - 1 if "frame" in df
                else rows.copy()
            )
            frame_idx = np.clip(frame_idx, 0, None)
            boxes = self._face_boxes(df, valid)
            boxes = boxes[rows] if self.crop_faces and boxes is not None else None
        else:
            logger.warning("%s: OpenFace returned no rows", video_path)
            of_feats = np.zeros((target_t, len(self.OPENFACE_COLS)), dtype=np.float32)
            valid_s = conf_s = np.zeros(target_t, dtype=np.float32)
            total = self._video_frame_count(video_path)
            frame_idx = (
                np.linspace(0, total - 1, target_t).round().astype(int)
                if total else np.zeros(target_t, dtype=int)
            )
            boxes = None

        parts = [of_feats]
        if self.include_quality_flags:
            parts.append(np.column_stack((valid_s, conf_s)))
        if self.add_pose_delta:
            start = self.OPENFACE_COLS.index(self.POSE[0])
            pose = of_feats[:, start:start + len(self.POSE)]
            parts.append(np.diff(pose, axis=0, prepend=pose[:1]).astype(np.float32))

        tabular = torch.from_numpy(np.concatenate(parts, axis=1).astype(np.float32))
        emotion = self.extract_emotion_probs(video_path, frame_idx, boxes)
        features = torch.cat((tabular, emotion), dim=-1).unsqueeze(0)

        if not torch.isfinite(features).all():
            logger.warning("%s: replacing non-finite features with zero", video_path)
            features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        expected = (1, target_t, self.feature_dim)
        if tuple(features.shape) != expected:
            raise ValueError(f"Expected shape {expected}, got {tuple(features.shape)}")
        return features