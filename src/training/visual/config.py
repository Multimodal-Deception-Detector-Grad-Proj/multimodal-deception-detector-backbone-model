"""Single source of truth for the visual-modality training run."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch


@dataclass
class TrainConfig:
    # --- data ---
    use_quality_gate: bool = False            # drop videos with any scene failing the Stage-3 gate
    include_prefixes: Optional[Tuple[str, ...]] = None   # e.g. ("au_", "gaze_"); None = all features
    exclude_prefixes: Optional[Tuple[str, ...]] = None   # e.g. ("emo_",) for an ablation
    val_frac: float = 0.15
    test_frac: float = 0.15
    batch_size: int = 32

    # --- model ---
    rnn_type: str = "gru"                    # "lstm" | "gru"
    hidden_dim: int = 64
    num_layers: int = 1
    bidirectional: bool = False
    dropout: float = 0.3
    pooling: str = "attention"                # "attention" | "mean" | "last"

    # --- optimisation ---
    lr: float = 1e-3
    weight_decay: float = 1e-2
    epochs: int = 100
    patience: int = 14                        # early stopping on val macro-F1
    grad_clip: float = 1.0
    class_weighted_loss: bool = True
    noise_std: float = 0.05                   # train-time gaussian noise (on standardised features)
    feature_dropout: float = 0.1              # train-time random feature-channel dropout

    # --- misc ---
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "../outputs/visual_training"
    num_workers: int = 0
