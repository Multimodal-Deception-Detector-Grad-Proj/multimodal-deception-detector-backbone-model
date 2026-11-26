"""One full run: split -> standardise -> train -> evaluate. The notebook only calls this."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import TrainConfig
from .dataset import SequenceData, Standardizer, VisualSequenceDataset
from .evaluator import Evaluator
from .model import RNNClassifier
from .splits import SpeakerSplitter
from .trainer import Trainer
from .utils import set_seed


@dataclass
class ExperimentResult:
    seed: int
    model: torch.nn.Module
    history: pd.DataFrame
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    split_summary: pd.DataFrame
    loaders: Dict[str, DataLoader]
    standardizer: Standardizer


class Experiment:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg

    def run(self, data: SequenceData, seed: int | None = None) -> ExperimentResult:
        cfg = dataclasses.replace(self.cfg, seed=self.cfg.seed if seed is None else seed)
        set_seed(cfg.seed)

        splits = SpeakerSplitter(cfg.val_frac, cfg.test_frac, cfg.seed).split(data)
        split_summary = SpeakerSplitter.summary(data, splits)
        parts = {k: data.subset(v) for k, v in splits.items()}

        std = Standardizer().fit(parts["train"])
        parts = {k: std.transform(v) for k, v in parts.items()}

        datasets = {k: VisualSequenceDataset(v, train=(k == "train"), noise_std=cfg.noise_std,
                                             feature_dropout=cfg.feature_dropout) for k, v in parts.items()}
        g = torch.Generator().manual_seed(cfg.seed)
        loaders = {k: DataLoader(ds, batch_size=cfg.batch_size, shuffle=(k == "train"), generator=g,
                                 num_workers=cfg.num_workers) for k, ds in datasets.items()}
        # un-augmented loader over train for honest train-set metrics
        eval_loaders = dict(loaders)
        eval_loaders["train"] = DataLoader(VisualSequenceDataset(parts["train"]), batch_size=cfg.batch_size)

        counts = np.bincount(parts["train"].y, minlength=2)
        weights = torch.tensor(counts.sum() / (2.0 * np.maximum(counts, 1)), dtype=torch.float32)

        model = RNNClassifier.from_config(cfg, data.input_dim)
        trainer = Trainer(model, cfg, weights)
        history = trainer.fit(loaders["train"], loaders["val"])

        metrics, preds = Evaluator(cfg.device).evaluate(trainer.model, eval_loaders, parts)
        return ExperimentResult(cfg.seed, trainer.model, history, metrics, preds, split_summary, eval_loaders, std)

    def run_seeds(self, data: SequenceData, seeds: List[int]) -> pd.DataFrame:
        """Repeat with different seeds (new speaker split each time); returns long-form metrics."""
        frames = []
        for s in seeds:
            print(f"\n=== seed {s} ===")
            r = self.run(data, s)
            frames.append(r.metrics.assign(seed=s).rename_axis("split").reset_index())
        return pd.concat(frames).set_index(["seed", "split"])
