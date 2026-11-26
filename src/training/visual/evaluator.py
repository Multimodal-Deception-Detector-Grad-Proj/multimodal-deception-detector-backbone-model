from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader

from .dataset import ID_TO_LABEL, SequenceData


class Evaluator:
    def __init__(self, device: str, use_mask: bool = True):
        self.device = torch.device(device)
        self.use_mask = use_mask          # keep equal to cfg.use_mask used in training

    @torch.no_grad()
    def predict(self, model: torch.nn.Module, loader: DataLoader) -> np.ndarray:
        model.eval()
        probs = []
        for x, _, _, *rest in loader:
            x = x.to(self.device)
            if self.use_mask and rest:
                logits = model(x, rest[0].to(self.device))
            else:
                logits = model(x)
            probs.append(torch.softmax(logits.float(), -1).cpu().numpy())
        return np.concatenate(probs)

    @staticmethod
    def metrics(y: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
        pred = probs.argmax(1)
        out = dict(accuracy=accuracy_score(y, pred), balanced_acc=balanced_accuracy_score(y, pred),
                   f1_macro=f1_score(y, pred, average="macro"),
                   precision_lie=precision_score(y, pred, zero_division=0),
                   recall_lie=recall_score(y, pred, zero_division=0))
        out["auc"] = roc_auc_score(y, probs[:, 1]) if len(np.unique(y)) == 2 else float("nan")
        return out

    def evaluate(self, model, loaders: Dict[str, DataLoader], datas: Dict[str, SequenceData]
                 ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Returns (metrics table indexed by split, per-video predictions for every split)."""
        rows, preds = {}, []
        for name, loader in loaders.items():
            d = datas[name]
            probs = self.predict(model, loader)
            rows[name] = self.metrics(d.y, probs)
            preds.append(pd.DataFrame({
                "split": name, "video_id": d.video_ids, "speaker": d.speakers,
                "label": [ID_TO_LABEL[i] for i in d.y], "pred": [ID_TO_LABEL[i] for i in probs.argmax(1)],
                "p_lie": probs[:, 1], "correct": probs.argmax(1) == d.y,
                "valid_frac": d.mask.mean(axis=1)}))
        return pd.DataFrame(rows).T.round(4), pd.concat(preds, ignore_index=True)

    @staticmethod
    def confusion(pred_df: pd.DataFrame, split: str) -> pd.DataFrame:
        d = pred_df[pred_df["split"] == split]
        labels = [ID_TO_LABEL[0], ID_TO_LABEL[1]]
        return pd.DataFrame(confusion_matrix(d["label"], d["pred"], labels=labels),
                            index=[f"true {l}" for l in labels], columns=[f"pred {l}" for l in labels])

    @staticmethod
    def per_speaker(pred_df: pd.DataFrame, split: str = "test") -> pd.DataFrame:
        d = pred_df[pred_df["split"] == split]
        return (d.groupby("speaker")["correct"].agg(videos="size", accuracy="mean").round(3)
                .sort_values("accuracy"))

    @staticmethod
    def by_validity(pred_df: pd.DataFrame, split: str = "test",
                    bins=(0.0, 0.25, 0.5, 0.75, 1.0)) -> pd.DataFrame:
        """Accuracy by fraction of valid face frames per clip (where visual features are reliable)."""
        d = pred_df[pred_df["split"] == split].copy()
        d["valid_bin"] = pd.cut(d["valid_frac"], bins=list(bins), include_lowest=True)
        return (d.groupby("valid_bin", observed=True)["correct"].agg(videos="size", accuracy="mean")
                .round(3))