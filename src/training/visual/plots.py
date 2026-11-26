from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


class TrainingPlots:
    @staticmethod
    def curves(history: pd.DataFrame, show: bool = True):
        fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
        for a, (cols, title) in zip(ax, [(("train_loss", "val_loss"), "Loss"),
                                         (("train_acc", "val_acc"), "Accuracy"),
                                         (("train_f1", "val_f1"), "Macro-F1")]):
            for c in cols:
                a.plot(history["epoch"], history[c], label=c)
            a.set_title(title); a.set_xlabel("epoch"); a.legend(); a.grid(alpha=0.3)
        fig.tight_layout()
        if show: plt.show()
        return fig


class EvalPlots:
    @staticmethod
    def confusion(cm: pd.DataFrame, title: str = "", show: bool = True):
        fig, ax = plt.subplots(figsize=(3.8, 3.4))
        ax.imshow(cm.values, cmap="Blues")
        ax.set_xticks(range(cm.shape[1]), cm.columns, rotation=20); ax.set_yticks(range(cm.shape[0]), cm.index)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm.values[i, j], ha="center", va="center",
                        color="white" if cm.values[i, j] > cm.values.max() / 2 else "black")
        ax.set_title(title); fig.tight_layout()
        if show: plt.show()
        return fig

    @staticmethod
    def roc(pred_df: pd.DataFrame, show: bool = True):
        fig, ax = plt.subplots(figsize=(4.5, 4))
        for split, d in pred_df.groupby("split"):
            y = (d["label"] == "Lie").astype(int)
            if y.nunique() == 2:
                fpr, tpr, _ = roc_curve(y, d["p_lie"]); ax.plot(fpr, tpr, label=split)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC (Lie = positive)"); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        if show: plt.show()
        return fig

    @staticmethod
    def seed_sweep(sweep_df: pd.DataFrame, metric: str = "accuracy", show: bool = True):
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sweep_df[metric].unstack("split")[["train", "val", "test"]].boxplot(ax=ax)
        ax.set_title(f"{metric} across seeds"); fig.tight_layout()
        if show: plt.show()
        return fig
