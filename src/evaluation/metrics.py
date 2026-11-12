from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score


@dataclass
class BootstrapCI:
    """Summary statistics from subject-level bootstrap resampling."""
    mean: float
    lower: float
    upper: float
    std: float
    valid_replicates: int


@dataclass
class EvaluationReport:
    """Complete evaluation summary for validation or test splits."""
    auc_roc: float
    macro_f1: float
    fpr: float
    tpr: float
    ece: float
    threshold: float
    num_windows: int
    num_clips: int
    num_subjects: int
    bootstrap_cis: Dict[str, BootstrapCI] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        """Flattens scalar metrics and CIs into a clean logging dictionary."""
        flat = {
            "auc_roc": self.auc_roc,
            "macro_f1": self.macro_f1,
            "fpr": self.fpr,
            "tpr": self.tpr,
            "ece": self.ece,
            "threshold": self.threshold,
            "num_clips": float(self.num_clips),
            "num_subjects": float(self.num_subjects),
        }
        for metric_name, ci in self.bootstrap_cis.items():
            flat[f"{metric_name}_ci_lower"] = ci.lower
            flat[f"{metric_name}_ci_upper"] = ci.upper
        return flat


def _to_numpy_1d(x: Union[torch.Tensor, np.ndarray, Sequence]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().reshape(-1).numpy()
    return np.asarray(x).reshape(-1)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid
    return np.where(
        logits >= 0,
        1.0 / (1.0 + np.exp(-np.clip(logits, -88.0, 88.0))),
        np.exp(np.clip(logits, -88.0, 88.0)) / (1.0 + np.exp(np.clip(logits, -88.0, 88.0))),
    )


def aggregate_by_clip(
    logits: np.ndarray,
    labels: np.ndarray,
    subject_ids: Sequence[str],
    clip_ids: Optional[Sequence[str]] = None,
    mode: str = "prob_mean",
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Aggregates window-level predictions into clip-level predictions.

    Args:
        logits: Raw model logits of shape (N,).
        labels: Binary ground-truth labels of shape (N,) in {0, 1}.
        subject_ids: Subject/speaker identifier per window (length N).
        clip_ids: Optional clip identifier per window (length N). If None, each
            window is treated as an independent clip.
        mode: Aggregation strategy ("prob_mean" or "logit_mean").
    Returns:
        clip_probs: (N_clips,) aggregated probabilities.
        clip_labels: (N_clips,) binary labels.
        clip_subjects: List of subject IDs of length N_clips.
        unique_clips: List of clip IDs of length N_clips.
    """
    n = len(logits)
    assert len(labels) == n and len(subject_ids) == n, "Input lengths must match."

    if clip_ids is None:
        probs = _sigmoid(logits)
        return probs, labels.astype(np.int64), list(subject_ids), [f"clip_{i}" for i in range(n)]

    assert len(clip_ids) == n, "clip_ids length must match logits."
    assert mode in ("prob_mean", "logit_mean"), f"Unknown aggregation mode: {mode}"

    grouped_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, cid in enumerate(clip_ids):
        grouped_indices[str(cid)].append(idx)

    unique_clips: List[str] = []
    clip_probs: List[float] = []
    clip_labels: List[int] = []
    clip_subjects: List[str] = []

    for cid, idxs in grouped_indices.items():
        grp_labels = labels[idxs]
        assert np.all(grp_labels == grp_labels[0]), (
            f"Inconsistent labels across windows for clip_id={cid}!"
        )
        grp_subjects = {subject_ids[i] for i in idxs}
        assert len(grp_subjects) == 1, (
            f"Multiple subjects assigned to single clip_id={cid}: {grp_subjects}"
        )

        if mode == "prob_mean":
            agg_prob = float(np.mean(_sigmoid(logits[idxs])))
        else:
            agg_prob = float(_sigmoid(np.mean(logits[idxs])))

        unique_clips.append(cid)
        clip_probs.append(agg_prob)
        clip_labels.append(int(grp_labels[0]))
        clip_subjects.append(subject_ids[idxs[0]])

    return (
        np.asarray(clip_probs, dtype=np.float64),
        np.asarray(clip_labels, dtype=np.int64),
        clip_subjects,
        unique_clips,
    )


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE) using confidence-accuracy binning
    (Guo et al., 2017) across `n_bins` equal-width bins in [0, 1].
    """
    if len(y_true) == 0:
        return float("nan")

    y_pred = (y_prob >= 0.5).astype(np.int64)
    confidences = np.where(y_prob >= 0.5, y_prob, 1.0 - y_prob)
    accuracies = (y_pred == y_true).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = float(len(y_true))

    for b in range(n_bins):
        low, high = bin_edges[b], bin_edges[b + 1]
        if b == 0:
            in_bin = (confidences >= low) & (confidences <= high)
        else:
            in_bin = (confidences > low) & (confidences <= high)

        bin_count = int(in_bin.sum())
        if bin_count > 0:
            bin_acc = float(accuracies[in_bin].mean())
            bin_conf = float(confidences[in_bin].mean())
            ece += (bin_count / n_total) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_point_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    n_ece_bins: int = 10,
) -> Dict[str, float]:
    """
    Computes AUC-ROC, Macro-F1, FPR, TPR, and ECE at a fixed decision threshold.
    Handles single-class degenerate subsets safely without raising exceptions.
    """
    y_true = y_true.astype(np.int64)
    y_pred = (y_prob >= threshold).astype(np.int64)

    # AUC-ROC requires both classes present
    if len(np.unique(y_true)) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y_true, y_prob))

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # Confusion matrix components
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else float("nan")
    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    ece = compute_ece(y_true, y_prob, n_bins=n_ece_bins)

    return {
        "auc_roc": auc,
        "macro_f1": macro_f1,
        "fpr": fpr,
        "tpr": tpr,
        "ece": ece,
    }


def find_optimal_threshold(
    val_true: np.ndarray,
    val_prob: np.ndarray,
    criterion: str = "macro_f1",
    max_fpr: float = 0.10,
) -> float:
    """
    Selects decision threshold strictly on validation predictions.

    Supported criteria:
      - "macro_f1": Maximizes validation Macro-F1.
      - "youden": Maximizes Youden's J statistic (TPR - FPR).
      - "target_fpr": Maximizes TPR subject to FPR <= max_fpr.
    """
    assert criterion in ("macro_f1", "youden", "target_fpr"), (
        f"Unsupported threshold criterion: {criterion}"
    )
    if len(val_true) == 0:
        return 0.5

    # Candidate thresholds derived from observed validation probabilities + grid
    candidates = np.unique(
        np.concatenate([np.linspace(0.05, 0.95, 181), np.clip(val_prob, 1e-4, 1.0 - 1e-4)])
    )

    best_thresh = 0.5
    best_score = -float("inf")

    for t in candidates:
        metrics = compute_point_metrics(val_true, val_prob, threshold=float(t))
        if criterion == "macro_f1":
            score = metrics["macro_f1"]
            # Tie-break toward thresholds closer to 0.5
            if (score > best_score + 1e-9) or (
                abs(score - best_score) <= 1e-9 and abs(t - 0.5) < abs(best_thresh - 0.5)
            ):
                best_score = score
                best_thresh = float(t)

        elif criterion == "youden":
            if np.isnan(metrics["tpr"]) or np.isnan(metrics["fpr"]):
                continue
            score = metrics["tpr"] - metrics["fpr"]
            if score > best_score:
                best_score = score
                best_thresh = float(t)

        elif criterion == "target_fpr":
            if np.isnan(metrics["fpr"]) or np.isnan(metrics["tpr"]):
                continue
            if metrics["fpr"] <= max_fpr and metrics["tpr"] > best_score:
                best_score = metrics["tpr"]
                best_thresh = float(t)

    return best_thresh


def subject_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    subject_ids: Sequence[str],
    threshold: float,
    n_bootstraps: int = 1000,
    ci_level: float = 0.95,
    n_ece_bins: int = 10,
    seed: int = 42,
) -> Dict[str, BootstrapCI]:
    """
    Computes non-parametric subject-level bootstrap confidence intervals.
    Resamples unique subjects WITH replacement and evaluates all clips belonging
    to the drawn subjects at the fixed validation threshold.
    """
    subject_to_indices: Dict[str, np.ndarray] = {}
    temp_map: Dict[str, List[int]] = defaultdict(list)
    for idx, sid in enumerate(subject_ids):
        temp_map[str(sid)].append(idx)
    for sid, idxs in temp_map.items():
        subject_to_indices[sid] = np.asarray(idxs, dtype=np.int64)

    unique_subjects = list(subject_to_indices.keys())
    n_subjects = len(unique_subjects)
    if n_subjects == 0 or n_bootstraps <= 0:
        return {}

    rng = np.random.default_rng(seed)
    metric_names = ("auc_roc", "macro_f1", "fpr", "tpr", "ece")
    samples: Dict[str, List[float]] = {m: [] for m in metric_names}

    for _ in range(n_bootstraps):
        drawn_subjects = rng.choice(unique_subjects, size=n_subjects, replace=True)
        boot_indices = np.concatenate([subject_to_indices[s] for s in drawn_subjects])

        boot_true = y_true[boot_indices]
        boot_prob = y_prob[boot_indices]

        # Skip degenerate replicates containing only one class
        if len(np.unique(boot_true)) < 2:
            continue

        m = compute_point_metrics(
            boot_true,
            boot_prob,
            threshold=threshold,
            n_ece_bins=n_ece_bins,
        )
        for name in metric_names:
            if not np.isnan(m[name]):
                samples[name].append(m[name])

    alpha = (1.0 - ci_level) / 2.0
    lower_q = 100.0 * alpha
    upper_q = 100.0 * (1.0 - alpha)

    results: Dict[str, BootstrapCI] = {}
    for name in metric_names:
        vals = np.asarray(samples[name], dtype=np.float64)
        if len(vals) == 0:
            results[name] = BootstrapCI(
                mean=float("nan"),
                lower=float("nan"),
                upper=float("nan"),
                std=float("nan"),
                valid_replicates=0,
            )
        else:
            results[name] = BootstrapCI(
                mean=float(np.mean(vals)),
                lower=float(np.percentile(vals, lower_q)),
                upper=float(np.percentile(vals, upper_q)),
                std=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                valid_replicates=int(len(vals)),
            )

    return results


class DeceptionEvaluator:
    """
    Stateful evaluation accumulator enforcing Block 9 protocols:
      1. Collects batch logits, labels, subject_ids, and optional clip_ids.
      2. Aggregates window predictions to clip level.
      3. Tunes threshold ONLY on validation split (`tune_threshold=True`).
      4. Evaluates test split using the locked validation threshold + subject bootstrap CIs.
    """
    def __init__(
        self,
        aggregation_mode: str = "prob_mean",
        n_ece_bins: int = 10,
        n_bootstraps: int = 1000,
        ci_level: float = 0.95,
        seed: int = 42,
    ):
        self.aggregation_mode = aggregation_mode
        self.n_ece_bins = n_ece_bins
        self.n_bootstraps = n_bootstraps
        self.ci_level = ci_level
        self.seed = seed
        self.reset()

    def reset(self) -> None:
        self._logits: List[np.ndarray] = []
        self._labels: List[np.ndarray] = []
        self._subject_ids: List[str] = []
        self._clip_ids: List[str] = []
        self._has_clip_ids: Optional[bool] = None

    def update(
        self,
        logits: Union[torch.Tensor, np.ndarray],
        labels: Union[torch.Tensor, np.ndarray],
        subject_ids: Sequence[str],
        clip_ids: Optional[Sequence[str]] = None,
    ) -> None:
        logits_np = _to_numpy_1d(logits)
        labels_np = _to_numpy_1d(labels).astype(np.int64)
        n = len(logits_np)

        assert len(labels_np) == n and len(subject_ids) == n, (
            f"Batch size mismatch: logits={n}, labels={len(labels_np)}, subjects={len(subject_ids)}"
        )

        if self._has_clip_ids is None:
            self._has_clip_ids = clip_ids is not None
        else:
            assert (clip_ids is not None) == self._has_clip_ids, (
                "Inconsistent clip_ids presence across batches in the same evaluation pass."
            )

        self._logits.append(logits_np)
        self._labels.append(labels_np)
        self._subject_ids.extend([str(s) for s in subject_ids])
        if clip_ids is not None:
            self._clip_ids.extend([str(c) for c in clip_ids])

    def compute(
        self,
        threshold: Optional[float] = None,
        tune_threshold: bool = False,
        threshold_criterion: str = "macro_f1",
        max_fpr: float = 0.10,
        run_bootstrap: bool = False,
    ) -> EvaluationReport:
        """
        Computes final EvaluationReport from accumulated batches.

        Protocol usage:
          - Validation pass: `val_report = evaluator.compute(tune_threshold=True)`
          - Test pass:       `test_report = evaluator.compute(threshold=val_report.threshold, run_bootstrap=True)`
        """
        assert len(self._logits) > 0, "No predictions accumulated. Call .update() first."
        assert not (tune_threshold and threshold is not None), (
            "Pass either `tune_threshold=True` (val) OR a fixed `threshold` (test), not both."
        )

        all_logits = np.concatenate(self._logits, axis=0)
        all_labels = np.concatenate(self._labels, axis=0)
        clip_arg = self._clip_ids if self._has_clip_ids else None

        clip_probs, clip_labels, clip_subjects, _ = aggregate_by_clip(
            logits=all_logits,
            labels=all_labels,
            subject_ids=self._subject_ids,
            clip_ids=clip_arg,
            mode=self.aggregation_mode,
        )

        if tune_threshold:
            active_threshold = find_optimal_threshold(
                val_true=clip_labels,
                val_prob=clip_probs,
                criterion=threshold_criterion,
                max_fpr=max_fpr,
            )
        else:
            active_threshold = 0.5 if threshold is None else float(threshold)

        point = compute_point_metrics(
            y_true=clip_labels,
            y_prob=clip_probs,
            threshold=active_threshold,
            n_ece_bins=self.n_ece_bins,
        )

        boot_cis: Dict[str, BootstrapCI] = {}
        if run_bootstrap:
            boot_cis = subject_bootstrap_ci(
                y_true=clip_labels,
                y_prob=clip_probs,
                subject_ids=clip_subjects,
                threshold=active_threshold,
                n_bootstraps=self.n_bootstraps,
                ci_level=self.ci_level,
                n_ece_bins=self.n_ece_bins,
                seed=self.seed,
            )

        return EvaluationReport(
            auc_roc=point["auc_roc"],
            macro_f1=point["macro_f1"],
            fpr=point["fpr"],
            tpr=point["tpr"],
            ece=point["ece"],
            threshold=active_threshold,
            num_windows=len(all_logits),
            num_clips=len(clip_probs),
            num_subjects=len(set(clip_subjects)),
            bootstrap_cis=boot_cis,
        )