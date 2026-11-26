"""All plotting, kept out of the pipeline classes."""

from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

LABEL_COLORS = {"Lie": "#E74C3C", "Truth": "#2ECC71", "Unknown": "#95A5A6"}


def _finish(fig, show: bool):
    fig.tight_layout()
    if show:
        plt.show()
    return fig


class DatasetPlots:
    @staticmethod
    def duration_and_chunks(plan_df: pd.DataFrame, chunk_length_sec: float,
                            overlap_sec: float, show: bool = True):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        sns.histplot(plan_df["duration_sec"], bins=40, kde=True, ax=axes[0], color="teal")
        axes[0].set(title="Video Duration Distribution", xlabel="Duration (seconds)",
                    ylabel="Number of Videos")
        counts = plan_df["projected_chunks"].astype(str)
        order = sorted(counts.unique(), key=int)
        sns.countplot(x=counts, hue=counts, order=order, hue_order=order, ax=axes[1],
                      palette="magma", legend=False)
        axes[1].set(title=f"Projected Chunks per Video (Window: {chunk_length_sec}s, "
                          f"Overlap: {overlap_sec}s)",
                    xlabel="Number of Chunks Generated", ylabel="Frequency")
        return _finish(fig, show)

    @staticmethod
    def class_balance(meta_df: pd.DataFrame, show: bool = True):
        fig, ax = plt.subplots(figsize=(8, 5))
        order = [l for l in LABEL_COLORS if l in set(meta_df["label"])]
        sns.countplot(data=meta_df, x="label", hue="label", order=order, hue_order=order,
                      palette=LABEL_COLORS, legend=False, ax=ax)
        ax.set(title="Dataset Class Balance: Truth vs. Lie", xlabel="Ground Truth Label",
               ylabel="Number of Video Clips")
        return _finish(fig, show)


class QualityPlots:
    @staticmethod
    def dashboard(q: pd.DataFrame, jitter_thresh: float, show: bool = True):
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        sns.histplot(q["dropout_pct"], bins=30, kde=True, ax=axes[0, 0], color="steelblue")
        axes[0, 0].set(title="OpenFace Dropout Rate Distribution",
                       xlabel="Frames Below Confidence Threshold (%)", ylabel="Scene Count")
        sns.histplot(q["mean_confidence"], bins=30, kde=True, ax=axes[0, 1], color="darkgreen")
        axes[0, 1].set(title="Mean Tracking Confidence", xlabel="Average Confidence",
                       ylabel="Scene Count")
        jit = q[["j_tx", "j_ty", "j_tz"]].melt(var_name="Axis", value_name="Jitter")
        sns.boxplot(data=jit, x="Axis", y="Jitter", hue="Axis", palette="Set2", legend=False,
                    ax=axes[1, 0])
        axes[1, 0].axhline(jitter_thresh, color="red", linestyle="--", linewidth=1.5,
                           label=f"Threshold = {jitter_thresh}")
        axes[1, 0].legend()
        axes[1, 0].set(title="Head Translation Jitter by Axis", xlabel="Translation Axis",
                       ylabel="RMS Jitter (mm)")
        gz = q[["gaze_x_std_deg", "gaze_y_std_deg"]].melt(var_name="Axis", value_name="StdDeg")
        sns.boxplot(data=gz, x="Axis", y="StdDeg", hue="Axis", palette="viridis", legend=False,
                    ax=axes[1, 1])
        axes[1, 1].set(title="Gaze Dispersion by Axis", xlabel="Gaze Axis",
                       ylabel="Std. Dev. (Degrees)")
        return _finish(fig, show)

    @staticmethod
    def gate(q: pd.DataFrame, show: bool = True):
        counts = q["quality_gate_pass"].map({True: "Pass", False: "Fail"}).value_counts()
        fig, ax = plt.subplots(figsize=(7, 5))
        order = [s for s in ("Pass", "Fail") if s in counts.index]
        sns.barplot(x=counts.index, y=counts.values, hue=counts.index, order=order,
                    hue_order=order, palette={"Pass": "#2ecc71", "Fail": "#e74c3c"},
                    legend=False, ax=ax)
        ax.set(title="OpenFace Quality Gate Result", xlabel="Scene Status", ylabel="Scene Count")
        return _finish(fig, show)

    @staticmethod
    def gate_by_label(q: pd.DataFrame, show: bool = True):
        """Pass rate per class: a gap here means filtering would unbalance the dataset."""
        rate = q.groupby("label")["quality_gate_pass"].mean().mul(100).reset_index()
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.barplot(data=rate, x="label", y="quality_gate_pass", hue="label",
                    palette=LABEL_COLORS, legend=False, ax=ax)
        ax.set(title="Quality Gate Pass Rate by Class", xlabel="Label", ylabel="Pass rate (%)",
               ylim=(0, 100))
        return _finish(fig, show)

    @staticmethod
    def unstable(q: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        cols = ["video", "label", "scene_index", "dropout_pct", "mean_confidence", "j_tx", "j_ty",
                "j_tz", "head_yaw_std_deg", "gaze_x_std_deg", "gaze_y_std_deg",
                "quality_gate_pass"]
        return q.sort_values(["dropout_pct", "j_tx", "j_ty", "j_tz", "gaze_x_std_deg",
                              "gaze_y_std_deg"], ascending=False).head(n)[cols]


class EmotionPlots:
    """Inputs: probs (N, T, K), labels (K names), conditions (N,) such as 'Lie'/'Truth'."""

    @staticmethod
    def summary(probs: np.ndarray, labels: Sequence[str]) -> Dict[str, float]:
        flat = probs.reshape(-1, probs.shape[-1])
        mean = flat.mean(axis=0)
        frame_entropy = -(flat * np.log2(flat + 1e-9)).sum(axis=1)
        return {
            "global_entropy_bits": float(-(mean * np.log2(mean + 1e-9)).sum()),
            "max_entropy_bits": float(np.log2(len(labels))),
            "mean_frame_entropy_bits": float(frame_entropy.mean()),
            "mean_top_prob": float(flat.max(axis=1).mean()),
            "dominant_emotion": labels[int(mean.argmax())],
        }

    @staticmethod
    def global_means(probs, labels, show: bool = True):
        flat = pd.DataFrame(probs.reshape(-1, len(labels)), columns=list(labels))
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=flat, errorbar="sd", palette="viridis", capsize=0.1, ax=ax)
        ax.set(title="Global Mean Emotion Probabilities Across All Frames (± SD)",
               ylabel="Mean Probability", ylim=(0, 1.0))
        return _finish(fig, show)

    @staticmethod
    def by_condition(probs, labels, conditions, show: bool = True):
        clip = pd.DataFrame(probs.mean(axis=1), columns=list(labels))
        clip["Condition"] = np.asarray(conditions)
        melted = clip.melt(id_vars="Condition", var_name="Emotion", value_name="Mean Clip Probability")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=melted, x="Emotion", y="Mean Clip Probability", hue="Condition",
                    palette=LABEL_COLORS, showfliers=False, ax=ax)
        ax.set_title("Emotion Distributions by Ground Truth")
        return _finish(fig, show)

    @staticmethod
    def trajectory(probs, labels, conditions, index: Optional[int] = None,
                   seed: Optional[int] = None, show: bool = True):
        idx = int(index if index is not None else np.random.default_rng(seed).integers(len(probs)))
        fig, ax = plt.subplots(figsize=(14, 5))
        for k, name in enumerate(labels):
            ax.plot(probs[idx, :, k], label=name, linewidth=2, alpha=0.8)
        ax.set(title=f"Emotion trajectory over {probs.shape[1]} frames "
                     f"(clip {idx} | {conditions[idx]})",
               xlabel="Frame index (t)", ylabel="Probability", ylim=(0, 1.05))
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
        return _finish(fig, show)

    @staticmethod
    def temporal_stability(probs, conditions, show: bool = True):
        """Mean |p(t) - p(t-1)| per clip. High values = flickering predictions."""
        flicker = np.abs(np.diff(probs, axis=1)).sum(axis=2).mean(axis=1)
        df = pd.DataFrame({"Condition": np.asarray(conditions), "Mean frame-to-frame change": flicker})
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df, x="Condition", y="Mean frame-to-frame change", hue="Condition",
                    palette=LABEL_COLORS, showfliers=False, legend=False, ax=ax)
        ax.set_title("Temporal Stability of Emotion Predictions")
        return _finish(fig, show)
