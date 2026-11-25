"""Non-deep baselines on the SAME train/val .pt files. If the CNN cannot beat these, it adds nothing.
python -m src.evaluation.baseline_audio"""
import argparse
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.common.constants import AUDIO_N_MELS, AUDIO_CHANNEL_INDEX as CH
from src.dataset.audio_data import load_split


def summary_feats(x):
    """x (N,T,D) -> per-clip stats of the prosody channels, over real frames (voiced frames for f0/hnr)."""
    out = []
    for a in x.numpy():
        v = a[:, CH["valid"]] > 0
        vo = v & (a[:, CH["voiced"]] > 0)
        def m(c, mask): return a[mask, CH[c]].mean() if mask.any() else np.nan
        def s(c, mask): return a[mask, CH[c]].std() if mask.sum() > 1 else np.nan
        out.append([m("f0_semitone", vo), s("f0_semitone", vo), m("hnr_db", vo), s("hnr_db", vo),
                    m("loudness_db", v), s("loudness_db", v), vo.sum() / max(v.sum(), 1), v.sum()])
    return np.array(out)


def mel_feats(x):
    out = []
    for a in x.numpy():
        v = a[:, CH["valid"]] > 0
        mel = a[v, :AUDIO_N_MELS]
        out.append(np.concatenate([mel.mean(0), mel.std(0)]))
    return np.array(out)


def boot_ci(y, p, n=2000, seed=0):
    r = np.random.default_rng(seed); aucs = []
    for _ in range(n):
        i = r.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            aucs.append(roc_auc_score(y[i], p[i]))
    return np.percentile(aucs, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/processed/audio")
    a = ap.parse_args()
    xtr, ytr, _ = load_split(a.data, "train"); xva, yva, _ = load_split(a.data, "val")
    ytr, yva = ytr.numpy(), yva.numpy()
    names = ["f0_mean", "f0_std", "hnr_mean", "hnr_std", "loud_mean", "loud_std", "voiced_ratio", "n_frames"]
    no_dur = lambda x: summary_feats(x)[:, :-1]            # same features without clip length
    for name, f, C in [("prosody summary (8 feats)", summary_feats, 1.0),
                       ("prosody summary WITHOUT duration (7)", no_dur, 1.0),
                       ("mel mean+std (118 feats)", mel_feats, 0.02)]:
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=C, max_iter=3000))
        m.fit(f(xtr), ytr); p = m.predict_proba(f(xva))[:, 1]
        lo, hi = boot_ci(yva, p)
        print(f"{name:<40} val AUC {roc_auc_score(yva, p):.4f}  95% CI [{lo:.3f}, {hi:.3f}]")
        if f is summary_feats:
            coef = m[-1].coef_[0]
            print("   standardized coefficients (+ = higher for lies):",
                  ", ".join(f"{n} {c:+.2f}" for n, c in sorted(zip(names, coef), key=lambda t: -abs(t[1]))))


if __name__ == "__main__":
    main()
