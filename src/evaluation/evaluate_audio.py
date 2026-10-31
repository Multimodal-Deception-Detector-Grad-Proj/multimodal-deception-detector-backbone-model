"""Frozen evaluation of the audio branch: CNN seed-ensemble + logistic-regression reference.
Defaults to VAL. The test split needs --final, so it is not touched by accident; run it once, after everything is frozen.
python -m src.models.audio.evaluate_audio                     # val
python -m src.models.audio.evaluate_audio --split test --final
"""
import argparse, glob, os
import numpy as np, pandas as pd, torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.models.audio.audio_cnn import AudioClassifier
from src.models.audio.baseline_audio import boot_ci, summary_feats
from src.models.audio.train_audio_cnn import load_split, predict


def report(name, y, p):
    lo, hi = boot_ci(y, p)
    print(f"{name:<34} AUC {roc_auc_score(y, p):.4f}  95% CI [{lo:.3f}, {hi:.3f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/audio")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--final", action="store_true", help="required to evaluate the test split")
    ap.add_argument("--ckpt-glob", default="checkpoints/audio_cnn_seed*.pt")
    ap.add_argument("--hidden", type=int, default=48, help="must match training")
    ap.add_argument("--with-mel", action="store_true", help="only if the checkpoints were trained without --prosody-only")
    ap.add_argument("--out", default="outputs/audio_predictions_{split}.csv")
    a = ap.parse_args()
    if a.split == "test" and not a.final:
        raise SystemExit("Refusing to touch the test split without --final. Freeze features/model/checkpoints first.")

    ckpts = sorted(glob.glob(a.ckpt_glob))
    if not ckpts:
        raise SystemExit(f"no checkpoints match {a.ckpt_glob}")
    x, y, ids = load_split(a.data, a.split)
    y = y.numpy()
    print(f"{a.split}: {len(y)} clips (lie {int(y.sum())}) | {len(ckpts)} checkpoints")

    preds = []
    for c in ckpts:
        m = AudioClassifier(d_hidden=a.hidden, prosody_only=not a.with_mel)
        m.load_state_dict(torch.load(c, map_location="cpu"))
        p = predict(m, x); preds.append(p)
        report(os.path.basename(c), y, p)
    p_cnn = np.mean(preds, axis=0)
    report(f"CNN seed-ensemble ({len(ckpts)})", y, p_cnn)

    # logistic-regression reference, fitted on TRAIN only
    xtr, ytr, _ = load_split(a.data, "train")
    lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
    lr.fit(summary_feats(xtr), ytr.numpy())
    p_lr = lr.predict_proba(summary_feats(x))[:, 1]
    report("prosody-summary logistic regression", y, p_lr)

    out = a.out.format(split=a.split)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    pd.DataFrame({"clip_id": ids, "label": y.astype(int), "p_cnn": p_cnn, "p_lr": p_lr}).to_csv(out, index=False)
    print(f"predictions saved to {out}")


if __name__ == "__main__":
    main()