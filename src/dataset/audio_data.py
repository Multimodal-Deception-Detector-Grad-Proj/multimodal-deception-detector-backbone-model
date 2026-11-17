"""Loading of the extracted audio samples (.pt files written by extract_audio) and batched prediction."""
import glob

import torch


def load_split(root, split):
    """Returns (audio_feats [N,T,D], labels [N], clip_ids) for data/processed/audio/<split>/*.pt."""
    xs, ys, ids = [], [], []
    for f in sorted(glob.glob(f"{root}/{split}/*.pt")):
        s = torch.load(f)
        xs.append(s["audio_feats"].float()); ys.append(float(s["label"])); ids.append(s["clip_id"])
    return torch.stack(xs), torch.tensor(ys), ids


@torch.no_grad()
def predict(model, x, bs=64):
    """Sigmoid probabilities (numpy) for a stack of samples."""
    model.eval()
    return torch.cat([torch.sigmoid(model(x[i:i + bs])) for i in range(0, len(x), bs)]).numpy()
