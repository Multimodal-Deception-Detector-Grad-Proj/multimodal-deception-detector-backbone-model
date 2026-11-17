"""Train-only normalization for visual feature sequences."""
from dataclasses import dataclass
import torch


@dataclass
class VisualNormalizer:
    """Fit statistics on training data only, then reuse for val/test."""

    eps: float = 1e-6
    mean: torch.Tensor | None = None
    std: torch.Tensor | None = None

    def fit(self, train_feats: torch.Tensor):
        if train_feats.ndim != 3:
            raise ValueError("Expected (B, T, D) training tensor")
        if not torch.isfinite(train_feats).all():
            raise ValueError("Training features contain NaN or Inf")
        x = train_feats.float()
        self.mean = x.mean(dim=(0, 1), keepdim=True)
        self.std = x.std(dim=(0, 1), unbiased=False, keepdim=True).clamp_min(self.eps)
        return self

    def transform(self, feats: torch.Tensor):
        if self.mean is None or self.std is None:
            raise RuntimeError("Call fit() on training data before transform()")
        if feats.ndim != 3 or feats.shape[-1] != self.mean.shape[-1]:
            raise ValueError("Feature dimension mismatch")
        if not torch.isfinite(feats).all():
            raise ValueError("Features contain NaN or Inf")
        return (feats.float() - self.mean.to(feats.device)) / self.std.to(feats.device)

    def fit_transform(self, train_feats: torch.Tensor):
        return self.fit(train_feats).transform(train_feats)

    def state_dict(self):
        if self.mean is None or self.std is None:
            raise RuntimeError("Normalizer has not been fitted")
        return {"mean": self.mean, "std": self.std, "eps": self.eps}

    def load_state_dict(self, state):
        self.mean = state["mean"]
        self.std = state["std"]
        self.eps = state.get("eps", self.eps)
        return self
