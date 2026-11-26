from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class RNNClassifier(nn.Module):
    """(B, T, D) [+ (B, T) bool mask] -> logits (B, num_classes), swappable LSTM/GRU encoder.

    mask: True = valid frame. Invalid frames are replaced by a learned "missing" embedding before
    the RNN (so zeroed features aren't read as real data) and are excluded from pooling.
    mask=None behaves like the original unmasked model.

    For multimodal fusion use `encode()`, which returns the pooled clip embedding and (optionally)
    the per-frame sequence, without the classification head.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, rnn_type: str = "lstm",
                 bidirectional: bool = True, dropout: float = 0.3, pooling: str = "attention", num_classes: int = 2):
        super().__init__()
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[rnn_type.lower()]
        self.pooling = pooling
        self.proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.missing = nn.Parameter(torch.zeros(hidden_dim))
        self.rnn = rnn_cls(hidden_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional,
                           dropout=dropout if num_layers > 1 else 0.0)
        out = hidden_dim * (2 if bidirectional else 1)
        self.embedding_dim = out
        self.attn = nn.Linear(out, 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(out, out // 2), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(out // 2, num_classes))

    # ------------------------------------------------------------------ encoder
    def encode(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
               return_sequence: bool = False) -> Dict[str, torch.Tensor]:
        """Returns a dict:
            pooled   (B, E)    clip embedding (zero vector for clips with no valid frames)
            present  (B,)      bool, clip has >= 1 valid frame (all True when mask is None)
            sequence (B, T, E) per-frame states, only if return_sequence=True
            mask     (B, T)    the frame mask used (all True when mask is None)
        """
        B, T = x.shape[:2]
        z = self.proj(x)                                        # (B, T, H)
        if mask is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=x.device)
            unmasked = True
        else:
            mask = mask.bool()
            z = torch.where(mask.unsqueeze(-1), z, self.missing.to(z.dtype).expand_as(z))
            unmasked = False
        h, _ = self.rnn(z)                                      # (B, T, E)

        present = mask.any(dim=1)                               # (B,)
        if unmasked:
            if self.pooling == "attention":
                w = torch.softmax(self.attn(h).squeeze(-1), dim=1)
                pooled = (h * w.unsqueeze(-1)).sum(1)
            elif self.pooling == "mean":
                pooled = h.mean(1)
            else:
                pooled = h[:, -1]
        else:
            m = mask.unsqueeze(-1).to(h.dtype)                  # (B, T, 1)
            if self.pooling == "attention":
                s = self.attn(h).squeeze(-1).masked_fill(~mask, float("-inf"))
                s = s.masked_fill(~present[:, None], 0.0)       # all-invalid rows: avoid NaN softmax
                w = torch.softmax(s, dim=1).unsqueeze(-1) * m
                pooled = (h * w).sum(1)
            elif self.pooling == "mean":
                pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
            else:                                               # last *valid* frame
                idx = (T - 1 - mask.flip(1).float().argmax(dim=1)).clamp(min=0)
                pooled = h[torch.arange(B, device=h.device), idx]
            pooled = pooled * present[:, None].to(pooled.dtype)  # no valid frames -> zero vector

        out = {"pooled": pooled, "present": present, "mask": mask}
        if return_sequence:
            out["sequence"] = h
        return out

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.head(self.encode(x, mask)["pooled"])

    @classmethod
    def from_config(cls, cfg, input_dim: int) -> "RNNClassifier":
        return cls(input_dim, cfg.hidden_dim, cfg.num_layers, cfg.rnn_type, cfg.bidirectional,
                   cfg.dropout, cfg.pooling)