from __future__ import annotations

import torch
import torch.nn as nn


class RNNClassifier(nn.Module):
    """(B, T, D) -> logits (B, num_classes) with a swappable LSTM/GRU encoder."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, rnn_type: str = "lstm",
                 bidirectional: bool = True, dropout: float = 0.3, pooling: str = "attention", num_classes: int = 2):
        super().__init__()
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[rnn_type.lower()]
        self.pooling = pooling
        self.proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.rnn = rnn_cls(hidden_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional,
                           dropout=dropout if num_layers > 1 else 0.0)
        out = hidden_dim * (2 if bidirectional else 1)
        self.attn = nn.Linear(out, 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(out, out // 2), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(out // 2, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(self.proj(x))                       # (B, T, H*dirs)
        if self.pooling == "attention":
            w = torch.softmax(self.attn(h).squeeze(-1), dim=1)
            z = (h * w.unsqueeze(-1)).sum(1)
        elif self.pooling == "mean":
            z = h.mean(1)
        else:
            z = h[:, -1]
        return self.head(z)

    @classmethod
    def from_config(cls, cfg, input_dim: int) -> "RNNClassifier":
        return cls(input_dim, cfg.hidden_dim, cfg.num_layers, cfg.rnn_type, cfg.bidirectional,
                   cfg.dropout, cfg.pooling)
