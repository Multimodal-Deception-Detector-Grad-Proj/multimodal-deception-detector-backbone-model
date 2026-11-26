"""Mask-aware pooling for the visual stream (use in unimodal training and fusion)."""

import torch
import torch.nn as nn


class MaskedAttentionPooling(nn.Module):
    """Attention pooling over time that ignores invalid frames.

    x:    (B, T, D) sequence (e.g. GRU outputs)
    mask: (B, T) bool, True = valid frame
    Returns (pooled (B, D), present (B,) bool). Clips with no valid frames get
    a zero vector and present=False (use present to gate / modality-drop in fusion).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        scores = self.score(x).squeeze(-1)                      # (B, T)
        present = mask.any(dim=1)                               # (B,)
        scores = scores.masked_fill(~mask, float("-inf"))
        scores = scores.masked_fill(~present[:, None], 0.0)     # avoid NaN softmax
        w = torch.softmax(scores, dim=1) * mask                 # zero out invalid
        pooled = (w.unsqueeze(-1) * x).sum(dim=1)
        pooled = pooled * present[:, None]
        return pooled, present


# Fusion usage sketch (cross-attention, visual as keys/values):
#   attn_out, _ = mha(query=audio_seq, key=vis_seq, value=vis_seq,
#                     key_padding_mask=~vis_mask)   # True = ignore
#   attn_out = attn_out * vis_present[:, None, None]  # zero if no visual at all
# Note: if a row is fully masked, MultiheadAttention returns NaN, so replace
# those rows' mask with all-False -> all-True (then zero the output via vis_present).