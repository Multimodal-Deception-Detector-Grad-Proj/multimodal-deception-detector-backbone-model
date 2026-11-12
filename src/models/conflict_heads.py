from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import EncoderOutputs


@dataclass
class ConflictOutput:
    """Block 10: Level-1 audio-visual physical incongruence outputs."""
    validated: bool = False
    c_trace: Optional[torch.Tensor] = None        # (B, T) frame-level conflict in [0, 2]
    frame_valid: Optional[torch.Tensor] = None    # (B, T) bool mask where both V and A exist
    word_conflict: Optional[torch.Tensor] = None  # (B, L) mean conflict per word span
    word_valid: Optional[torch.Tensor] = None     # (B, L) bool mask where word has valid frames
    align_loss: Optional[torch.Tensor] = None     # Scalar InfoNCE loss (training only)


class ConflictHead(nn.Module):
    """
    Block 10: Level-1 audio-visual synchrony / physical incongruence head.
    Trained via label-free temporal InfoNCE (`align_loss`), with `validated` gated
    via config only after `sync_gap()` passes on held-out validation clips.
    """
    def __init__(
        self,
        d_model: int,
        d_share: int = 64,
        temperature: float = 0.07,
        min_shift: int = 30,          # frames (~1 s at 30 FPS)
        detach_inputs: bool = True,   # stop-gradient into H_v, H_a initially
        validated: bool = False,      # set True from config ONLY after validation checks pass
    ):
        super().__init__()
        self.proj_v = self._make_head(d_model, d_share)
        self.proj_a = self._make_head(d_model, d_share)
        self.tau = temperature
        self.min_shift = min_shift
        self.detach_inputs = detach_inputs
        self.validated = validated

    @staticmethod
    def _make_head(d_in: int, d_out: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.GELU(),
            nn.Linear(d_in, d_out),
        )

    def _shared(self, encoder_outputs: EncoderOutputs, natural_mask: torch.Tensor):
        H_v, H_a = encoder_outputs.H_v, encoder_outputs.H_a
        if self.detach_inputs:
            H_v, H_a = H_v.detach(), H_a.detach()

        v = F.normalize(self.proj_v(H_v).float(), dim=-1)
        a = F.normalize(self.proj_a(H_a).float(), dim=-1)

        both_present = natural_mask[:, 0] & natural_mask[:, 1]             # (B,)
        frame_valid = encoder_outputs.vis_mask & encoder_outputs.aud_mask  # (B, T)
        frame_valid = frame_valid & both_present[:, None]
        return v, a, frame_valid

    def _align_loss(
        self,
        v: torch.Tensor,
        a: torch.Tensor,
        frame_valid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Label-free InfoNCE.
        Positive: same clip, same frame t.
        Negatives: same clip >= min_shift frames away, and other clips at same t.
        """
        B, T, _ = v.shape
        pos = (v * a).sum(-1) / self.tau                                    # (B, T)

        # Within-clip temporal negatives (excluding |t - s| < min_shift)
        sim_w = torch.einsum("btd,bsd->bts", v, a) / self.tau               # (B, T, T)
        idx = torch.arange(T, device=v.device)
        far = (idx[:, None] - idx[None, :]).abs() >= self.min_shift         # (T, T)
        ok_w = far[None] & frame_valid[:, None, :]                          # (B, T, T)
        sim_w = sim_w.masked_fill(~ok_w, float("-inf"))

        # Cross-clip negatives at the same timestamp t
        sim_c = torch.einsum("btd,ctd->btc", v, a) / self.tau               # (B, T, B)
        own = torch.eye(B, dtype=torch.bool, device=v.device)[:, None, :]   # (B, 1, B)
        ok_c = (~own) & frame_valid.t()[None]                               # (B, T, B)
        sim_c = sim_c.masked_fill(~ok_c, float("-inf"))

        logits = torch.cat([pos.unsqueeze(-1), sim_w, sim_c], dim=-1)       # (B, T, 1+T+B)
        loss_bt = torch.logsumexp(logits, dim=-1) - pos
        w = frame_valid.float()
        return (loss_bt * w).sum() / w.sum().clamp(min=1.0)

    def forward(
        self,
        encoder_outputs: EncoderOutputs,
        natural_mask: torch.Tensor,
        active_mask: torch.Tensor,   # kept for signature compatibility; deliberately unused
        alignment_indices: torch.Tensor,
    ) -> ConflictOutput:
        v, a, frame_valid = self._shared(encoder_outputs, natural_mask)
        B, T, _ = v.shape

        c_trace = (1.0 - (v * a).sum(-1)) * frame_valid.float()             # (B, T)

        # Word-level aggregation (vectorized): mean c_trace over each word's valid frames
        t = torch.arange(T, device=v.device)
        start = alignment_indices[..., 0].unsqueeze(-1)
        end = alignment_indices[..., 1].unsqueeze(-1)
        span = (t >= start) & (t <= end) & frame_valid[:, None, :]          # (B, L, T)
        span = span & encoder_outputs.txt_mask[:, :, None]

        cnt = span.sum(-1)                                                  # (B, L)
        word_conflict = (span.float() @ c_trace.unsqueeze(-1)).squeeze(-1) / cnt.clamp(min=1)
        word_valid = cnt > 0

        align_loss = self._align_loss(v, a, frame_valid) if self.training else None

        return ConflictOutput(
            validated=self.validated,
            c_trace=c_trace,
            frame_valid=frame_valid,
            word_conflict=word_conflict,
            word_valid=word_valid,
            align_loss=align_loss,
        )

    @torch.no_grad()
    def sync_gap(
        self,
        encoder_outputs: EncoderOutputs,
        natural_mask: torch.Tensor,
        shift_frames: Optional[int] = None,
    ) -> float:
        """
        Validation check: shifting audio in time should RAISE the conflict score.
        Returns mean(c_shifted) - mean(c_true) over valid non-wrapped frames (> 0 expected).
        """
        shift = shift_frames or self.min_shift
        v, a, valid = self._shared(encoder_outputs, natural_mask)
        if v.shape[1] <= shift:
            return float("nan")

        v_sub = v[:, :-shift]
        a_true = a[:, :-shift]
        a_shift = a[:, shift:]
        valid_pair = valid[:, :-shift] & valid[:, shift:]

        c_true = (1.0 - (v_sub * a_true).sum(-1))[valid_pair]
        c_shift = (1.0 - (v_sub * a_shift).sum(-1))[valid_pair]
        if c_true.numel() == 0:
            return float("nan")
        return (c_shift.mean() - c_true.mean()).item()