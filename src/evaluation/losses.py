from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.fusion_net import FusionNetOutput


@dataclass
class LossOutput:
    """Structured loss breakdown for backpropagation and metric logging."""
    total_loss: torch.Tensor    # Scalar tensor with grad_fn for .backward()
    main_loss: torch.Tensor     # Detached scalar for logging
    aux_total: torch.Tensor     # Detached unweighted sum of valid aux losses
    aux_visual: torch.Tensor    # Detached scalar for logging
    aux_audio: torch.Tensor     # Detached scalar for logging
    aux_text: torch.Tensor      # Detached scalar for logging


class MultimodalDeceptionLoss(nn.Module):
    
    MODALITY_INDEX = (("visual", 0), ("audio", 1), ("text", 2))

    def __init__(
        self,
        lambda_aux: float = 0.3,
        label_smoothing: float = 0.05,
        apply_smoothing_to_aux: bool = True,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        assert 0.0 <= label_smoothing < 1.0, "label_smoothing must be in [0.0, 1.0)."
        self.lambda_aux = lambda_aux
        self.label_smoothing = label_smoothing
        self.apply_smoothing_to_aux = apply_smoothing_to_aux

        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None

    def _smooth_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """Applies binary label smoothing: y_smooth = y * (1 - eps) + 0.5 * eps."""
        if self.label_smoothing <= 0.0:
            return targets
        return targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

    def forward(self, output: FusionNetOutput, labels: torch.Tensor) -> LossOutput:
        raw_targets = labels.view_as(output.logits).to(
            dtype=output.logits.dtype,
            device=output.logits.device,
        )
        main_targets = self._smooth_targets(raw_targets)
        aux_targets = main_targets if self.apply_smoothing_to_aux else raw_targets

        # Main classifier loss (Block 9)
        main_loss = F.binary_cross_entropy_with_logits(
            output.logits,
            main_targets,
            pos_weight=self.pos_weight,
            reduction="mean",
        )

        # Unimodal auxiliary losses masked by natural_mask (Block 8)
        aux_losses = {}
        aux_sum = torch.zeros_like(main_loss)

        for mod_name, col_idx in self.MODALITY_INDEX:
            mod_logits = output.aux_logits[mod_name]  # (B,)
            valid_mask = output.natural_mask[:, col_idx].to(mod_logits.dtype)  # (B,)

            per_sample_bce = F.binary_cross_entropy_with_logits(
                mod_logits,
                aux_targets,
                pos_weight=self.pos_weight,
                reduction="none",
            )

            valid_count = torch.clamp(valid_mask.sum(), min=1.0)
            mod_loss = (per_sample_bce * valid_mask).sum() / valid_count

            aux_losses[mod_name] = mod_loss
            aux_sum = aux_sum + mod_loss

        total_loss = main_loss + (self.lambda_aux * aux_sum)

        return LossOutput(
            total_loss=total_loss,
            main_loss=main_loss.detach(),
            aux_total=aux_sum.detach(),
            aux_visual=aux_losses["visual"].detach(),
            aux_audio=aux_losses["audio"].detach(),
            aux_text=aux_losses["text"].detach(),
        )