from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn

from .encoders import EncoderOutputs


@dataclass
class ConflictOutput:
    """ Level-1 physical incongruence output schema (stub)."""
    validated: bool = False
    av_incongruence: Optional[torch.Tensor] = None
    vt_incongruence: Optional[torch.Tensor] = None
    at_incongruence: Optional[torch.Tensor] = None
    overall_conflict_score: Optional[torch.Tensor] = None


class ConflictHookStub(nn.Module):
    """
    Placeholder hook returning unvalidated conflict schema.
    Signature pre-wired with alignment indices and availability masks
    so MultimodalFusionNet does not need refactoring when Level-1 heads are added.
    """
    def __init__(self):
        super().__init__()

    def forward(
        self,
        encoder_outputs: EncoderOutputs,
        natural_mask: torch.Tensor,
        active_mask: torch.Tensor,
        alignment_indices: torch.Tensor,
    ) -> ConflictOutput:
        return ConflictOutput(validated=False)