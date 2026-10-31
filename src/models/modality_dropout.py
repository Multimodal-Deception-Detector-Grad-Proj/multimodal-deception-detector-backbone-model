import torch 
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class ModalityGateOutput:
    P: torch.Tensor
    active_mask: torch.Tensor
    weights: torch.Tensor

class ModalityDropoutAndGate(nn.Module):
    def __init__(self, d_model: int = 128, p_drop: float = 0.2, use_softmax_gate: bool = True):
        super().__init__()
        self.d_model = d_model
        self.p_drop = p_drop
        self.use_softmax_gate = use_softmax_gate

        if self.use_softmax_gate:
            self.gate_scorers = nn.ModuleList([
                nn.Linear(d_model, 1),  # Visual logit
                nn.Linear(d_model, 1),  # Audio logit
                nn.Linear(d_model, 1),  # Text logit
            ])

    def _sample_active_mask(self, natural_mask: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p_drop <= 0.0:
            return natural_mask

        B,M = natural_mask.shape
        device = natural_mask.device

        keep_draw = torch.rand(B, M, device=device) >= self.p_drop
        active_mask = natural_mask & keep_draw
        all_dropped = (~active_mask.any(dim=-1)) & natural_mask.any(dim=-1)

        if all_dropped.any():
            rescue_scores = torch.rand(B, M, device=device) + 0.1
            rescue_scores = rescue_scores * natural_mask.float()
            rescue_idx = rescue_scores.argmax(dim=-1)  # (B,)

            rescue_one_hot = F.one_hot(rescue_idx, num_classes=M).bool()  # (B, 3)
            active_mask = torch.where(
                all_dropped.unsqueeze(-1),
                rescue_one_hot,
                active_mask,
            )

        return active_mask

    def forward(
        self,
        P_v_raw: torch.Tensor,
        P_a_raw: torch.Tensor,
        P_t_raw: torch.Tensor,
        visual_mask: torch.Tensor,
        audio_mask: torch.Tensor,
        text_mask: torch.Tensor,
        ) -> ModalityGateOutput:
        
        v_avail = visual_mask.any(dim=-1)
        a_avail = audio_mask.any(dim=-1)
        t_avail = text_mask.any(dim=-1)
        natural_mask = torch.stack([v_avail, a_avail, t_avail], dim=-1)  # (B, 3)

        active_mask = self._sample_active_mask(natural_mask)
        P_stack = torch.stack([P_v_raw, P_a_raw, P_t_raw], dim=1) 
        has_any_active = active_mask.any(dim=-1, keepdim=True) # Guard for case where all 3 modalities are naturally empty

        if self.use_softmax_gate:
            logits_v = self.gate_scorers[0](P_v_raw)  # (B, 1)
            logits_a = self.gate_scorers[1](P_a_raw)  # (B, 1)
            logits_t = self.gate_scorers[2](P_t_raw)  # (B, 1)
            logits = torch.cat([logits_v, logits_a, logits_t], dim=-1)  # (B, 3)

            masked_logits = logits.masked_fill(~active_mask, float("-inf"))

            safe_logits = torch.where(has_any_active, masked_logits, torch.zeros_like(masked_logits))
            weights = F.softmax(safe_logits, dim=-1)
            weights = weights * active_mask.to(P_stack.dtype)
        else:
            # Uniform renormalized average over active modalities
            active_f = active_mask.to(P_stack.dtype)
            denom = torch.clamp(active_f.sum(dim=-1, keepdim=True), min=1.0)
            weights = active_f / denom

        P = (weights.unsqueeze(-1) * P_stack).sum(dim=1)
        
        return ModalityGateOutput(
            P=P,
            active_mask=active_mask,
            weights=weights,
        )
