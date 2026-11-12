from dataclasses import dataclass
from typing import Dict, Optional
import torch
import torch.nn as nn

from .encoders import TemporalEncoderBank
from .modality_dropout import ModalityDropoutAndGate
from .span_interaction import SpanInteraction
from .conflict_heads import ConflictHookStub, ConflictOutput


@dataclass
class FusionNetOutput:
    """Complete forward pass output container for loss computation and inference."""
    logits: torch.Tensor                        # (B,) Main deception logit (Block 7)
    aux_logits: Dict[str, torch.Tensor]         # {"visual": (B,), "audio": (B,), "text": (B,)} (Block 8)
    Z: torch.Tensor                             # (B, D) Fused latent representation (Block 6)
    P: torch.Tensor                             # (B, D) Global modality-gated representation (Block 4)
    S: torch.Tensor                             # (B, D) Span interaction representation (Block 5)
    gate_g: Optional[torch.Tensor]              # (B, D) Residual gate values (None if interaction off)
    modality_weights: torch.Tensor            # (B, 3) Weights [V, A, T] from Block 4
    natural_mask: torch.Tensor                # (B, 3) Naturally available modalities [V, A, T]
    active_mask: torch.Tensor                 # (B, 3) Shared post-dropout availability mask [V, A, T]
    conflict: ConflictOutput                    # Block 10 schema stub (validated=False)


class MultimodalFusionNet(nn.Module):
    """
    End-to-end downstream multimodal deception architecture (Blocks 2–7 & 10):
      - Blocks 2 & 3: TemporalEncoderBank -> H_m, P_m_raw
      - Block 4: ModalityDropoutAndGate -> P, active_mask
      - Block 5: SpanInteraction -> S (optional via `use_interaction`)
      - Block 6: Residual fusion g = sigmoid(W_g[P; S] + b_g), Z = P + g * S (or Z = P)
      - Block 7: Classifier LN -> Linear(D -> D/2) -> GELU -> Dropout -> Linear(D/2 -> 1)
      - Block 8 support: Unimodal aux heads reading undropped P_m_raw
      - Block 10: Conflict hook stub
    """
    def __init__(
        self,
        d_vis: int,
        d_aud: int,
        d_txt: int,
        d_model: int = 128,
        conv_kernel_size: int = 3,
        p_drop: float = 0.2,
        use_softmax_gate: bool = True,
        use_interaction: bool = True,
        k_padding: int = 2,
        use_span_qkv: bool = False,
        use_attentive_pool: bool = True,
        classifier_dropout: float = 0.2,
    ):
        super().__init__()
        assert d_model >= 2, "d_model must be at least 2 for D -> D/2 classifier bottleneck."

        self.d_model = d_model
        self.use_interaction = use_interaction

        # Blocks 2 & 3: Modality encoders and NaN-safe masked pooling
        self.encoder_bank = TemporalEncoderBank(
            d_vis=d_vis,
            d_aud=d_aud,
            d_txt=d_txt,
            d_model=d_model,
            conv_kernel_size=conv_kernel_size,
        )

        # Block 4: Shared modality dropout and global gating
        self.modality_gate = ModalityDropoutAndGate(
            d_model=d_model,
            p_drop=p_drop,
            use_softmax_gate=use_softmax_gate,
        )

        # Block 5 & Block 6: Word-span interaction path and residual fusion gate
        if self.use_interaction:
            self.span_interaction = SpanInteraction(
                d_model=d_model,
                k_padding=k_padding,
                use_span_qkv=use_span_qkv,
                use_attentive_pool=use_attentive_pool,
            )
            # g = sigmoid(W_g[P; S] + b_g)
            self.residual_gate = nn.Linear(2 * d_model, d_model)

        # Block 7: Main classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(d_model // 2, 1),
        )

        # Block 8 support: Auxiliary unimodal heads reading undropped P_m_raw
        self.aux_head_v = nn.Linear(d_model, 1)
        self.aux_head_a = nn.Linear(d_model, 1)
        self.aux_head_t = nn.Linear(d_model, 1)

        # Block 10: Conflict hook stub
        self.conflict_hook = ConflictHookStub()

    def forward(self, batch) -> FusionNetOutput:
        """
        Accepts either a MultimodalBatch dataclass or a dict matching the contract.
        """
        get = (lambda k: getattr(batch, k)) if hasattr(batch, "visual_feats") else (lambda k: batch[k])

        visual_mask = get("visual_mask")
        audio_mask = get("audio_mask")
        text_mask = get("text_mask")
        alignment_indices = get("alignment_indices")

        # Blocks 2 & 3: Encode sequences & compute undropped P_m_raw
        enc = self.encoder_bank(batch)

        # Block 4: Sample shared active_mask and compute global path P
        gate_out = self.modality_gate(
            P_v_raw=enc.P_v_raw,
            P_a_raw=enc.P_a_raw,
            P_t_raw=enc.P_t_raw,
            visual_mask=visual_mask,
            audio_mask=audio_mask,
            text_mask=text_mask,
        )
        P = gate_out.P

        # Blocks 5 & 6: Interaction path S and Residual Fusion Z = P + g * S
        if self.use_interaction:
            S = self.span_interaction(
                H_v=enc.H_v,
                H_a=enc.H_a,
                H_t=enc.H_t,
                alignment_indices=alignment_indices,
                visual_mask=visual_mask,
                audio_mask=audio_mask,
                text_mask=text_mask,
                active_mask=gate_out.active_mask,
            )
            g = torch.sigmoid(self.residual_gate(torch.cat([P, S], dim=-1)))
            Z = P + g * S
        else:
            S = torch.zeros_like(P)
            g = None
            Z = P

        # Block 7: Main binary deception logit (squeeze(-1) preserves (B,) even when B=1)
        logits = self.classifier(Z).squeeze(-1)

        # Block 8: Unimodal auxiliary logits from undropped P_m_raw
        aux_logits = {
            "visual": self.aux_head_v(enc.P_v_raw).squeeze(-1),
            "audio": self.aux_head_a(enc.P_a_raw).squeeze(-1),
            "text": self.aux_head_t(enc.P_t_raw).squeeze(-1),
        }

        # Block 10: Unvalidated conflict schema hook
        conflict = self.conflict_hook(
            encoder_outputs=enc,
            natural_mask=gate_out.natural_mask,
            active_mask=gate_out.active_mask,
            alignment_indices=alignment_indices,
        )

        return FusionNetOutput(
            logits=logits,
            aux_logits=aux_logits,
            Z=Z,
            P=P,
            S=S,
            gate_g=g,
            modality_weights=gate_out.weights,
            natural_mask=gate_out.natural_mask,
            active_mask=gate_out.active_mask,
            conflict=conflict,
        )