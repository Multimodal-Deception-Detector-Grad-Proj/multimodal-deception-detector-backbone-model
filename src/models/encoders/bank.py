import torch 
import torch.nn as nn

from dataclasses import dataclass
from .conv_encoder import TemporalConvEncoder
from .text_encoder import TextEncoder
from .pooling import Masked3DPooling

@dataclass
class EncoderOutputs:
    """Structured output from Blocks 2 and 3."""
    H_v: torch.Tensor        # (B, T, D)
    H_a: torch.Tensor        # (B, T, D)
    H_t: torch.Tensor        # (B, L, D)
    P_v_raw: torch.Tensor    # (B, D)
    P_a_raw: torch.Tensor    # (B, D)
    P_t_raw: torch.Tensor    # (B, D)


class TemporalEncoderBank(nn.Module):
    def __init__(self,d_vis: int, d_aud: int, d_txt: int, d_model: int, conv_kernel_size: int = 3):
        super().__init__()

        # Encoders
        self.vis_encoder = TemporalConvEncoder(d_vis,d_model,conv_kernel_size)
        self.aud_encoder = TemporalConvEncoder(d_aud,d_model,conv_kernel_size)
        self.txt_encoder = TextEncoder(d_txt,d_model)

        # Per Modality Pooling
        self.vis_pool = Masked3DPooling(d_model)
        self.aud_pool = Masked3DPooling(d_model)
        self.txt_pool = Masked3DPooling(d_model)

    def forward(self, batch) -> EncoderOutputs:

        get = (lambda k: getattr(batch, k)) if hasattr(batch, "visual_feats") else (lambda k: batch[k])
        vis_feats, vis_mask = get("visual_feats"), get("visual_mask")
        aud_feats, aud_mask = get("audio_feats"), get("audio_mask")
        txt_feats, txt_mask = get("text_feats"), get("text_mask")

        H_v = self.vis_encoder(vis_feats,vis_mask)
        H_a = self.aud_encoder(aud_feats,aud_mask)
        H_t = self.txt_encoder(txt_feats,txt_mask)

        P_v = self.vis_pool(H_v,vis_mask)
        P_a = self.aud_pool(H_a,aud_mask)
        P_t = self.txt_pool(H_t,txt_mask)

        return EncoderOutputs(
            H_v=H_v,
            H_a=H_a,
            H_t=H_t,
            P_v_raw=P_v,
            P_a_raw=P_a,
            P_t_raw=P_t,
        )
