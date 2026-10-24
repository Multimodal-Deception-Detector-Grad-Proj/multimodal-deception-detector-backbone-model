import torch 
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, d_in: int, d_model: int):
        
        super().__init__()

        self.input_proj = nn.Linear(d_in,d_model)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.unsqueeze(-1).to(x.dtype)
        h = self.ln(self.act(self.input_proj(x*mask_f))) * mask_f
        return h