import torch
import torch.nn as nn

class Masked3DPooling(nn.Module):
    def __init__(self,d_model:int, eps:float=1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.proj = nn.Linear(3*d_model, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        mask_f = mask.unsqueeze(-1).to(x.dtype)  # (B, T, 1)
        valid_count = mask_f.sum(dim=1)  # (B, 1)
        has_valid = (valid_count > 0).to(x.dtype)  # (B, 1)   
        safe_denom = torch.clamp(valid_count, min=1.0)  

        # Masked mean
        masked_x = x * mask_f 
        mean = masked_x.sum(dim=1) / safe_denom  # (B, D)

        # Masked std
        diff = (x - mean.unsqueeze(1)) * mask_f
        var = (diff**2).sum(dim=1 ) / safe_denom
        std = torch.sqrt(var + self.eps)*has_valid  # (B, D)

        # Masked max
        neg_inf = torch.finfo(x.dtype).min / 2.0
        x_masked_for_max = x.masked_fill(~mask.unsqueeze(-1), neg_inf)
        max_val = x_masked_for_max.max(dim=1).values * has_valid # (B, D)

        # Projection
        pooled_cat = torch.cat([mean,std,max_val],dim=1)
        out = self.ln(self.proj(pooled_cat))

        return out * has_valid

