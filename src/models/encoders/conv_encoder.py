import torch 
import torch.nn as nn

class TemporalConvEncoder(nn.Module):
    def __init__(self, d_in: int, d_model: int, kernel_size: int):

        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd to preserve sequence length T." 

        self.input_proj = nn.Linear(d_in,d_model)
        self.act = nn.GELU()

        pad_d1 = 1 * (kernel_size - 1) // 2
        pad_d2 = 2 * (kernel_size - 1) // 2

        self.conv_d1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            dilation=1,
            padding=pad_d1,
        )

        self.conv_d2 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            dilation=2,
            padding=pad_d2,
        )

        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.unsqueeze(-1).to(x.dtype)  # (B, T, 1)
        h = self.act(self.input_proj(x * mask_f)) * mask_f  # (B, T, D)

        mask_conv = mask_f.transpose(1, 2)  # (B, 1, T)
        h_conv = h.transpose(1, 2)          # (B, D, T)

        c1 = self.act(self.conv_d1(h_conv)) * mask_conv
        c2 = self.act(self.conv_d2(c1)) * mask_conv

        h_out = self.ln(h + c2.transpose(1,2))

        return h_out * mask_f