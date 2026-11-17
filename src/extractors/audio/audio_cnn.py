"""1D-CNN audio encoder.

Input : audio_feats (B, T, D_AUD)  -- channel AUDIO_CHANNEL_INDEX["valid"] marks real frames.
Output: pooled (B, d_latent) for late fusion, and seq (B, T, d_hidden) for cross-attention fusion.
GroupNorm (not BatchNorm) so zero-padded frames never contaminate the statistics; features are
re-masked after every block so padding stays exactly zero.
"""
import torch
import torch.nn as nn

from src.common.constants import AUDIO_CHANNEL_INDEX, AUDIO_EXTRA_CHANNELS, D_AUD, TARGET_T

D_LATENT = 256          # PLACEHOLDER: must equal the fusion latent size agreed with the multimodal head


class _Block(nn.Module):
    def __init__(self, c_in, c_out, k=5, dilation=1, p_drop=0.1):
        super().__init__()
        pad = dilation * (k - 1) // 2
        self.conv = nn.Conv1d(c_in, c_out, k, padding=pad, dilation=dilation)
        self.norm = nn.GroupNorm(8, c_out)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p_drop)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, m):                       # x (B,C,T), m (B,1,T)
        y = self.drop(self.act(self.norm(self.conv(x))))
        return (y + self.skip(x)) * m


class AudioCNN(nn.Module):
    def __init__(self, d_in=D_AUD, d_hidden=128, d_latent=D_LATENT, p_drop=0.2, prosody_only=False):
        super().__init__()
        self.valid_idx = AUDIO_CHANNEL_INDEX["valid"]
        # `valid` is the mask, never an input. prosody_only drops the mel bins and keeps f0/voiced/loudness/hnr.
        extra = [AUDIO_CHANNEL_INDEX[c] for c in AUDIO_EXTRA_CHANNELS if c != "valid"]
        idx = extra if prosody_only else [i for i in range(d_in) if i != self.valid_idx]
        self.register_buffer("input_idx", torch.tensor(idx), persistent=False)
        self.stem = nn.Conv1d(len(idx), d_hidden, 1)
        self.blocks = nn.ModuleList([
            _Block(d_hidden, d_hidden, dilation=1, p_drop=p_drop),
            _Block(d_hidden, d_hidden, dilation=2, p_drop=p_drop),
            _Block(d_hidden, d_hidden, dilation=4, p_drop=p_drop),   # receptive field 29 frames = 1.16 s at 25 fps
        ])
        self.proj = nn.Sequential(nn.Linear(2 * d_hidden, d_latent), nn.GELU(), nn.Dropout(p_drop))
        self.d_hidden = d_hidden

    def forward(self, audio_feats):
        assert audio_feats.ndim == 3 and audio_feats.shape[-1] == D_AUD, audio_feats.shape
        m = audio_feats[..., self.valid_idx].unsqueeze(1)                  # (B,1,T)
        x = audio_feats[..., self.input_idx].transpose(1, 2) * m           # (B,C_in,T)
        x = self.stem(x) * m
        for b in self.blocks:
            x = b(x, m)
        n = m.sum(-1).clamp(min=1)                                         # (B,1)
        mean = x.sum(-1) / n
        var = (((x - mean.unsqueeze(-1)) * m) ** 2).sum(-1) / n
        pooled = self.proj(torch.cat([mean, var.add(1e-6).sqrt()], dim=-1))  # masked mean + std
        return pooled, x.transpose(1, 2)


class AudioClassifier(nn.Module):
    """Audio-only baseline head (also used for the overfit check)."""
    def __init__(self, **kw):
        super().__init__()
        self.encoder = AudioCNN(**kw)
        self.head = nn.Linear(self.encoder.proj[0].out_features, 1)

    def forward(self, audio_feats):
        z, _ = self.encoder(audio_feats)
        return self.head(z).squeeze(-1)                                    # logits (B,)