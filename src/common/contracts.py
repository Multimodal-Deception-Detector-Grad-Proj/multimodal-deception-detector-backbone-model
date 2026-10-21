import torch
from .constants import D_TXT, L_TXT

def check_text_sample(d: dict):
    f = d["text_feats"]
    assert f.shape == (L_TXT, D_TXT), f"text_feats {tuple(f.shape)} != {(L_TXT, D_TXT)}"
    assert not torch.isnan(f).any() and not torch.isinf(f).any()
    assert d["text_mask"].shape == (L_TXT,) and d["text_mask"].dtype == torch.bool