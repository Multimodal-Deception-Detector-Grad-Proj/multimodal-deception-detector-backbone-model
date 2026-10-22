import torch
from .constants import D_TXT, L_TXT

def check_text_sample(d: dict):
    f = d["text_feats"]
    assert f.shape == (L_TXT, D_TXT), f"text_feats {tuple(f.shape)} != {(L_TXT, D_TXT)}"
    assert not torch.isnan(f).any() and not torch.isinf(f).any()
    assert d["text_mask"].shape == (L_TXT,) and d["text_mask"].dtype == torch.bool # Assert that text_mask is a boolean tensor of shape (L_TXT,)
    
    
def verify_text_tensors(text_feats: torch.Tensor, alignment_indices: torch.Tensor):
    # 1. Check for numerical corruption
    assert not torch.isnan(text_feats).any(), "NaN values detected in text extraction!"
    assert not torch.isinf(text_feats).any(), "Inf values detected in text extraction!"
    
    # 2. Check Sequence Length (L)
    assert text_feats.shape[0] == L_TXT, f"Expected L={L_TXT}, got {text_feats.shape[0]}"
    assert alignment_indices.shape[0] == L_TXT, f"Expected L={L_TXT}, got {alignment_indices.shape[0]}"
    
    # 3. Check Embedding Dimension (D_txt)
    assert text_feats.shape[1] == D_TXT, f"Expected D_txt={D_TXT}, got {text_feats.shape[1]}"
    
    # 4. Check Alignment indices shape (L, 2)
    assert alignment_indices.shape[1] == 2, f"Alignment indices must map to [start, end], got shape {alignment_indices.shape}"
    # 5. Temporal Logic Constraints
    assert (alignment_indices >= 0).all(), "Frame indices cannot be negative"
    assert (alignment_indices[:, 0] <= alignment_indices[:, 1]).all(), "Start frames cannot occur after end frames"