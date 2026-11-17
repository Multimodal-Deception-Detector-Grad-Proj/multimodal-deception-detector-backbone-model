import torch
from .constants import D_TXT, L_TXT, D_AUD, TARGET_T, AUDIO_CHANNEL_INDEX


def check_text_sample(d: dict):
    # 1. Verify text mask and assert that text_mask is a boolean tensor of shape (L_TXT,)
    assert "text_mask" in d, "Missing 'text_mask' in sample dictionary"
    assert d["text_mask"].shape == (L_TXT,), f"text_mask shape {d['text_mask'].shape} != {(L_TXT,)}"
    assert d["text_mask"].dtype == torch.bool, f"text_mask dtype {d['text_mask'].dtype} != torch.bool" 

    # 2. Verify features and alignment indices together
    assert "text_feats" in d, "Missing 'text_feats' in sample dictionary"
    assert "alignment_indices" in d, "Missing 'alignment_indices' in sample dictionary"
    verify_text_tensors(d["text_feats"], d["alignment_indices"])


def verify_text_tensors(text_feats: torch.Tensor, alignment_indices: torch.Tensor):
    # 1. Numerical corruption checks
    assert not torch.isnan(text_feats).any(), "NaN values detected in text extraction!"
    assert not torch.isinf(text_feats).any(), "Inf values detected in text extraction!"

    # 2. Sequence Length & Feature Dimension checks
    assert text_feats.shape[0] == L_TXT, f"Expected L={L_TXT}, got {text_feats.shape[0]}"
    assert alignment_indices.shape[0] == L_TXT, f"Expected L={L_TXT}, got {alignment_indices.shape[0]}"
    assert text_feats.shape[1] == D_TXT, f"Expected D_txt={D_TXT}, got {text_feats.shape[1]}"
    assert alignment_indices.shape[1] == 2, f"Alignment indices must map to [start, end], got {alignment_indices.shape}"

    # 3. Data type assertions (Prevents GPU type mismatch errors)
    assert text_feats.dtype == torch.float32, f"text_feats must be torch.float32, got {text_feats.dtype}"
    assert alignment_indices.dtype == torch.long, f"alignment_indices must be torch.long, got {alignment_indices.dtype}"

    # 4. Temporal Logic Constraints
    assert (alignment_indices >= 0).all(), "Frame indices cannot be negative"
    assert (alignment_indices[:, 0] <= alignment_indices[:, 1]).all(), "Start frames cannot occur after end frames"


def check_audio_sample(d: dict):
    # 1. Verify that the label is a scalar long tensor
    assert "label" in d, "Missing 'label' in sample dictionary"
    assert d["label"].shape == (), f"label shape {tuple(d['label'].shape)} != ()"
    assert d["label"].dtype == torch.long, f"label dtype {d['label'].dtype} != torch.long"

    # 2. Verify features (and the valid mask that lives inside them)
    assert "audio_feats" in d, "Missing 'audio_feats' in sample dictionary"
    verify_audio_tensors(d["audio_feats"])


def verify_audio_tensors(audio_feats: torch.Tensor):
    # 1. Numerical corruption checks
    assert not torch.isnan(audio_feats).any(), "NaN values detected in audio extraction!"
    assert not torch.isinf(audio_feats).any(), "Inf values detected in audio extraction!"

    # 2. Sequence Length & Feature Dimension checks
    assert audio_feats.ndim == 2, f"Expected (T, D_aud), got shape {tuple(audio_feats.shape)}"
    assert audio_feats.shape[0] == TARGET_T, f"Expected T={TARGET_T}, got {audio_feats.shape[0]}"
    assert audio_feats.shape[1] == D_AUD, f"Expected D_aud={D_AUD}, got {audio_feats.shape[1]}"

    # 3. Data type assertions (Prevents GPU type mismatch errors)
    assert audio_feats.dtype == torch.float32, f"audio_feats must be torch.float32, got {audio_feats.dtype}"

    # 4. Valid-mask logic: real frames are a prefix of 1s, padding follows and is exactly zero
    valid = audio_feats[:, AUDIO_CHANNEL_INDEX["valid"]]
    assert ((valid == 0) | (valid == 1)).all(), "valid channel must be binary (0 or 1)"
    n_valid = int(valid.sum())
    assert n_valid >= 1, "Sample has no valid audio frames"
    assert (valid[:n_valid] == 1).all(), "valid frames must come first, padding after"
    assert (audio_feats[n_valid:] == 0).all(), "Padded frames must be exactly zero"