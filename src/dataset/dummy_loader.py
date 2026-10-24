from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader

DEFAULT_T = 90
DEFAULT_L_TXT = 15
DEFAULT_D_VIS = 35
DEFAULT_D_AUD = 128
DEFAULT_D_TXT = 768

@dataclass
class MultimodalBatch:
    """Strongly typed container for downstream pipeline batches."""
    visual_feats: torch.FloatTensor      # (B, T, D_vis)
    visual_mask: torch.BoolTensor        # (B, T)
    audio_feats: torch.FloatTensor       # (B, T, D_aud)
    audio_mask: torch.BoolTensor         # (B, T)
    text_feats: torch.FloatTensor        # (B, L, D_txt)
    text_mask: torch.BoolTensor          # (B, L)
    alignment_indices: torch.LongTensor  # (B, L, 2) -> [start_idx, end_idx]
    label: torch.LongTensor              # (B,)
    subject_id: Optional[list] = None    # Metadata for zero-leakage evaluation

    def to(self, device: torch.device):
        return MultimodalBatch(
            visual_feats=self.visual_feats.to(device),
            visual_mask=self.visual_mask.to(device),
            audio_feats=self.audio_feats.to(device),
            audio_mask=self.audio_mask.to(device),
            text_feats=self.text_feats.to(device),
            text_mask=self.text_mask.to(device),
            alignment_indices=self.alignment_indices.to(device),
            label=self.label.to(device),
            subject_id=self.subject_id,
        )

class DummyMultimodalDataset(Dataset):
    """
    Synthetic dataset generating contract-compliant multimodal samples.
    Simulates variable sequence lengths, padded slots, and missing modalities.
    """
    def __init__(
        self,
        num_samples: int = 64,
        t_frames: int = DEFAULT_T,
        l_tokens: int = DEFAULT_L_TXT,
        d_vis: int = DEFAULT_D_VIS,
        d_aud: int = DEFAULT_D_AUD,
        d_txt: int = DEFAULT_D_TXT,
        allow_empty_modalities: bool = True,
    ):
        self.num_samples = num_samples
        self.t_frames = t_frames
        self.l_tokens = l_tokens
        self.d_vis = d_vis
        self.d_aud = d_aud
        self.d_txt = d_txt
        self.allow_empty_modalities = allow_empty_modalities

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 1. Continuous sequence lengths (Visual & Audio)
        # Random valid frames between [T // 2, T] to simulate variable clip lengths
        valid_t_vis = torch.randint(self.t_frames // 2, self.t_frames + 1, (1,)).item()
        valid_t_aud = torch.randint(self.t_frames // 2, self.t_frames + 1, (1,)).item()

        # Edge case: Simulate total face occlusion for stress-testing NaN pooling
        if self.allow_empty_modalities and (idx % 17 == 0):
            valid_t_vis = 0

        visual_mask = torch.arange(self.t_frames) < valid_t_vis
        audio_mask = torch.arange(self.t_frames) < valid_t_aud

        # Features (zeroed out at padding frames)
        visual_feats = torch.randn(self.t_frames, self.d_vis) * visual_mask.unsqueeze(-1)
        audio_feats = torch.randn(self.t_frames, self.d_aud) * audio_mask.unsqueeze(-1)

        # 2. Linguistic sequence length (Text)
        valid_l = torch.randint(1, self.l_tokens + 1, (1,)).item()
        text_mask = torch.arange(self.l_tokens) < valid_l

        text_feats = torch.randn(self.l_tokens, self.d_txt) * text_mask.unsqueeze(-1)

        # 3. Alignment indices: strictly within [0, T-1] and ordered
        alignment_indices = torch.zeros(self.l_tokens, 2, dtype=torch.long)
        if valid_l > 0:
            # Generate sorted partition points across time T
            cuts = torch.sort(torch.randint(0, self.t_frames, (valid_l * 2,)))[0]
            start_frames = cuts[0::2]
            end_frames = cuts[1::2]

            alignment_indices[:valid_l, 0] = start_frames
            alignment_indices[:valid_l, 1] = end_frames

        # 4. Target label (Binary: 0 = Truthful, 1 = Deceptive)
        label = torch.randint(0, 2, (1,)).squeeze(0).long()

        return {
            "visual_feats": visual_feats,
            "visual_mask": visual_mask,
            "audio_feats": audio_feats,
            "audio_mask": audio_mask,
            "text_feats": text_feats,
            "text_mask": text_mask,
            "alignment_indices": alignment_indices,
            "label": label,
            "subject_id": f"sub_{(idx % 8) + 1:03d}",  # Mock subject clustering
        }

def collate_multimodal_batch(batch_list: list) -> MultimodalBatch:
    """Collates a list of sample dicts into a strongly-typed MultimodalBatch."""
    return MultimodalBatch(
        visual_feats=torch.stack([b["visual_feats"] for b in batch_list]),
        visual_mask=torch.stack([b["visual_mask"] for b in batch_list]),
        audio_feats=torch.stack([b["audio_feats"] for b in batch_list]),
        audio_mask=torch.stack([b["audio_mask"] for b in batch_list]),
        text_feats=torch.stack([b["text_feats"] for b in batch_list]),
        text_mask=torch.stack([b["text_mask"] for b in batch_list]),
        alignment_indices=torch.stack([b["alignment_indices"] for b in batch_list]),
        label=torch.stack([b["label"] for b in batch_list]),
        subject_id=[b["subject_id"] for b in batch_list],
    )


def get_dummy_dataloader(batch_size: int = 8, num_samples: int = 32, **kwargs) -> DataLoader:
    """Helper utility for immediate plug-and-play testing."""
    dataset = DummyMultimodalDataset(num_samples=num_samples, **kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_multimodal_batch,
    )

if __name__ == "__main__":
    loader = get_dummy_dataloader(batch_size=4, num_samples=8)
    batch = next(iter(loader))

    print(f"visual_feats:      {batch.visual_feats.shape} (mask: {batch.visual_mask.shape})")
    print(f"audio_feats:       {batch.audio_feats.shape}  (mask: {batch.audio_mask.shape})")
    print(f"text_feats:        {batch.text_feats.shape}   (mask: {batch.text_mask.shape})")
    print(f"alignment_indices: {batch.alignment_indices.shape}")
    print(f"label:             {batch.label.shape}")

    # Temporal sanity assertions
    valid_aligns = batch.alignment_indices[batch.text_mask]
    assert (valid_aligns[:, 0] <= valid_aligns[:, 1]).all(), "Start frame cannot be after end frame!"
    assert (valid_aligns >= 0).all() and (valid_aligns < DEFAULT_T).all(), "Indices outside [0, T-1]!"
    print("✓ All dummy batch assertions passed.")