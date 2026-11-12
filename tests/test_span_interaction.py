import torch
import pytest
from src.models.span_interaction import SpanInteraction

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.mark.parametrize("use_span_qkv", [False, True])
@pytest.mark.parametrize("use_attentive_pool", [False, True])
def test_span_interaction(use_span_qkv: bool, use_attentive_pool: bool):
    B, L, T, D = 4, 15, 90, 128
    
    interact = SpanInteraction(
        d_model=D, 
        k_padding=2, 
        use_span_qkv=use_span_qkv,
        use_attentive_pool=use_attentive_pool
    ).to(DEVICE)

    # Mock encoded inputs (H_v, H_a, H_t)
    H_v = torch.randn(B, T, D, device=DEVICE, requires_grad=True)
    H_a = torch.randn(B, T, D, device=DEVICE, requires_grad=True)
    H_t = torch.randn(B, L, D, device=DEVICE, requires_grad=True)

    # Mock masks
    v_mask = torch.ones(B, T, dtype=torch.bool, device=DEVICE)
    a_mask = torch.ones(B, T, dtype=torch.bool, device=DEVICE)
    t_mask = torch.ones(B, L, dtype=torch.bool, device=DEVICE)
    
    # Missingness edges: sample 0 has missing video, sample 1 has missing text
    v_mask[0, :] = False
    t_mask[1, :] = False

    # Mock Active Mask from Block 4 ([V, A, T])
    active_mask = torch.ones(B, 3, dtype=torch.bool, device=DEVICE)
    active_mask[0, 0] = False # Visual naturally missing
    active_mask[2, 1] = False # Audio randomly dropped in Block 4

    # Alignment indices [start, end]
    # Sample 0, token 0: frames [10, 20]
    align = torch.zeros(B, L, 2, dtype=torch.long, device=DEVICE)
    align[:, :, 1] = 5 # arbitrary end
    
    out = interact(H_v, H_a, H_t, align, v_mask, a_mask, t_mask, active_mask)
    
    loss = out.sum()
    loss.backward()

    # 1. Shape check
    assert out.shape == (B, D)

    # 2. Zero-init check (before any weight updates, forward pass must output exact zeros)
    assert (out == 0).all(), "Output must be exactly zero at initialization due to out_proj!"

    # 3. Gradients flow correctly through non-dropped paths
    assert not torch.isnan(H_v.grad).any()
    assert not torch.isnan(H_t.grad).any()

    # 4. Modality Dropout propagation check
    # Since active_mask dropped audio for sample 2, its gradients should be zero
    assert (H_a.grad[2] == 0).all(), "Dropped audio modality must not receive gradients!"