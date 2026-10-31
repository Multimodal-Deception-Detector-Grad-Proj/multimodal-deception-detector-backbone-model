import torch
import pytest
from src.models.modality_dropout import ModalityDropoutAndGate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("use_softmax_gate", [True, False])
def test_modality_dropout_and_gate(use_softmax_gate: bool):
    B, T, L, D = 8, 90, 15, 128
    gate = ModalityDropoutAndGate(
        d_model=D,
        p_drop=0.95,  # High drop rate to stress-test the ">= 1 modality" rescue logic
        use_softmax_gate=use_softmax_gate,
    ).to(DEVICE)
    gate.train()

    P_v = torch.randn(B, D, device=DEVICE, requires_grad=True)
    P_a = torch.randn(B, D, device=DEVICE, requires_grad=True)
    P_t = torch.randn(B, D, device=DEVICE, requires_grad=True)

    v_mask = torch.ones(B, T, dtype=torch.bool, device=DEVICE)
    a_mask = torch.ones(B, T, dtype=torch.bool, device=DEVICE)
    t_mask = torch.ones(B, L, dtype=torch.bool, device=DEVICE)

    # Simulate natural missingness (face dropout on sample 0)
    v_mask[0, :] = False

    out = gate(P_v, P_a, P_t, v_mask, a_mask, t_mask)
    loss = out.P.sum()
    loss.backward()

    # Assertions
    assert out.P.shape == (B, D)
    assert out.active_mask.shape == (B, 3)
    assert out.weights.shape == (B, 3)

    # 1. Sample 0 had no valid visual frames, so V must never be active
    assert not out.active_mask[0, 0].item(), "Naturally missing V must be False!"
    assert out.weights[0, 0].item() == 0.0, "Dropped modality weight must be 0!"

    # 2. Every sample had at least one naturally available modality, so >= 1 must remain active
    assert out.active_mask.any(dim=-1).all(), "Every sample must retain >= 1 active modality!"

    # 3. Weights must sum to 1.0 across active modalities
    assert torch.allclose(out.weights.sum(dim=-1), torch.ones(B, device=DEVICE), atol=1e-5)

    # 4. No NaNs in output or gradients
    assert not torch.isnan(out.P).any()
    assert not torch.isnan(P_v.grad).any()