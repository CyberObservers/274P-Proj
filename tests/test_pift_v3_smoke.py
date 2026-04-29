"""End-to-end smoke test for PIFT v3 (NSI integrated): build, forward, backward.
Catches "frozen KAN" bugs (NSI in module but not receiving any grad)."""
from __future__ import annotations

import torch

from src.data.feature_groups import get_groups
from src.models.pift import PIFT


def test_pift_v3_forward_shape_higgs_low():
    spec = get_groups("higgs", "low_level")
    model = PIFT(spec, d_out=1, use_nsi=True, nsi_K=16, feature_dim=21)
    model.nsi.set_scaler_stats(torch.zeros(21), torch.ones(21))
    x = torch.randn(8, 21)
    y = model(x)
    assert y.shape == (8, 1)
    assert torch.isfinite(y).all()


def test_pift_v3_forward_shape_higgs_all():
    """HIGGS all (28-D feature vec, 7 engineered + 21 raw) should still work."""
    spec = get_groups("higgs", "all")
    model = PIFT(spec, d_out=1, use_nsi=True, nsi_K=16, feature_dim=28)
    model.nsi.set_scaler_stats(torch.zeros(28), torch.ones(28))
    x = torch.randn(4, 28)
    y = model(x)
    assert y.shape == (4, 1)


def test_pift_v3_backward_grads_all_params():
    """Every named parameter (including KAN edges) must receive gradient."""
    spec = get_groups("higgs", "low_level")
    model = PIFT(spec, d_out=1, use_nsi=True, nsi_K=8, feature_dim=21)
    model.nsi.set_scaler_stats(torch.zeros(21), torch.ones(21))
    x = torch.randn(8, 21)
    target = torch.zeros(8)
    logits = model(x).squeeze(-1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss.backward()
    no_grad = []
    for n, p in model.named_parameters():
        if p.grad is None or torch.all(p.grad == 0):
            no_grad.append(n)
    assert not no_grad, f"params with no grad: {no_grad}"


def test_pift_v2_backward_compat():
    """Without --pift-nsi, v2 behavior must be unchanged (no nsi attr exposed)."""
    spec = get_groups("higgs", "low_level")
    model = PIFT(spec, d_out=1, use_nsi=False)
    assert model.use_nsi is False
    assert not hasattr(model, "nsi")
    x = torch.randn(4, 21)
    y = model(x)
    assert y.shape == (4, 1)


def test_pift_v3_susy_low():
    spec = get_groups("susy", "low_level")
    model = PIFT(spec, d_out=1, use_nsi=True, nsi_K=8, feature_dim=8)
    model.nsi.set_scaler_stats(torch.zeros(8), torch.ones(8))
    x = torch.randn(4, 8)
    y = model(x)
    assert y.shape == (4, 1)
    # SUSY low: 2 full × 4 + 1 transverse × 3 = 11-D KAN input
    assert model.nsi.kan.layers[0].in_features == 11
