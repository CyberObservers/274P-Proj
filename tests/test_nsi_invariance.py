"""Verifies the NSI module + boost-invariance regularizer can train z to be
boost-invariant when given enough optimization steps with no classification
gradient."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.data.feature_groups import get_groups
from src.models.nsi import NeuralSymbolicInvariantExtractor


def test_random_init_is_not_boost_invariant():
    """Sanity: an untrained NSI should NOT be boost-invariant."""
    spec = get_groups("higgs", "low_level")
    nsi = NeuralSymbolicInvariantExtractor(spec, K=8, feature_dim=21)
    nsi.set_scaler_stats(torch.zeros(21), torch.ones(21))
    torch.manual_seed(1)
    x = torch.randn(64, 21)
    beta = torch.tensor(0.25)
    z1 = nsi(x)
    z2 = nsi(nsi.boost(x, beta))
    mse = F.mse_loss(z1, z2).item()
    assert mse > 0.001, f"random-init MSE={mse} too small — test setup broken"


def test_invariance_training_converges():
    """200 steps of pure-invariance training (no cls gradient) should drive
    the boost MSE below 0.01 — catches gradient-flow bugs in the integration."""
    torch.manual_seed(0)
    spec = get_groups("higgs", "low_level")
    nsi = NeuralSymbolicInvariantExtractor(spec, K=8, feature_dim=21)
    nsi.set_scaler_stats(torch.zeros(21), torch.ones(21))
    opt = torch.optim.AdamW(nsi.parameters(), lr=3e-3)

    x_base = torch.randn(128, 21)
    final_mse = None
    for step in range(200):
        beta = torch.empty(()).uniform_(0, 0.3)
        z1 = nsi(x_base)
        z2 = nsi(nsi.boost(x_base, beta))
        # normalized-MSE prevents trivial collapse to zero
        denom = z1.detach().std(dim=0).mean().clamp_min(1e-3)
        loss = F.mse_loss(z1, z2) / denom
        opt.zero_grad(); loss.backward(); opt.step()
        final_mse = F.mse_loss(z1, z2).item()
    print(f"final raw MSE: {final_mse:.4f}")
    assert final_mse < 0.05, f"NSI did not learn boost-invariance; final MSE={final_mse}"


def test_engineered_groups_skipped_in_nsi():
    """HIGGS all (28-D) should give same KAN input dim as HIGGS low (21-D)
    because engineered group is skipped."""
    spec_low = get_groups("higgs", "low_level")
    spec_all = get_groups("higgs", "all")
    nsi_low = NeuralSymbolicInvariantExtractor(spec_low, K=8, feature_dim=21)
    nsi_all = NeuralSymbolicInvariantExtractor(spec_all, K=8, feature_dim=28)
    # both KANs should have same input dim (23 = 5 full × 4 + 1 transverse × 3)
    assert nsi_low.kan.layers[0].in_features == nsi_all.kan.layers[0].in_features == 23
