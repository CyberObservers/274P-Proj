"""PIFT v5 smoke tests: ChebyKAN, FastKAN-NSI, TabM."""
from __future__ import annotations

import torch
import pytest

from src.data.feature_groups import get_groups
from src.models.cheby_kan import Cheby1KANLayer, ChebyEdgeBlock
from src.models.edges import EdgeTokenizer
from src.models.pift import PIFT


# ---------------------------------------------------------------------------
# ChebyKAN
# ---------------------------------------------------------------------------
def test_cheby1kan_shape():
    layer = Cheby1KANLayer(in_dim=4, out_dim=16, degree=4)
    x = torch.randn(8, 4)
    y = layer(x)
    assert y.shape == (8, 16)
    assert torch.isfinite(y).all()


def test_cheby_edge_block_with_residual():
    blk = ChebyEdgeBlock(d_in=4, d_out=64, degree=4)
    x = torch.randn(2, 6, 4)
    y = blk(x)
    assert y.shape == (2, 6, 64)
    assert torch.isfinite(y).all()


def _valid_p4(B: int, N: int) -> torch.Tensor:
    """Build valid massless 4-momenta from (pT, eta, phi) so E > |p|."""
    pT  = torch.rand(B, N) * 50 + 10
    eta = torch.randn(B, N) * 0.5
    phi = torch.rand(B, N) * 2 * torch.pi - torch.pi
    E  = pT * torch.cosh(eta)
    px = pT * torch.cos(phi)
    py = pT * torch.sin(phi)
    pz = pT * torch.sinh(eta)
    return torch.stack([E, px, py, pz], dim=-1)


def test_cheby_edge_tokenizer_shape_match():
    """ChebyKAN edge tokenizer matches MLP variant in shape."""
    et_mlp = EdgeTokenizer(d_token=64, edge_k=16, edge_select="kt", mlp_impl="mlp")
    et_cheby = EdgeTokenizer(d_token=64, edge_k=16, edge_select="kt", mlp_impl="cheby")
    p4 = _valid_p4(4, 8)
    out_mlp = et_mlp(p4)
    out_cheby = et_cheby(p4)
    assert out_mlp.shape == out_cheby.shape == (4, 16, 64)
    assert torch.isfinite(out_cheby).all()


# ---------------------------------------------------------------------------
# Fast-KAN NSI
# ---------------------------------------------------------------------------
def test_fastkan_nsi_smoke():
    spec = get_groups("higgs", "low_level")
    F_dim = max(i for g in spec.groups for i in g.indices) + 1
    m = PIFT(spec, d_out=1, feature_dim=F_dim, use_nsi=True, nsi_K=8,
             nsi_kan_impl="fast")
    m.nsi.set_scaler_stats(torch.zeros(F_dim), torch.ones(F_dim))
    x = torch.randn(4, F_dim)
    out = m(x)
    assert out.shape == (4, 1)
    assert torch.isfinite(out).all()


def test_fastkan_nsi_no_grid_update():
    """fast-kan implementation should not call update_kan_grid."""
    spec = get_groups("higgs", "low_level")
    F_dim = max(i for g in spec.groups for i in g.indices) + 1
    m = PIFT(spec, d_out=1, feature_dim=F_dim, use_nsi=True, nsi_K=4,
             nsi_kan_impl="fast")
    m.nsi.set_scaler_stats(torch.zeros(F_dim), torch.ones(F_dim))
    x = torch.randn(4, F_dim)
    # Should be a no-op for fast-kan, not error
    m.nsi.update_kan_grid(x)


# ---------------------------------------------------------------------------
# TabM
# ---------------------------------------------------------------------------
def test_pift_tabm_output_shape():
    spec = get_groups("higgs", "low_level")
    F_dim = max(i for g in spec.groups for i in g.indices) + 1
    m = PIFT(spec, d_out=1, feature_dim=F_dim, use_tabm=True, tabm_k=8)
    x = torch.randn(4, F_dim)
    out = m(x)
    # Output is (B, k) when d_out=1 (squeeze last dim)
    assert out.shape == (4, 8)
    assert torch.isfinite(out).all()


def test_pift_tabm_ensemble_diversity():
    """TabM members should produce diverse predictions (pairwise corr < 0.99)."""
    spec = get_groups("higgs", "low_level")
    F_dim = max(i for g in spec.groups for i in g.indices) + 1
    m = PIFT(spec, d_out=1, feature_dim=F_dim, use_tabm=True, tabm_k=4)
    m.eval()
    x = torch.randn(64, F_dim)
    with torch.no_grad():
        out = m(x)  # (64, 4)
    # Check pairwise correlation between submodels
    out_centered = out - out.mean(dim=0, keepdim=True)
    norm = out_centered.norm(dim=0).clamp_min(1e-6)
    out_norm = out_centered / norm
    corr = out_norm.T @ out_norm  # (4, 4)
    off_diag = corr - torch.eye(4)
    max_corr = off_diag.abs().max().item()
    assert max_corr < 0.99, f"ensemble members too correlated: max |corr| = {max_corr}"


def test_pift_v5_combo_smoke():
    """PIFT-Edge + ChebyKAN-Edge + TabM combo: 3 v5 features stacked."""
    spec = get_groups("top_tagging", "all")
    F_dim = max(i for g in spec.groups for i in g.indices) + 1
    m = PIFT(spec, d_out=1, feature_dim=F_dim, use_edge_tokens=True,
             edge_k=16, edge_kan="cheby", use_tabm=True, tabm_k=4)
    m.set_edge_scaler_stats(torch.zeros(F_dim), torch.ones(F_dim))
    # Build x with valid 4-momenta in subjet slots (E,px,py,pz at indices 8k..8k+3)
    torch.manual_seed(0)
    B = 4
    x = torch.zeros(B, F_dim)
    for k in range(8):
        pT  = torch.rand(B) * 50 + 10
        eta = torch.randn(B) * 0.5
        phi = torch.rand(B) * 2 * torch.pi - torch.pi
        x[:, 8*k+0] = pT * torch.cosh(eta)
        x[:, 8*k+1] = pT * torch.cos(phi)
        x[:, 8*k+2] = pT * torch.sin(phi)
        x[:, 8*k+3] = pT * torch.sinh(eta)
        x[:, 8*k+4] = torch.log(torch.tensor(5.0))
        x[:, 8*k+5] = torch.zeros(B)
        x[:, 8*k+6] = torch.zeros(B)
        x[:, 8*k+7] = torch.log(pT + 1)
    out = m(x)
    assert out.shape == (4, 4)
    assert torch.isfinite(out).all()
