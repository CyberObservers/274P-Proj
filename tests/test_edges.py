"""Tests for pairwise Lorentz scalar computation and edge tokenization."""
from __future__ import annotations

import torch

from src.data.feature_groups import get_groups
from src.models.edges import EdgeTokenizer, compute_pairwise_invariants
from src.models.pift import PIFT


def test_pairwise_invariants_shape_and_diagonal():
    B, N = 2, 5
    p4 = torch.randn(B, N, 4) * 10 + torch.tensor([100., 0., 0., 0.])  # E ≈ 100
    p4[..., 0] = p4[..., 0].abs() + 50.0  # ensure E > |p|
    edges = compute_pairwise_invariants(p4)
    assert edges.shape == (B, N, N, 4)
    # diagonal = 0 (i==j)
    for i in range(N):
        assert torch.allclose(edges[:, i, i, :], torch.zeros(B, 4))


def test_pairwise_invariants_lorentz_invariance_of_m_sq():
    """m² should be invariant under longitudinal boost of all p4."""
    torch.manual_seed(0)
    # Build valid 4-momenta from (pT, eta, phi) with m=0
    pT  = torch.tensor([[10., 20., 30., 40.]])
    eta = torch.tensor([[0.5, -0.3, 1.1, -0.8]])
    phi = torch.tensor([[0.2, 1.5, -1.0, 2.0]])
    E  = pT * torch.cosh(eta)
    px = pT * torch.cos(phi)
    py = pT * torch.sin(phi)
    pz = pT * torch.sinh(eta)
    p4 = torch.stack([E, px, py, pz], dim=-1)            # (1, 4, 4)
    m_sq_before = compute_pairwise_invariants(p4)[..., 0]
    # apply longitudinal boost β=0.3 to all p4
    beta = 0.3
    gamma = 1.0 / (1 - beta**2) ** 0.5
    E_b  = gamma * (p4[..., 0] - beta * p4[..., 3])
    pz_b = gamma * (p4[..., 3] - beta * p4[..., 0])
    p4_b = torch.stack([E_b, p4[..., 1], p4[..., 2], pz_b], dim=-1)
    m_sq_after = compute_pairwise_invariants(p4_b)[..., 0]
    diff = (m_sq_before - m_sq_after).abs()
    diff_off_diag = diff[diff > 0]
    assert diff_off_diag.max() < 1e-3, f"max diff = {diff_off_diag.max()}"


def test_edge_tokenizer_top_k_selects_correctly():
    torch.manual_seed(0)
    et = EdgeTokenizer(d_token=32, edge_k=4, edge_select="kt")
    p4 = torch.randn(2, 5, 4) * 10
    p4[..., 0] = p4[..., 0].abs() + 50.0
    out = et(p4)
    # 5 nodes -> 10 unique pairs; we keep 4
    assert out.shape == (2, 4, 32)


def test_edge_tokenizer_all_pairs_when_k_too_large():
    et = EdgeTokenizer(d_token=32, edge_k=999, edge_select="kt")
    p4 = torch.randn(1, 4, 4)
    p4[..., 0] = p4[..., 0].abs() + 50.0
    out = et(p4)
    # 4 nodes -> 6 pairs total; we asked for 999 → keep all 6
    assert out.shape == (1, 6, 32)


def test_pift_v4_edge_smoke_top_tagging():
    """Build PIFT-Edge for top_tagging spec; forward + backward without NaN."""
    spec = get_groups("top_tagging", "all")
    model = PIFT(spec, d_out=1, use_edge_tokens=True, edge_k=8, feature_dim=64)
    model.set_edge_scaler_stats(torch.zeros(64), torch.ones(64))
    # Construct valid 4-momenta from (pT, eta, phi) so E² > |p|²
    torch.manual_seed(0)
    B = 4
    x = torch.zeros(B, 64)
    for k in range(8):
        pT  = torch.rand(B) * 50 + 10
        eta = torch.randn(B) * 0.5
        phi = torch.rand(B) * 2 * torch.pi - torch.pi
        E  = pT * torch.cosh(eta)
        x[:, 8*k+0] = E
        x[:, 8*k+1] = pT * torch.cos(phi)
        x[:, 8*k+2] = pT * torch.sin(phi)
        x[:, 8*k+3] = pT * torch.sinh(eta)
        x[:, 8*k+4] = torch.log(torch.tensor(5.0))           # log_n
        x[:, 8*k+5] = torch.zeros(B)                          # mass
        x[:, 8*k+6] = torch.zeros(B)                          # width
        x[:, 8*k+7] = torch.log(pT + 1)                       # log_pt
    y = model(x)
    assert y.shape == (B, 1)
    assert torch.isfinite(y).all(), f"y={y}"
    loss = y.sum()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.edge_tokenizer.parameters())


def test_pift_subjet_no_edge_smoke():
    """Plain PIFT-Subjet (v2 group emb on subjet input)."""
    spec = get_groups("top_tagging", "all")
    model = PIFT(spec, d_out=1, use_edge_tokens=False)
    x = torch.randn(4, 64)
    y = model(x)
    assert y.shape == (4, 1)
    assert not hasattr(model, "edge_tokenizer")
