"""Verifies the boost / 4-momentum utilities respect the physical invariants
they claim (m² preserved, eta-shift equivalent to full SO(1,1) boost for
massless particles)."""
from __future__ import annotations

import torch

from src.models.lorentz import (
    apply_longitudinal_boost_in_kin,
    feature_vec_to_p4_concat,
    kan_input_dim,
    kinematics_to_p4,
    pair_invariant_mass_sq,
    transverse_to_p3,
)
from src.data.feature_groups import get_groups


def test_kinematics_to_p4_massless_E_eq_p():
    """For m=0, E should equal |p|."""
    pT  = torch.tensor([10.0, 50.0, 100.0])
    eta = torch.tensor([-1.0, 0.0, 1.5])
    phi = torch.tensor([0.3, 1.0, -2.0])
    p4 = kinematics_to_p4(pT, eta, phi, m=0.0)
    E  = p4[..., 0]
    p_norm = torch.sqrt(p4[..., 1] ** 2 + p4[..., 2] ** 2 + p4[..., 3] ** 2)
    assert torch.allclose(E, p_norm, atol=1e-5)


def test_longitudinal_boost_preserves_invariant_mass_squared():
    """For two massless 4-vectors, (p1+p2)^2 is Lorentz invariant."""
    pT1, eta1, phi1 = torch.tensor([30.0]), torch.tensor([0.5]), torch.tensor([1.0])
    pT2, eta2, phi2 = torch.tensor([40.0]), torch.tensor([-0.7]), torch.tensor([-1.0])
    p1 = kinematics_to_p4(pT1, eta1, phi1)
    p2 = kinematics_to_p4(pT2, eta2, phi2)
    m2_before = pair_invariant_mass_sq(p1, p2)

    # Apply longitudinal boost to both via eta-shift form
    for beta_val in [-0.4, -0.1, 0.1, 0.3, 0.5]:
        beta = torch.tensor(beta_val)
        shift = torch.atanh(beta)
        p1_b = kinematics_to_p4(pT1, eta1 - shift, phi1)
        p2_b = kinematics_to_p4(pT2, eta2 - shift, phi2)
        m2_after = pair_invariant_mass_sq(p1_b, p2_b)
        assert torch.allclose(m2_before, m2_after, rtol=1e-4, atol=1e-3), \
            f"beta={beta_val}: m² {m2_before.item()} -> {m2_after.item()}"


def test_apply_longitudinal_boost_only_shifts_eta_in_full_groups():
    """boost should leave pT, phi, transverse groups, engineered groups untouched."""
    spec = get_groups("higgs", "low_level")
    torch.manual_seed(0)
    x = torch.randn(4, 21)
    beta = torch.tensor(0.2)
    x_b = apply_longitudinal_boost_in_kin(x, spec, beta)

    # full groups: only eta (index 1 within group) changes
    for g in spec.groups:
        if g.kinematic_kind == "full":
            # pT (g.indices[0]) unchanged
            assert torch.allclose(x[:, g.indices[0]], x_b[:, g.indices[0]])
            # phi (g.indices[2]) unchanged
            assert torch.allclose(x[:, g.indices[2]], x_b[:, g.indices[2]])
            # eta (g.indices[1]) shifted by exactly artanh(beta)
            eta_shift = torch.atanh(beta)
            assert torch.allclose(x[:, g.indices[1]] - eta_shift, x_b[:, g.indices[1]])
        elif g.kinematic_kind == "transverse":
            # all features unchanged
            for idx in g.indices:
                assert torch.allclose(x[:, idx], x_b[:, idx])


def test_feature_vec_to_p4_concat_skips_engineered():
    """HIGGS low: 5 full × 4 + 1 transverse × 3 = 23.  HIGGS all: same (eng skipped)."""
    spec_low  = get_groups("higgs", "low_level")
    spec_all  = get_groups("higgs", "all")
    x_low = torch.randn(2, 21)
    x_all = torch.randn(2, 28)
    p4_low = feature_vec_to_p4_concat(x_low, spec_low)
    p4_all = feature_vec_to_p4_concat(x_all, spec_all)
    assert p4_low.shape == (2, 23) == (2, kan_input_dim(spec_low))
    assert p4_all.shape == (2, 23) == (2, kan_input_dim(spec_all))


def test_transverse_p3_E_eq_mag():
    """MET-style 3-vec: E must equal mag, output is 3-D (pz=0 dropped)."""
    mag = torch.tensor([20.0, 50.0])
    phi = torch.tensor([0.5, -1.2])
    p3 = transverse_to_p3(mag, phi)
    assert p3.shape == (2, 3)
    assert torch.allclose(p3[..., 0], mag)        # E = mag


def test_eta_shift_equivalent_to_full_4momentum_boost_massless():
    """Independent verification: explicit Lorentz matrix on (E, px, py, pz)
    should match eta-shift for m=0."""
    pT  = torch.tensor([25.0])
    eta = torch.tensor([0.8])
    phi = torch.tensor([0.4])
    beta = torch.tensor(0.3)
    gamma = 1.0 / torch.sqrt(1 - beta ** 2)

    # eta-shift form
    p4_a = kinematics_to_p4(pT, eta - torch.atanh(beta), phi)

    # full Lorentz boost form (along z)
    p4_orig = kinematics_to_p4(pT, eta, phi)
    E, px, py, pz = p4_orig[..., 0], p4_orig[..., 1], p4_orig[..., 2], p4_orig[..., 3]
    E_b  = gamma * (E - beta * pz)
    pz_b = gamma * (pz - beta * E)
    p4_b = torch.stack([E_b, px, py, pz_b], dim=-1)

    assert torch.allclose(p4_a, p4_b, rtol=1e-5, atol=1e-5), \
        f"eta-shift: {p4_a}\nLorentz:    {p4_b}"
