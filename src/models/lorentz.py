"""Lorentz / 4-momentum utilities for PIFT v3 NSI.

Pure-tensor functions, no nn.Module.  All work in standardized broadcastable
shapes so they compose with autograd.

Conventions
-----------
- Massless particles (m=0) for all jets and leptons (Baldi 2014 convention).
- (pT, eta, phi) -> (E, px, py, pz) via:
      E  = pT cosh(eta)        px = pT cos(phi)
      pz = pT sinh(eta)        py = pT sin(phi)
- Longitudinal boost along z by beta: under m=0, equivalent to
      eta' = eta - artanh(beta)  ;   pT, phi unchanged
- MET is treated as a massless transverse 4-vector (pz=0, m=0); under
  longitudinal boost the (mag, phi) representation is invariant.
"""
from __future__ import annotations

from typing import List

import torch

from src.data.feature_groups import DatasetGroupSpec, FeatureGroupConfig


def kinematics_to_p4(pT: torch.Tensor, eta: torch.Tensor, phi: torch.Tensor,
                     m: float | torch.Tensor = 0.0) -> torch.Tensor:
    """Stack (E, px, py, pz) along the last axis.  Inputs broadcastable.

    Returns shape (..., 4).
    """
    px = pT * torch.cos(phi)
    py = pT * torch.sin(phi)
    pz = pT * torch.sinh(eta)
    if isinstance(m, torch.Tensor) or m != 0.0:
        E = torch.sqrt(pT * pT * torch.cosh(eta) ** 2 + (m * m if not isinstance(m, torch.Tensor) else m * m))
    else:
        E = pT * torch.cosh(eta)
    return torch.stack([E, px, py, pz], dim=-1)


def transverse_to_p3(mag: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """MET-style: (mag, phi) -> (E=mag, px, py), pz=0 dropped because constants
    break KAN spline fitting (rank-deficient lstsq).  For longitudinally
    boosted MET, this 3-vec is invariant (pz=0 stays 0)."""
    px = mag * torch.cos(phi)
    py = mag * torch.sin(phi)
    return torch.stack([mag, px, py], dim=-1)


def feature_vec_to_p4_concat(x_raw: torch.Tensor, spec: DatasetGroupSpec) -> torch.Tensor:
    """Slice a (B, F) feature vector into per-group 4-momenta and concat.

    Skips groups with kinematic_kind="engineered".
    Returns (B, 4*n_full_groups + 3*n_transverse_groups + 4*n_subjet_groups).
    """
    p4_blocks: List[torch.Tensor] = []
    for g in spec.groups:
        if g.kinematic_kind == "full":
            assert len(g.indices) >= 3, f"full group {g.name!r} needs >=3 features"
            pT  = x_raw[:, g.indices[0]]
            eta = x_raw[:, g.indices[1]]
            phi = x_raw[:, g.indices[2]]
            p4_blocks.append(kinematics_to_p4(pT, eta, phi))   # (B, 4)
        elif g.kinematic_kind == "transverse":
            assert len(g.indices) >= 2, f"transverse group {g.name!r} needs >=2 features"
            mag = x_raw[:, g.indices[0]]
            phi = x_raw[:, g.indices[1]]
            p4_blocks.append(transverse_to_p3(mag, phi))       # (B, 3) — drops pz=0
        elif g.kinematic_kind == "subjet":
            # (E, px, py, pz) directly at indices[0..3]; ignore substructure feats
            assert len(g.indices) >= 4, f"subjet group {g.name!r} needs >=4 features"
            p4_blocks.append(x_raw[:, g.indices[:4]])           # (B, 4)
        elif g.kinematic_kind == "engineered":
            continue
        else:
            raise ValueError(f"unknown kinematic_kind={g.kinematic_kind!r}")
    return torch.cat(p4_blocks, dim=-1)


def kan_input_dim(spec: DatasetGroupSpec) -> int:
    """Dimension of the concatenated 4-momentum vector that NSI feeds to KAN."""
    n_full = sum(1 for g in spec.groups if g.kinematic_kind == "full")
    n_trans = sum(1 for g in spec.groups if g.kinematic_kind == "transverse")
    n_subjet = sum(1 for g in spec.groups if g.kinematic_kind == "subjet")
    return 4 * n_full + 3 * n_trans + 4 * n_subjet


def apply_longitudinal_boost_in_kin(x_raw: torch.Tensor, spec: DatasetGroupSpec,
                                    beta: torch.Tensor) -> torch.Tensor:
    """Apply longitudinal boost in (pT, eta, phi) representation to a raw
    feature vector (B, F).  Returns same shape, same schema.

    For m=0 particles this is an exact symmetry; for nonzero m it is ~0.1%
    accurate at LHC scales (see PIFT_GUIDE §10).
    """
    out = x_raw.clone()
    beta_c = beta.clamp(-0.999, 0.999)
    eta_shift = torch.atanh(beta_c)               # scalar or (B,)
    gamma = 1.0 / torch.sqrt(1.0 - beta_c * beta_c)
    for g in spec.groups:
        if g.kinematic_kind == "full":
            eta_idx = g.indices[1]
            out[:, eta_idx] = out[:, eta_idx] - eta_shift
        elif g.kinematic_kind == "subjet":
            # direct (E, pz) SO(1,1) boost matrix; px, py untouched
            E_idx = g.indices[0]
            pz_idx = g.indices[3]
            E_old = out[:, E_idx].clone()
            pz_old = out[:, pz_idx].clone()
            out[:, E_idx]  = gamma * (E_old - beta_c * pz_old)
            out[:, pz_idx] = gamma * (pz_old - beta_c * E_old)
        # "transverse" and "engineered": invariant under longitudinal boost
    return out


def pair_invariant_mass_sq(p4_a: torch.Tensor, p4_b: torch.Tensor) -> torch.Tensor:
    """(p_a + p_b)^mu (p_a + p_b)_mu  with metric (+,-,-,-).  Inputs (..., 4)."""
    s = p4_a + p4_b
    return s[..., 0] ** 2 - s[..., 1] ** 2 - s[..., 2] ** 2 - s[..., 3] ** 2


def pair_deltaR(eta_a: torch.Tensor, phi_a: torch.Tensor,
                eta_b: torch.Tensor, phi_b: torch.Tensor) -> torch.Tensor:
    dphi = (phi_a - phi_b + torch.pi) % (2 * torch.pi) - torch.pi
    deta = eta_a - eta_b
    return torch.sqrt(deta * deta + dphi * dphi + 1e-12)
