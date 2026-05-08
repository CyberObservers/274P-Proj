"""Chebyshev-polynomial KAN layers (PIFT v5).

Used in two places:
  - As a v3 NSI KAN backbone alternative (`kan_impl="cheby"` in nsi.py)
  - As a v4 PIFT-Edge tokenizer alternative (`mlp_impl="cheby"` in edges.py)

Why Chebyshev: T_n(cos θ) = cos(n θ) is a natural orthogonal basis for the
sinh/cosh transforms that appear in Lorentz boosts, so degree-n Cheby
features can express N-body Lorentz invariants more directly than ReLU MLP
or B-spline.  Cheby1KANLayer formulation follows SS, 2024; rank-collapse
mitigation (residual + GELU) follows AC-PKAN (arXiv:2505.08687).
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class Cheby1KANLayer(nn.Module):
    """Chebyshev-T KAN layer.

    forward(x: (..., in_dim)) -> (..., out_dim) by:
      1. LayerNorm + tanh squeeze to keep input in [-1, 1] (T_n unstable outside)
      2. Compute T_0..T_degree via recurrence T_n = 2x T_{n-1} - T_{n-2}
      3. Contract T tensor with learnable coefficients (in_dim, out_dim, degree+1)
    """

    def __init__(self, in_dim: int, out_dim: int, degree: int = 4,
                 use_layernorm: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.degree = degree
        # Init scale ~ 1/sqrt((degree+1) * in_dim) keeps output variance ~1
        std = (1.0 / max(1, in_dim * (degree + 1))) ** 0.5
        self.cheby_coeffs = nn.Parameter(
            torch.randn(in_dim, out_dim, degree + 1) * std
        )
        self.norm = nn.LayerNorm(in_dim) if use_layernorm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = torch.tanh(x)  # squeeze to [-1, 1]
        # Recurrence T_0 = 1, T_1 = x, T_n = 2x T_{n-1} - T_{n-2}
        T_list = [torch.ones_like(x), x]
        for _ in range(2, self.degree + 1):
            T_list.append(2.0 * x * T_list[-1] - T_list[-2])
        T = torch.stack(T_list, dim=-1)  # (..., in_dim, degree+1)
        # einsum: contract in_dim and degree, output last dim = out_dim
        return torch.einsum("...id,iod->...o", T, self.cheby_coeffs)


class ChebyEdgeBlock(nn.Module):
    """Cheby1KANLayer + linear skip + GELU; mitigates rank collapse (AC-PKAN)."""

    def __init__(self, d_in: int, d_out: int, degree: int = 4):
        super().__init__()
        self.cheby = Cheby1KANLayer(d_in, d_out, degree=degree)
        self.skip = nn.Linear(d_in, d_out)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.cheby(x) + self.skip(x))


class ChebyKANBackbone(nn.Module):
    """Stacked Cheby1KANLayer for use as a KAN backbone (drop-in for KAN /
    FastKAN).  Used by NSI when kan_impl='cheby'."""

    def __init__(self, layers_hidden: List[int], degree: int = 4):
        super().__init__()
        layers = []
        for d_in, d_out in zip(layers_hidden[:-1], layers_hidden[1:]):
            layers.append(ChebyEdgeBlock(d_in, d_out, degree=degree))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
