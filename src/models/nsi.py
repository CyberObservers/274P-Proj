"""NeuralSymbolicInvariantExtractor (NSI) — PIFT v3 core module.

Maps a raw feature vector (after StandardScaler) of HIGGS / SUSY low-level
data to K Lorentz-boost-invariant scalars via a KAN over per-particle
4-momenta.  Boost-invariance is enforced by a soft regularizer in train.py
(this module only provides forward and a `boost` helper).
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from src.data.feature_groups import DatasetGroupSpec
from src.models import lorentz
from src.models.lorentz import kan_input_dim


class NeuralSymbolicInvariantExtractor(nn.Module):
    """raw feature vec (B, F)  ->  K invariants (B, K).

    StandardScaler stats must be installed via `set_scaler_stats()` after the
    scaler is fit on the train set, *before* training begins — the module
    needs to undo standardization to recover physical (pT, eta, phi).

    `kan_impl` selects the KAN backbone:
      - "efficient" (default): efficient-kan with B-spline grid, supports
        update_grid(); v3 main behaviour.
      - "fast": fastkan FastKAN with RBF basis; no grid update needed,
        ~3× faster training; v5 default for new runs.
      - "cheby": Chebyshev polynomial KAN (in-house); experimental, see
        src/models/cheby_kan.py.
    """

    def __init__(
        self,
        spec: DatasetGroupSpec,
        K: int = 16,
        kan_hidden: Tuple[int, ...] = (32,),
        kan_grid: int = 5,
        kan_spline_order: int = 3,
        feature_dim: int | None = None,
        kan_impl: str = "efficient",
    ):
        super().__init__()
        self.spec = spec
        self.K = K
        self.kan_impl = kan_impl
        # full: 4 components (E, px, py, pz)   transverse: 3 components (E, px, py)
        self.n_kin_groups = sum(
            1 for g in spec.groups if g.kinematic_kind in ("full", "transverse")
        )
        d_p4 = kan_input_dim(spec)

        layers_hidden = [d_p4] + list(kan_hidden) + [K]
        if kan_impl == "fast":
            from fastkan import FastKAN
            # num_grids ~ kan_grid * 2 because RBF doesn't have spline_order
            self.kan = FastKAN(layers_hidden=layers_hidden,
                               num_grids=max(8, kan_grid * 2))
        elif kan_impl == "cheby":
            from src.models.cheby_kan import ChebyKANBackbone
            self.kan = ChebyKANBackbone(layers_hidden, degree=kan_spline_order + 1)
        elif kan_impl == "efficient":
            from efficient_kan import KAN
            self.kan = KAN(
                layers_hidden=layers_hidden,
                grid_size=kan_grid,
                spline_order=kan_spline_order,
            )
        else:
            raise ValueError(f"unknown kan_impl={kan_impl!r}, expected efficient/fast/cheby")

        # scaler stats — populated by set_scaler_stats() before training
        F = feature_dim if feature_dim is not None else max(
            i for g in spec.groups for i in g.indices
        ) + 1
        self.register_buffer("scaler_mean", torch.zeros(F))
        self.register_buffer("scaler_std", torch.ones(F))
        self._scaler_set = False

    def set_scaler_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        assert mean.shape == self.scaler_mean.shape, (mean.shape, self.scaler_mean.shape)
        self.scaler_mean.copy_(mean.to(self.scaler_mean.device).float())
        self.scaler_std.copy_(std.to(self.scaler_std.device).float().clamp_min(1e-6))
        self._scaler_set = True

    def _unscale(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scaler_std + self.scaler_mean

    def _rescale(self, x_raw: torch.Tensor) -> torch.Tensor:
        return (x_raw - self.scaler_mean) / self.scaler_std

    def particles_to_p4(self, x_scaled: torch.Tensor) -> torch.Tensor:
        x_raw = self._unscale(x_scaled)
        return lorentz.feature_vec_to_p4_concat(x_raw, self.spec)

    def forward(self, x_scaled: torch.Tensor) -> torch.Tensor:
        # KAN runs in fp32 even under autocast — splines are fp16-unstable
        with torch.amp.autocast("cuda", enabled=False):
            p4 = self.particles_to_p4(x_scaled.float())
            z = self.kan(p4)
        return z

    def boost(self, x_scaled: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Returns x_scaled' such that x_scaled'_raw = boost(x_scaled_raw).
        Same schema as input.  beta is scalar tensor (or shape (B,))."""
        with torch.amp.autocast("cuda", enabled=False):
            x_raw = self._unscale(x_scaled.float())
            x_raw_b = lorentz.apply_longitudinal_boost_in_kin(x_raw, self.spec, beta)
            return self._rescale(x_raw_b)

    def update_kan_grid(self, x_scaled: torch.Tensor) -> None:
        """Refresh efficient-kan spline grids from a sample batch.  Call early
        in training; per-layer call required.  No-op for fast/cheby (RBF/Cheby
        bases have fixed centers / coefficients)."""
        if self.kan_impl != "efficient":
            return
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            p4 = self.particles_to_p4(x_scaled.float())
            h = p4
            for layer in self.kan.layers:
                layer.update_grid(h)
                h = layer(h)
