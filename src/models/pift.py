"""PIFT: Physics-Informed Feature Tokenizer.

Three sub-modules in front of an FT-Transformer encoder:
  (a) PhysicsGroupEmbedding   — one token per physics object
  (b) InvariantAugmentation   — explicit Lorentz invariants (low-level only)
  (c) SymmetryAwarePE         — shared type embedding for same-type particles

Status: skeleton with end-to-end forward path. Sub-modules work but have
TODOs marked where the team should extend (e.g. more invariants, learnable
group MLP depth, cross-attention between groups).
"""
from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn

from src.data.feature_groups import DatasetGroupSpec, FeatureGroupConfig


# ---------------------------------------------------------------------------
# (a) Physics group embedding
# ---------------------------------------------------------------------------
class PhysicsGroupEmbedding(nn.Module):
    """Per-group MLP: features of one physics object -> one d_token vector."""

    def __init__(self, groups: List[FeatureGroupConfig], d_token: int,
                 hidden_mult: float = 1.0):
        super().__init__()
        self.groups = groups
        self.embeds = nn.ModuleList()
        for g in groups:
            d_in = len(g.indices)
            d_hidden = max(d_token, int(d_token * hidden_mult))
            self.embeds.append(nn.Sequential(
                nn.Linear(d_in, d_hidden),
                nn.GELU(),
                nn.Linear(d_hidden, d_token),
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F)  ->  (B, n_groups, d_token)
        tokens = []
        for g, emb in zip(self.groups, self.embeds):
            tokens.append(emb(x[:, g.indices]))
        return torch.stack(tokens, dim=1)


# ---------------------------------------------------------------------------
# (b) Invariant augmentation
# ---------------------------------------------------------------------------
def _four_momentum_from_pt_eta_phi(pt: torch.Tensor, eta: torch.Tensor,
                                   phi: torch.Tensor, mass: torch.Tensor | None = None):
    """Return (E, px, py, pz). Massless if `mass` is None (typical for jets/leptons)."""
    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    if mass is None:
        e = torch.sqrt(px * px + py * py + pz * pz + 1e-12)
    else:
        e = torch.sqrt(px * px + py * py + pz * pz + mass * mass + 1e-12)
    return e, px, py, pz


def _invariant_mass(p1, p2):
    e1, px1, py1, pz1 = p1
    e2, px2, py2, pz2 = p2
    e, px, py, pz = e1 + e2, px1 + px2, py1 + py2, pz1 + pz2
    m2 = e * e - (px * px + py * py + pz * pz)
    return torch.sqrt(torch.clamp(m2, min=1e-12))


def _delta_r(eta1, phi1, eta2, phi2):
    dphi = (phi1 - phi2 + math.pi) % (2 * math.pi) - math.pi
    deta = eta1 - eta2
    return torch.sqrt(deta * deta + dphi * dphi + 1e-12)


class InvariantAugmentation(nn.Module):
    """Compute m_inv and ΔR for declared object pairs and append as new tokens.

    NOTE: assumes groups follow `pT_eta_phi[_btag]` layout.
    """

    def __init__(self, spec: DatasetGroupSpec, d_token: int):
        super().__init__()
        self.spec = spec
        self.pairs = spec.invariant_pairs
        self._group_lookup = {g.name: g for g in spec.groups}
        # 2 invariants per pair (m_inv, ΔR). One token per pair.
        self.embed = nn.Sequential(
            nn.Linear(2, d_token),
            nn.GELU(),
            nn.Linear(d_token, d_token),
        )

    def _pt_eta_phi(self, x: torch.Tensor, name: str):
        idx = self._group_lookup[name].indices
        # convention: first three indices are pT, eta, phi
        return x[:, idx[0]], x[:, idx[1]], x[:, idx[2]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.pairs:
            return x.new_zeros(x.size(0), 0, self.embed[-1].out_features)
        feats = []
        for name_a, name_b in self.pairs:
            pt1, eta1, phi1 = self._pt_eta_phi(x, name_a)
            pt2, eta2, phi2 = self._pt_eta_phi(x, name_b)
            p1 = _four_momentum_from_pt_eta_phi(pt1, eta1, phi1)
            p2 = _four_momentum_from_pt_eta_phi(pt2, eta2, phi2)
            m  = _invariant_mass(p1, p2)
            dr = _delta_r(eta1, phi1, eta2, phi2)
            feats.append(torch.stack([m, dr], dim=-1))
        # (B, n_pairs, 2) -> (B, n_pairs, d_token)
        x_inv = torch.stack(feats, dim=1)
        return self.embed(x_inv)


# ---------------------------------------------------------------------------
# (c) Symmetry-aware PE: shared type embedding for same-type particles
# ---------------------------------------------------------------------------
class SymmetryAwarePE(nn.Module):
    def __init__(self, groups: List[FeatureGroupConfig], d_token: int,
                 num_types: int | None = None):
        super().__init__()
        type_ids = [g.type_id for g in groups]
        n_types = num_types or (max(type_ids) + 1)
        self.type_emb = nn.Embedding(n_types, d_token)
        self.register_buffer("type_ids", torch.tensor(type_ids, dtype=torch.long))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, n_groups, d_token)
        return tokens + self.type_emb(self.type_ids)[None, :, :]


# ---------------------------------------------------------------------------
# Top-level PIFT model
# ---------------------------------------------------------------------------
class PIFT(nn.Module):
    """PIFT = (group emb + invariant aug + sym PE) -> Transformer encoder -> CLS head.

    Drop-in replacement for FTTransformer when a `DatasetGroupSpec` is available
    (i.e. HEP datasets in low-level setup).
    """

    def __init__(
        self,
        spec: DatasetGroupSpec,
        d_out: int,
        *,
        d_token: int = 128,
        n_blocks: int = 3,
        n_heads: int = 8,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        use_invariants: bool = True,
        use_sym_pe: bool = True,
    ):
        super().__init__()
        self.group_emb = PhysicsGroupEmbedding(spec.groups, d_token=d_token)
        self.use_invariants = use_invariants and bool(spec.invariant_pairs)
        if self.use_invariants:
            self.inv_aug = InvariantAugmentation(spec, d_token=d_token)
        self.use_sym_pe = use_sym_pe
        if use_sym_pe:
            self.sym_pe = SymmetryAwarePE(spec.groups, d_token=d_token)

        # CLS token + transformer encoder
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.trunc_normal_(self.cls, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=int(d_token * 4 / 3),
            dropout=ffn_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.group_emb(x)
        if self.use_sym_pe:
            tokens = self.sym_pe(tokens)
        if self.use_invariants:
            inv_tokens = self.inv_aug(x)
            tokens = torch.cat([tokens, inv_tokens], dim=1)
        cls = self.cls.expand(tokens.size(0), -1, -1)
        h = torch.cat([cls, tokens], dim=1)
        h = self.encoder(h)
        h = self.norm(h[:, 0])
        return self.head(h)
