"""PIFT v2: Physics-Informed Feature Tokenizer (simplified).

Architecture: PhysicsGroupEmbedding -> [CLS] + tokens -> Transformer encoder -> head.

v1 had three sub-modules (group emb + invariant augmentation + symmetry-aware PE).
Ablation on HIGGS low-level showed only the group embedding contributed measurable
AUC; the other two were dropped (see EXPERIMENT_REPORT.md §8.2).
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from src.data.feature_groups import DatasetGroupSpec, FeatureGroupConfig
from src.models.nsi import NeuralSymbolicInvariantExtractor
from src.models.edges import EdgeTokenizer


class PhysicsGroupEmbedding(nn.Module):
    """Group features per physics object into one d_token vector.

    `weight_tied=True` (default): one MLP per unique `type_id` shared across
    all groups of that type — restores permutation equivariance for same-type
    particles (e.g. jet_1..jet_4 all use the same MLP weights).

    `weight_tied=False`: one MLP per group (v1 behaviour, kept for ablation).
    """

    def __init__(self, groups: List[FeatureGroupConfig], d_token: int,
                 hidden_mult: float = 1.0, weight_tied: bool = True):
        super().__init__()
        self.groups = groups
        self.weight_tied = weight_tied
        d_hidden = max(d_token, int(d_token * hidden_mult))

        if weight_tied:
            # One MLP per unique type_id. Same type_id requires same d_in.
            type_d_in: dict[int, int] = {}
            for g in groups:
                d_in = len(g.indices)
                if g.type_id in type_d_in:
                    if type_d_in[g.type_id] != d_in:
                        raise ValueError(
                            f"weight_tied=True needs same d_in per type_id; "
                            f"type {g.type_id}: got d_in {d_in} for {g.name!r}, "
                            f"already saw d_in {type_d_in[g.type_id]}")
                else:
                    type_d_in[g.type_id] = d_in
            self.type_mlps = nn.ModuleDict({
                str(t): nn.Sequential(
                    nn.Linear(d_in, d_hidden),
                    nn.GELU(),
                    nn.Linear(d_hidden, d_token),
                ) for t, d_in in type_d_in.items()
            })
        else:
            self.embeds = nn.ModuleList()
            for g in groups:
                d_in = len(g.indices)
                self.embeds.append(nn.Sequential(
                    nn.Linear(d_in, d_hidden),
                    nn.GELU(),
                    nn.Linear(d_hidden, d_token),
                ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = []
        if self.weight_tied:
            for g in self.groups:
                tokens.append(self.type_mlps[str(g.type_id)](x[:, g.indices]))
        else:
            for g, emb in zip(self.groups, self.embeds):
                tokens.append(emb(x[:, g.indices]))
        return torch.stack(tokens, dim=1)


class SetPooledGroupEmbedding(nn.Module):
    """Set-transformer style pooling: same-type particles get pooled into a single
    token via cross-attention with a learnable query.

    Output:  (B, n_unique_types, d_token).
    Naturally permutation-invariant over same-type particles + true equivariance
    even without weight tying inside the per-particle embedder.
    """

    def __init__(self, groups: List[FeatureGroupConfig], d_token: int,
                 hidden_mult: float = 1.0, n_pool_heads: int = 4):
        super().__init__()
        self.groups = groups
        d_hidden = max(d_token, int(d_token * hidden_mult))

        # one MLP per type_id (weight-tied within type for the per-particle embedding)
        type_d_in: dict[int, int] = {}
        for g in groups:
            d_in = len(g.indices)
            if g.type_id in type_d_in and type_d_in[g.type_id] != d_in:
                raise ValueError("set-pooled requires same d_in per type_id")
            type_d_in[g.type_id] = d_in
        self.type_mlps = nn.ModuleDict({
            str(t): nn.Sequential(
                nn.Linear(d_in, d_hidden), nn.GELU(),
                nn.Linear(d_hidden, d_token),
            ) for t, d_in in type_d_in.items()
        })

        # one learnable pooling query per type_id (only useful when multiple
        # particles share a type; for unique types the pool is a no-op identity).
        self.type_to_groups: dict[int, list[int]] = {}
        for i, g in enumerate(groups):
            self.type_to_groups.setdefault(g.type_id, []).append(i)
        self.pool_queries = nn.ParameterDict({
            str(t): nn.Parameter(torch.randn(1, 1, d_token) * 0.02)
            for t, idxs in self.type_to_groups.items() if len(idxs) > 1
        })
        self.pool_attn = nn.ModuleDict({
            str(t): nn.MultiheadAttention(d_token, n_pool_heads, batch_first=True)
            for t, idxs in self.type_to_groups.items() if len(idxs) > 1
        })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Embed each group
        per_group_emb = []  # list of (B, d_token)
        for g in self.groups:
            per_group_emb.append(self.type_mlps[str(g.type_id)](x[:, g.indices]))

        # 2. For each type: if singleton, pass through; if multiple, attention pool
        out = []
        for t, group_idxs in self.type_to_groups.items():
            if len(group_idxs) == 1:
                out.append(per_group_emb[group_idxs[0]])
            else:
                stack = torch.stack([per_group_emb[i] for i in group_idxs], dim=1)  # (B, k, d)
                B = stack.size(0)
                q = self.pool_queries[str(t)].expand(B, -1, -1)  # (B, 1, d)
                pooled, _ = self.pool_attn[str(t)](q, stack, stack)
                out.append(pooled.squeeze(1))
        return torch.stack(out, dim=1)  # (B, n_unique_types, d_token)


class _PerFeatureLinearEmbed(nn.Module):
    """Used only by --pift-no-group ablation (degenerates to FT-T-style per-feature emb)."""

    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d_token) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, d_token))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) * self.weight + self.bias


class PIFT(nn.Module):
    """PIFT = group emb -> [CLS] + tokens -> Transformer encoder -> linear head."""

    def __init__(
        self,
        spec: DatasetGroupSpec,
        d_out: int,
        *,
        d_token: int = 128,
        n_blocks: int = 3,
        n_heads: int = 8,
        ffn_dropout: float = 0.1,
        use_group_emb: bool = True,
        weight_tied_groups: bool = True,
        set_pool: bool = False,
        use_nsi: bool = False,
        nsi_K: int = 16,
        nsi_kan_hidden: tuple = (32,),
        feature_dim: int | None = None,
        use_edge_tokens: bool = False,
        edge_k: int = 32,
        edge_select: str = "kt",
    ):
        super().__init__()
        self.spec = spec
        self.use_group_emb = use_group_emb
        if use_group_emb:
            if set_pool:
                self.group_emb = SetPooledGroupEmbedding(spec.groups, d_token=d_token)
            else:
                self.group_emb = PhysicsGroupEmbedding(
                    spec.groups, d_token=d_token, weight_tied=weight_tied_groups,
                )
        else:
            n_feats = sum(len(g.indices) for g in spec.groups)
            self.group_emb = _PerFeatureLinearEmbed(n_feats, d_token=d_token)

        # PIFT v3: optional Neural Symbolic Invariant Extractor → 1 [PHYSICS] token
        self.use_nsi = use_nsi
        if use_nsi:
            self.nsi = NeuralSymbolicInvariantExtractor(
                spec, K=nsi_K, kan_hidden=nsi_kan_hidden, feature_dim=feature_dim,
            )
            self.nsi_proj = nn.Sequential(
                nn.LayerNorm(nsi_K),
                nn.Linear(nsi_K, d_token),
            )

        # PIFT v4: optional pairwise Lorentz scalar edge tokens
        self.use_edge_tokens = use_edge_tokens
        if use_edge_tokens:
            self.edge_tokenizer = EdgeTokenizer(
                d_token=d_token, edge_k=edge_k, edge_select=edge_select,
            )
            # scaler stats for edge p4 reconstruction (mirrors NSI mechanism)
            F = feature_dim if feature_dim is not None else max(
                i for g in spec.groups for i in g.indices
            ) + 1
            self.register_buffer("edge_scaler_mean", torch.zeros(F))
            self.register_buffer("edge_scaler_std",  torch.ones(F))

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

    def set_edge_scaler_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Install StandardScaler stats for edge token p4 reconstruction."""
        self.edge_scaler_mean.copy_(mean.to(self.edge_scaler_mean.device).float())
        self.edge_scaler_std.copy_(std.to(self.edge_scaler_std.device).float().clamp_min(1e-6))

    def _extract_p4_per_group(self, x_scaled: torch.Tensor) -> torch.Tensor:
        """Build (B, n_kin_groups, 4) per-group 4-momenta from raw feature vec.
        Used by edge tokenizer."""
        x_raw = x_scaled.float() * self.edge_scaler_std + self.edge_scaler_mean
        p4_list = []
        for g in self.spec.groups:
            if g.kinematic_kind == "subjet":
                p4_list.append(x_raw[:, g.indices[:4]])              # (B, 4)
            elif g.kinematic_kind == "full":
                pT, eta, phi = x_raw[:, g.indices[0]], x_raw[:, g.indices[1]], x_raw[:, g.indices[2]]
                E  = pT * torch.cosh(eta)
                px = pT * torch.cos(phi)
                py = pT * torch.sin(phi)
                pz = pT * torch.sinh(eta)
                p4_list.append(torch.stack([E, px, py, pz], dim=-1))
            elif g.kinematic_kind == "transverse":
                mag, phi = x_raw[:, g.indices[0]], x_raw[:, g.indices[1]]
                E  = mag
                px = mag * torch.cos(phi)
                py = mag * torch.sin(phi)
                pz = torch.zeros_like(mag)
                p4_list.append(torch.stack([E, px, py, pz], dim=-1))
            elif g.kinematic_kind == "engineered":
                continue
        return torch.stack(p4_list, dim=1)                            # (B, n_kin, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.group_emb(x)
        if self.use_nsi:
            z = self.nsi(x)                                          # (B, K), fp32
            physics_tok = self.nsi_proj(z.to(tokens.dtype)).unsqueeze(1)
            tokens = torch.cat([physics_tok, tokens], dim=1)
        if self.use_edge_tokens:
            with torch.amp.autocast("cuda", enabled=False):
                p4 = self._extract_p4_per_group(x)                   # (B, n_kin, 4)
                edge_toks = self.edge_tokenizer(p4)                  # (B, k_edge, d_token)
            tokens = torch.cat([tokens, edge_toks.to(tokens.dtype)], dim=1)
        cls = self.cls.expand(tokens.size(0), -1, -1)
        h = torch.cat([cls, tokens], dim=1)
        h = self.encoder(h)
        h = self.norm(h[:, 0])
        return self.head(h)
