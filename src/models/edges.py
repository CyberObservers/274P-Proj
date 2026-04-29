"""Pairwise Lorentz scalar edge tokens for PIFT-Edge.

For each pair (i, j) of node tokens (subjets or particles), compute four
Lorentz scalars:
    m_ij²    = (p_i + p_j)·(p_i + p_j)        — pair invariant mass squared
    ΔR_ij    = sqrt(Δη² + Δφ²)                — angular distance
    k_T_ij   = min(pT_i, pT_j) · ΔR_ij        — soft-collinear momentum
    z_ij     = min(pT_i, pT_j) / (pT_i+pT_j)  — momentum fraction

These are the standard "soft-drop / Cambridge-Aachen" pairwise features used by
ParT (2022).  Where ParT uses them as additive *attention bias*, we promote
each pair to a learnable token so the transformer can do edge↔edge attention.

Public API:
- compute_pairwise_invariants(p4) -> (B, N, N, 4)
- EdgeTokenizer(d_token, edge_k=32, edge_select="kt")
"""
from __future__ import annotations

import torch
import torch.nn as nn


def compute_pairwise_invariants(p4: torch.Tensor) -> torch.Tensor:
    """Compute the four Lorentz scalars per pair (i, j).

    Args:
        p4: (B, N, 4)  per-token 4-momenta in (E, px, py, pz) order

    Returns:
        edges: (B, N, N, 4)  per-pair features [ln(m²+ε), ln(ΔR+ε),
                                                ln(k_T+ε), ln(z+ε)]
        Diagonals are zero.  ε=1e-6 to avoid log(0) on padded tokens.
    """
    B, N, _ = p4.shape
    eps = 1e-6

    E  = p4[..., 0]                    # (B, N)
    px = p4[..., 1]
    py = p4[..., 2]
    pz = p4[..., 3]
    pT = torch.sqrt(px ** 2 + py ** 2 + eps)

    # m² = (pi + pj)^2  with metric (+,-,-,-)
    Esum  = E[:, :, None] + E[:, None, :]
    pxsum = px[:, :, None] + px[:, None, :]
    pysum = py[:, :, None] + py[:, None, :]
    pzsum = pz[:, :, None] + pz[:, None, :]
    m_sq = Esum ** 2 - pxsum ** 2 - pysum ** 2 - pzsum ** 2     # (B, N, N)
    m_sq = m_sq.clamp_min(0.0)

    # ΔR = sqrt(Δη² + Δφ²)
    eta = 0.5 * torch.log((E + pz + eps) / (E - pz + eps))
    phi = torch.atan2(py, px)
    deta = eta[:, :, None] - eta[:, None, :]
    dphi = phi[:, :, None] - phi[:, None, :]
    dphi = (dphi + torch.pi) % (2 * torch.pi) - torch.pi
    deltaR = torch.sqrt(deta ** 2 + dphi ** 2 + eps)

    # k_T and z
    pT_min = torch.minimum(pT[:, :, None], pT[:, None, :])
    pT_sum = pT[:, :, None] + pT[:, None, :] + eps
    k_T = pT_min * deltaR
    z = pT_min / pT_sum

    edges = torch.stack([
        torch.log(m_sq + eps),
        torch.log(deltaR + eps),
        torch.log(k_T + eps),
        torch.log(z + eps),
    ], dim=-1)                                                  # (B, N, N, 4)

    # zero out diagonal (i == j)
    diag = torch.eye(N, device=p4.device, dtype=torch.bool)
    edges = edges.masked_fill(diag[None, :, :, None], 0.0)
    return edges


class EdgeTokenizer(nn.Module):
    """Embed top-K pairwise Lorentz scalars into d_token tokens.

    Args:
        d_token: output token dim
        edge_k:  how many edges per event to keep (top-K by selection score)
                 If edge_k <= 0 or >= N(N-1)/2, keep all upper-triangular edges.
        edge_select: "kt" (top by k_T), "mass" (top by m²), or "all" (no select)

    Output shape: (B, k_keep, d_token)
    """

    def __init__(self, d_token: int, edge_k: int = 32,
                 edge_select: str = "kt", hidden_mult: float = 1.0):
        super().__init__()
        self.edge_k = edge_k
        self.edge_select = edge_select
        d_h = max(d_token, int(d_token * hidden_mult))
        self.mlp = nn.Sequential(
            nn.Linear(4, d_h), nn.GELU(),
            nn.Linear(d_h, d_token),
        )

    def forward(self, p4: torch.Tensor) -> torch.Tensor:
        """p4: (B, N, 4)  ->  edge_tokens: (B, k_keep, d_token)"""
        B, N, _ = p4.shape
        edges = compute_pairwise_invariants(p4)                  # (B, N, N, 4)

        # Take upper triangular indices i<j to avoid double-counting
        iu, ju = torch.triu_indices(N, N, offset=1, device=p4.device)  # (n_pairs,) each
        n_pairs = iu.numel()
        edge_feats = edges[:, iu, ju, :]                         # (B, n_pairs, 4)

        # Select top-K (or keep all)
        k_keep = min(self.edge_k if self.edge_k > 0 else n_pairs, n_pairs)
        if k_keep < n_pairs:
            if self.edge_select == "kt":
                # ln k_T is at index 2; larger k_T = harder, more important
                score = edge_feats[..., 2]
            elif self.edge_select == "mass":
                score = edge_feats[..., 0]                       # ln m²
            elif self.edge_select == "all":
                score = torch.zeros(B, n_pairs, device=p4.device)
            else:
                raise ValueError(f"unknown edge_select={self.edge_select!r}")
            # top-k per batch
            topk_idx = score.topk(k_keep, dim=-1).indices         # (B, k_keep)
            edge_feats = torch.gather(
                edge_feats, 1, topk_idx[..., None].expand(-1, -1, 4)
            )
        return self.mlp(edge_feats)                              # (B, k_keep, d_token)
