"""Post-hoc analysis of PIFT v3 NSI: 'what did the K invariants learn?'

For a trained PIFT-v3 HIGGS-all checkpoint:
  1. Run NSI forward over the test set, collect z ∈ ℝ^{N×K}.
  2. Compute Spearman correlation of each z_k vs the 7 engineered Lorentz
     invariants in HIGGS all (m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb, m_wwbb).
     A high |corr| means z_k re-discovered an engineered physical quantity.
  3. KDE histogram for each z_k → save 4×4 grid figure.
  4. Save:
        results/higgs/nsi_correlation.json     — corr matrix + best-match per k
        results/figures/nsi_invariants.pdf      — KDE grid

Usage:
    python -m scripts.nsi_symbolic_readout \
        --ckpt results/higgs/pift__all__seed0_v3/model.pt \
        --config configs/higgs.yaml --setup all \
        --out-dir results/higgs/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.datasets import load_split
from src.data.feature_groups import get_groups
from src.models.pift import PIFT


# HIGGS all-features schema: indices 21..27 are the 7 engineered Lorentz invariants
ENGINEERED_NAMES = ["m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb"]
ENGINEERED_INDICES = list(range(21, 28))


def load_pift_from_ckpt(ckpt_path: Path, config_path: Path, setup: str):
    cfg = yaml.safe_load(open(config_path))
    spec = get_groups(cfg["name"], setup)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args_cli = ckpt["args_cli"]
    if not args_cli.get("pift_nsi", False):
        raise ValueError("checkpoint was not trained with --pift-nsi")
    d_in = ckpt["scaler_mean"].shape[0]
    model = PIFT(
        spec, d_out=1,
        weight_tied_groups=not args_cli.get("pift_untied", False),
        use_nsi=True,
        nsi_K=int(args_cli["nsi_k"]),
        feature_dim=d_in,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.nsi.set_scaler_stats(
        torch.from_numpy(ckpt["scaler_mean"]),
        torch.from_numpy(ckpt["scaler_scale"]),
    )
    model.eval()
    return model, ckpt, cfg


@torch.no_grad()
def collect_invariants(model, X_scaled, batch_size=4096, device="cuda"):
    model = model.to(device)
    Z = []
    n = X_scaled.shape[0]
    for i in range(0, n, batch_size):
        xb = torch.from_numpy(X_scaled[i:i+batch_size]).to(device)
        z = model.nsi(xb).cpu().numpy()
        Z.append(z)
    return np.concatenate(Z, axis=0)  # (N, K)


def spearman_corr_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Spearman corr between cols of A (N,K) and cols of B (N,M).  Returns (K,M)."""
    from scipy.stats import rankdata
    Ar = np.apply_along_axis(rankdata, 0, A)
    Br = np.apply_along_axis(rankdata, 0, B)
    Ar -= Ar.mean(axis=0); Br -= Br.mean(axis=0)
    Ar /= (Ar.std(axis=0) + 1e-12); Br /= (Br.std(axis=0) + 1e-12)
    return (Ar.T @ Br) / Ar.shape[0]   # (K, M)


def render_kde_grid(Z: np.ndarray, corr: np.ndarray, eng_names: list,
                    out_pdf: Path, K: int, fig_cols: int = 4) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    fig_rows = (K + fig_cols - 1) // fig_cols
    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(fig_cols * 3.2, fig_rows * 2.6))
    axes = np.array(axes).reshape(-1)
    for k in range(K):
        ax = axes[k]
        zk = Z[:, k]
        if zk.std() < 1e-6:
            ax.text(0.5, 0.5, "(constant)", ha="center", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        # subsample for KDE speed
        zk_s = zk[np.random.default_rng(0).choice(len(zk), size=min(20000, len(zk)), replace=False)]
        kde = gaussian_kde(zk_s)
        x = np.linspace(zk_s.min(), zk_s.max(), 300)
        ax.plot(x, kde(x), lw=1.2)
        ax.fill_between(x, kde(x), alpha=0.25)
        # title with best-match engineered feature
        best_m = int(np.argmax(np.abs(corr[k])))
        best_r = corr[k, best_m]
        ax.set_title(f"z_{k}  ~ {eng_names[best_m]}  ρ={best_r:+.2f}", fontsize=8)
        ax.tick_params(labelsize=6)
    for k in range(K, len(axes)):
        axes[k].axis("off")
    fig.suptitle("PIFT-v3 NSI: learned invariant distributions vs engineered Lorentz scalars",
                 fontsize=10)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"[plot] {out_pdf}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--setup", default="all",
                   help="Use 'all' to access engineered features for correlation analysis")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-test", type=int, default=50000)
    args = p.parse_args()

    model, ckpt, cfg = load_pift_from_ckpt(Path(args.ckpt), Path(args.config), args.setup)
    Xtr, ytr, Xte, yte = load_split(cfg, args.setup)
    Xte = Xte[: args.n_test]

    # Re-standardize using checkpoint's scaler stats
    sc_mean = ckpt["scaler_mean"]
    sc_scale = ckpt["scaler_scale"]
    Xte_s = ((Xte - sc_mean) / sc_scale).astype(np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Z = collect_invariants(model, Xte_s, device=device)
    K = Z.shape[1]
    print(f"[nsi] z shape={Z.shape}  per-output std: min={Z.std(axis=0).min():.3f}  "
          f"max={Z.std(axis=0).max():.3f}")

    # Engineered features (only for HIGGS all)
    if args.setup == "all" and Xte.shape[1] >= 28:
        E = Xte[:, ENGINEERED_INDICES]                          # (N, 7), un-scaled
        corr = spearman_corr_matrix(Z, E)                       # (K, 7)
        # best match per z_k
        best = []
        for k in range(K):
            m = int(np.argmax(np.abs(corr[k])))
            best.append({
                "k": k, "best_match": ENGINEERED_NAMES[m],
                "spearman_rho": float(corr[k, m]),
                "all_corr": {ENGINEERED_NAMES[i]: float(corr[k, i]) for i in range(7)},
                "z_std": float(Z[:, k].std()),
            })
        out = {
            "K": K, "n_test": Z.shape[0],
            "engineered_names": ENGINEERED_NAMES,
            "per_invariant": best,
            "n_match_above_0.5": int(np.sum(np.max(np.abs(corr), axis=1) > 0.5)),
            "n_match_above_0.3": int(np.sum(np.max(np.abs(corr), axis=1) > 0.3)),
        }
    else:
        corr = np.zeros((K, len(ENGINEERED_NAMES)))
        out = {"K": K, "n_test": Z.shape[0], "note": "setup != 'all', no engineered corr"}

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "nsi_correlation.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[json] {out_dir}/nsi_correlation.json")

    pdf_path = Path("results/figures/nsi_invariants.pdf")
    render_kde_grid(Z, corr, ENGINEERED_NAMES, pdf_path, K=K)


if __name__ == "__main__":
    main()
