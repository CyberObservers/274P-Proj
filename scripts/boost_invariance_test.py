"""Boost-invariance regression test for trained PIFT models.

Loads a trained PIFT checkpoint, applies random longitudinal Lorentz boosts to
the test set features, and reports the prediction shift.

PIFT v3 (with NSI + boost regularizer) should give max|Δp| << PIFT v2.
This is a paper-quality numerical claim distinct from raw AUC.

Usage:
    python -m scripts.boost_invariance_test \
        --ckpt-v2 results/higgs/pift__low_level__seed0_tied/model.pt \
        --ckpt-v3 results/higgs/pift__low_level__seed0_v3/model.pt \
        --config configs/higgs.yaml --setup low_level \
        --out results/higgs/boost_invariance.json
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
from src.models.lorentz import apply_longitudinal_boost_in_kin


def load_pift_from_ckpt(ckpt_path: Path, config_path: Path, setup: str) -> tuple:
    cfg = yaml.safe_load(open(config_path))
    spec = get_groups(cfg["name"], setup)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args_cli = ckpt["args_cli"]
    use_nsi = bool(args_cli.get("pift_nsi", False))
    nsi_K = int(args_cli.get("nsi_k", 16))

    # peek d_in from scaler
    d_in = ckpt["scaler_mean"].shape[0]
    model = PIFT(
        spec, d_out=1,
        use_group_emb=not args_cli.get("pift_no_group", False),
        weight_tied_groups=not args_cli.get("pift_untied", False),
        set_pool=args_cli.get("pift_set_pool", False),
        use_nsi=use_nsi,
        nsi_K=nsi_K,
        feature_dim=d_in,
    )
    model.load_state_dict(ckpt["state_dict"])
    if use_nsi:
        model.nsi.set_scaler_stats(
            torch.from_numpy(ckpt["scaler_mean"]),
            torch.from_numpy(ckpt["scaler_scale"]),
        )
    model.eval()
    scaler_mean = torch.from_numpy(ckpt["scaler_mean"]).float()
    scaler_scale = torch.from_numpy(ckpt["scaler_scale"]).float()
    return model, scaler_mean, scaler_scale, spec, cfg


@torch.no_grad()
def evaluate_boost_shift(model, x_scaled, scaler_mean, scaler_scale, spec,
                         beta_values: list[float], device) -> dict:
    """For each β: predict on (a) original (b) longitudinally-boosted, return
    distribution stats of |Δσ(logit)| and AUC stability."""
    model = model.to(device)
    x_scaled = x_scaled.to(device)
    scaler_mean = scaler_mean.to(device)
    scaler_scale = scaler_scale.to(device)

    p_orig = torch.sigmoid(model(x_scaled).squeeze(-1)).cpu()

    out = {"per_beta": []}
    for beta_val in beta_values:
        beta = torch.tensor(beta_val, device=device, dtype=torch.float32)
        # de-standardize, boost, re-standardize
        x_raw = x_scaled * scaler_scale + scaler_mean
        x_raw_b = apply_longitudinal_boost_in_kin(x_raw, spec, beta)
        x_scaled_b = (x_raw_b - scaler_mean) / scaler_scale
        p_boost = torch.sigmoid(model(x_scaled_b).squeeze(-1)).cpu()
        diff = (p_orig - p_boost).abs()
        out["per_beta"].append({
            "beta": beta_val,
            "max_diff": float(diff.max()),
            "mean_diff": float(diff.mean()),
            "p99_diff": float(diff.quantile(0.99)),
        })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="path to model.pt")
    p.add_argument("--config", required=True)
    p.add_argument("--setup", default="low_level")
    p.add_argument("--n-test", type=int, default=10000)
    p.add_argument("--out", required=True)
    p.add_argument("--label", default="model", help="model label in output JSON")
    args = p.parse_args()

    model, sc_mean, sc_scale, spec, cfg = load_pift_from_ckpt(
        Path(args.ckpt), Path(args.config), args.setup,
    )

    Xtr, ytr, Xte, yte = load_split(cfg, args.setup)
    # standardize using ckpt's scaler stats (must match training-time scaling)
    Xte_s = (Xte - sc_mean.numpy()) / sc_scale.numpy()
    Xte_s = torch.from_numpy(Xte_s.astype(np.float32))[: args.n_test]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = evaluate_boost_shift(
        model, Xte_s, sc_mean, sc_scale, spec,
        beta_values=[0.05, 0.1, 0.2, 0.3], device=device,
    )
    result["label"] = args.label
    result["n_test"] = int(Xte_s.shape[0])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
