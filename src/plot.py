"""Generate report figures from results/summary.csv."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = ["xgboost", "mlp", "resnet", "ft_transformer", "pift"]
SETUP_ORDER = ["low_level", "high_level", "all"]
COLOR = {
    "xgboost":         "#888888",
    "mlp":             "#9bd3a8",
    "resnet":          "#5fb878",
    "ft_transformer":  "#3b7bbf",
    "pift":            "#d6485f",
    "pift_no_inv":     "#d68762",
    "pift_no_sym":     "#d6c462",
    "pift_no_group":   "#9c62d6",
}


def _agg(df: pd.DataFrame, group_cols, metric="test_auc"):
    return df.groupby(group_cols, dropna=False)[metric].agg(["mean", "std", "count"]).reset_index()


def plot_main_table(df: pd.DataFrame, out: Path, dataset: str, metric: str = "test_auc"):
    sub = df[(df["dataset"] == dataset) & df["model"].isin(MODEL_ORDER)].copy()
    if sub.empty: return
    agg = _agg(sub, ["setup", "model"], metric)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    setups = [s for s in SETUP_ORDER if s in agg["setup"].unique()]
    width = 0.16
    x = np.arange(len(setups))
    for i, m in enumerate(MODEL_ORDER):
        rows = agg[agg["model"] == m].set_index("setup").reindex(setups)
        if rows["mean"].isna().all(): continue
        ax.bar(x + (i - 2) * width, rows["mean"], width=width,
               yerr=rows["std"].fillna(0), capsize=3,
               label=m, color=COLOR.get(m, None))
    ax.set_xticks(x); ax.set_xticklabels(setups)
    ax.set_ylabel(metric); ax.set_title(f"{dataset} — {metric}")
    ax.legend(loc="lower right", fontsize=8); ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def plot_learning_curve(df: pd.DataFrame, out: Path, dataset: str = "higgs"):
    sub = df[(df["dataset"] == dataset) & (df["setup"] == "low_level")].copy()
    sub["n"] = sub["tag"].str.extract(r"n(\d+)").astype(float)
    # 1M comes from un-tagged runs
    sub.loc[sub["tag"].isna() | (sub["tag"] == ""), "n"] = 1_000_000
    sub = sub[sub["n"].notna()]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in ["ft_transformer", "pift"]:
        rows = sub[sub["model"] == m].sort_values("n")
        if rows.empty: continue
        agg = rows.groupby("n")["test_auc"].agg(["mean", "std"]).reset_index()
        ax.errorbar(agg["n"], agg["mean"], yerr=agg["std"].fillna(0),
                    marker="o", capsize=3, label=m, color=COLOR.get(m, None))
    ax.set_xscale("log"); ax.set_xlabel("training samples"); ax.set_ylabel("test AUC")
    ax.set_title(f"Learning curve — {dataset} low-level")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def plot_ablation(df: pd.DataFrame, out: Path):
    sub = df[(df["dataset"] == "higgs") & (df["setup"] == "low_level") & (df["model"] == "pift")].copy()
    sub["variant"] = sub["tag"].fillna("").map({
        "": "full",
        "abl_no_inv": "no_inv",
        "abl_no_sym": "no_sym",
        "abl_no_group": "no_group",
    })
    sub = sub[sub["variant"].notna()]
    if sub.empty: return
    agg = sub.groupby("variant")["test_auc"].agg(["mean", "std"]).reindex(
        ["full", "no_inv", "no_sym", "no_group"]
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(agg.index, agg["mean"], yerr=agg["std"].fillna(0), capsize=3,
                  color=[COLOR["pift"], COLOR["pift_no_inv"], COLOR["pift_no_sym"], COLOR["pift_no_group"]])
    ax.set_ylabel("test AUC"); ax.set_title("PIFT ablation — HIGGS low-level")
    ax.grid(True, alpha=0.3, axis="y")
    for b, v in zip(bars, agg["mean"]):
        if not np.isnan(v):
            ax.text(b.get_x() + b.get_width()/2, v + 0.001, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def plot_setup_comparison(df: pd.DataFrame, out: Path):
    sub = df[df["dataset"].isin(["higgs", "susy"]) & df["model"].isin(["ft_transformer", "pift"])].copy()
    if sub.empty: return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, ds in zip(axes, ["higgs", "susy"]):
        s = sub[sub["dataset"] == ds]
        agg = _agg(s, ["setup", "model"])
        setups = [x for x in SETUP_ORDER if x in agg["setup"].unique()]
        x = np.arange(len(setups)); width = 0.35
        for i, m in enumerate(["ft_transformer", "pift"]):
            rows = agg[agg["model"] == m].set_index("setup").reindex(setups)
            ax.bar(x + (i - 0.5) * width, rows["mean"], width=width,
                   yerr=rows["std"].fillna(0), capsize=3,
                   label=m, color=COLOR[m])
        ax.set_xticks(x); ax.set_xticklabels(setups); ax.set_title(ds.upper())
        ax.set_ylabel("test AUC"); ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("FT-T vs PIFT across setups")
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default="results/summary.csv")
    p.add_argument("--out-dir", default="results/figures")
    args = p.parse_args()
    df = pd.read_csv(args.summary)
    # parse tag column from run_dir name if missing
    if "tag" not in df.columns:
        df["tag"] = ""
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    for ds in df["dataset"].unique():
        plot_main_table(df, out / f"main_{ds}.png", ds)
    plot_learning_curve(df, out / "learning_curve.png")
    plot_ablation(df, out / "ablation.png")
    plot_setup_comparison(df, out / "setup_comparison.png")
    print(f"[plot] figures -> {out}/")


if __name__ == "__main__":
    main()
