"""Render results/v5_table.tex — PIFT v5 (Phase 1) ablation table.

Pulls v5 runs from results/summary.csv and emits a LaTeX table grouped by
dataset, with a row per (model, tag) pair.  Used in EXPERIMENT_REPORT §11.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# (dataset, model, setup, tag, display_label, group_id)
ROWS = [
    # HIGGS group
    ("higgs", "pift", "low_level", "",                  "PIFT v2 (baseline)",                 "higgs_low"),
    ("higgs", "pift", "low_level", "v5_tabm",           "PIFT v2 + TabM",                     "higgs_low"),
    ("higgs", "pift", "low_level", "v5_fastkan",        "PIFT v3 NSI (Fast-KAN)",             "higgs_low"),
    ("higgs", "pift", "low_level", "v5_tabm_fastkan",   "PIFT v2 + TabM + Fast-KAN",          "higgs_low"),
    ("higgs", "pift", "all",       "",                  "PIFT v2 (baseline, all)",            "higgs_all"),
    ("higgs", "pift", "all",       "v5_tabm",           "PIFT v2 + TabM (all)",               "higgs_all"),
    ("higgs", "pift", "all",       "v5_fastkan",        "PIFT v3 NSI Fast-KAN (all)",         "higgs_all"),
    # SUSY
    ("susy",  "pift", "low_level", "",                  "PIFT v2 (baseline)",                 "susy_low"),
    ("susy",  "pift", "low_level", "v5_tabm",           "PIFT v2 + TabM",                     "susy_low"),
    ("susy",  "pift", "all",       "",                  "PIFT v2 (baseline, all)",            "susy_all"),
    ("susy",  "pift", "all",       "v5_tabm",           "PIFT v2 + TabM (all)",               "susy_all"),
    # Top Tagging
    ("top_tagging", "pift", "all", "v4_edge",           "PIFT-Edge (v4 baseline)",            "tt"),
    ("top_tagging", "pift", "all", "v5_edge_cheby",     "PIFT-Edge + ChebyKAN",               "tt"),
    ("top_tagging", "pift", "all", "v5_edge_tabm",      "PIFT-Edge + TabM",                   "tt"),
    ("top_tagging", "pift", "all", "v5_edge_cheby_tabm","PIFT-Edge + ChebyKAN + TabM",        "tt"),
    ("top_tagging", "pift", "all", "v4_subjet",         "PIFT-Subjet (v4 baseline)",          "tt"),
    ("top_tagging", "pift", "all", "v5_subjet_tabm",    "PIFT-Subjet + TabM",                 "tt"),
    # Adult
    ("adult", "xgboost",        "all", "", "XGBoost (baseline)",                              "adult"),
    ("adult", "ft_transformer", "all", "", "FT-Transformer (baseline)",                       "adult"),
    ("adult", "mlp",            "all", "v5_tabm",       "MLP + TabM",                         "adult"),
    ("adult", "resnet",         "all", "v5_tabm",       "ResNet + TabM",                      "adult"),
    # Forest Cover (use acc, not auc)
    ("forest_cover", "xgboost",        "all", "", "XGBoost (baseline)",                       "forest"),
    ("forest_cover", "ft_transformer", "all", "", "FT-Transformer (baseline)",                "forest"),
    ("forest_cover", "mlp",            "all", "v5_tabm",       "MLP + TabM",                  "forest"),
    ("forest_cover", "resnet",         "all", "v5_tabm",       "ResNet + TabM",               "forest"),
]


GROUP_HEADERS = {
    "higgs_low":  "HIGGS (low-level, 21 features) — Test AUC",
    "higgs_all":  "HIGGS (all 28 features) — Test AUC",
    "susy_low":   "SUSY (low-level, 8 features) — Test AUC",
    "susy_all":   "SUSY (all 18 features) — Test AUC",
    "tt":         "Top Tagging (Kasieczka 2019, 200-constituent jets) — Test AUC",
    "adult":      "Adult (non-physics control) — Test AUC",
    "forest":     "Forest Cover (non-physics control) — Test Accuracy",
}


def cell(df: pd.DataFrame, ds: str, model: str, setup: str, tag: str, metric: str):
    sub = df[(df["dataset"] == ds) & (df["model"] == model) & (df["setup"] == setup)]
    if tag == "":
        sub = sub[sub["tag"].isna() | (sub["tag"] == "")]
    else:
        sub = sub[sub["tag"] == tag]
    if sub.empty or sub[metric].isna().all():
        return float("nan"), float("nan"), 0
    vals = sub[metric].dropna()
    return float(vals.mean()), float(vals.std(ddof=0)), int(len(vals))


def fmt(mu: float, sd: float) -> str:
    if np.isnan(mu):
        return "—"
    if np.isnan(sd) or sd == 0:
        return f"{mu:.4f}"
    return f"{mu:.4f}{{\\tiny$\\pm${sd:.4f}}}"


def main() -> None:
    csv = Path("results/summary.csv")
    if not csv.exists():
        raise SystemExit("results/summary.csv missing — run `python -m src.eval --root results --out results/summary.csv` first")
    df = pd.read_csv(csv)
    if "tag" not in df.columns:
        df["tag"] = ""

    body_lines = []
    last_group = None
    rows_with_data = []
    for ds, model, setup, tag, label, group in ROWS:
        metric = "test_acc" if group == "forest" else "test_auc"
        mu, sd, n = cell(df, ds, model, setup, tag, metric)
        rows_with_data.append((group, label, mu, sd, n))

    for group, label, mu, sd, n in rows_with_data:
        if group != last_group:
            if last_group is not None:
                body_lines.append(r"\midrule")
            body_lines.append(rf"\multicolumn{{3}}{{c}}{{\emph{{{GROUP_HEADERS[group]}}}}} \\")
            body_lines.append(r"\midrule")
            last_group = group
        score = fmt(mu, sd)
        body_lines.append(f"{label} & {score} & {n} \\\\")

    tex = (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{PIFT v5 (Phase 1) ablation: TabM ensemble × Fast-KAN NSI × ChebyKAN-Edge.  "
        "Mean$\\pm$std over 3 seeds; n = number of completed seeds.  "
        "Baselines from §3, §4, §10 are reproduced for direct comparison.}\n"
        "\\label{tab:pift_v5}\n\\footnotesize\n"
        "\\begin{tabular}{l c c}\n\\toprule\n"
        "Variant & Test metric & n \\\\\n\\midrule\n"
        + "\n".join(body_lines) + "\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    out = Path("results/v5_table.tex")
    out.write_text(tex)
    print(f"[render] {out}")
    print(tex)


if __name__ == "__main__":
    main()
