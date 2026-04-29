"""Render results/top_tagging_table.tex (a focused 1-column 8-method comparison).

Used for EXPERIMENT_REPORT.md §10.  Run:
    python scripts/render_top_tagging_table.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_ORDER = [
    ("xgboost",        "",                 "XGBoost"),
    ("mlp",            "",                 "MLP"),
    ("resnet",         "",                 "ResNet"),
    ("ft_transformer", "",                 "FT-Transformer"),
    ("pift",           "v4_subjet",        r"PIFT-Subjet (v2 group emb on subjets)"),
    ("pift",           "v4_nsi",           r"PIFT-NSI (v3) on subjets"),
    ("pift",           "v4_edge",          r"\textbf{PIFT-Edge (NEW)}"),
    ("pift",           "v4_subjet_edge",   r"\textbf{PIFT-Subjet+Edge (NEW combo)}"),
]


def cell(df, model: str, tag: str, metric="test_auc"):
    sub = df[(df["dataset"] == "top_tagging") & (df["model"] == model)]
    if tag == "":
        sub = sub[sub["tag"].isna() | (sub["tag"] == "")]
    else:
        sub = sub[sub["tag"] == tag]
    if sub.empty or sub[metric].isna().all():
        return float("nan"), float("nan")
    return float(sub[metric].mean()), float(sub[metric].std(ddof=0))


def fmt(mu, sd):
    if np.isnan(mu): return "—"
    if np.isnan(sd) or sd == 0: return f"{mu:.4f}"
    return f"{mu:.4f}{{\\tiny$\\pm${sd:.4f}}}"


def main():
    df = pd.read_csv("results/summary.csv")
    if "tag" not in df.columns:
        df["tag"] = ""

    rows = []
    aucs = []
    for model, tag, name in METHOD_ORDER:
        mu, sd = cell(df, model, tag)
        rows.append((name, mu, sd))
        aucs.append(mu)

    # find best/second
    valid = [(a, i) for i, a in enumerate(aucs) if not np.isnan(a)]
    valid.sort(reverse=True)
    best  = valid[0][1] if valid else None
    second = valid[1][1] if len(valid) > 1 else None

    body = []
    for r, (name, mu, sd) in enumerate(rows):
        s = fmt(mu, sd)
        if r == best:
            s = r"\textbf{" + s + "}"
        elif r == second:
            s = r"\underline{" + s + "}"
        body.append(f"{name} & {s} \\\\")
    midrule_at = 4

    body_with_rule = []
    for i, line in enumerate(body):
        if i == midrule_at:
            body_with_rule.append(r"\midrule")
        body_with_rule.append(line)

    tex = (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Top Quark Tagging Reference Dataset (Kasieczka 2019, 1.2M jets, "
        "binary top vs QCD).  All PIFT variants use anti-kT pre-tokenization to "
        "K=8 subjets (kinematic\\_kind=`subjet').  Mean$\\pm$std over 3 seeds.  "
        "Best in \\textbf{bold}, second-best \\underline{underlined}.}\n"
        "\\label{tab:top_tagging}\n\\footnotesize\n"
        "\\begin{tabular}{l c}\n\\toprule\n"
        "Method & Test AUC \\\\\n\\midrule\n"
        + "\n".join(body_with_rule) + "\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    Path("results/top_tagging_table.tex").write_text(tex)
    print("[render] results/top_tagging_table.tex")
    print(tex)


if __name__ == "__main__":
    main()
