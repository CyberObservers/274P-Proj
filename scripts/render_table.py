"""Render results/main_table.tex (a `table*` for a two-column paper) from results/summary.csv.

Run:  python scripts/render_table.py
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd


# Order of method rows in the printed table (top → bottom).
METHOD_ORDER = [
    ("xgboost",                "",              "XGBoost"),
    ("mlp",                    "",              "MLP"),
    ("resnet",                 "",              "ResNet"),
    ("ft_transformer",         "",              "FT-Transformer"),
    ("pift",                   "",              r"PIFT$^\dagger$ (untied, v1)"),
    ("pift",                   "tied",          r"PIFT (v2, tied)"),
    ("pift",                   "setpool",       r"PIFT$_{\text{set-pool}}$"),
    ("pift",                   "v3",            r"\textbf{PIFT-v3 (ours, NSI)}"),
]

# (dataset, setup, metric_col, header) — column order in the printed table.
COLUMNS = [
    ("higgs", "low_level",  "test_auc", r"low (21)"),
    ("higgs", "high_level", "test_auc", r"high (7)"),
    ("higgs", "all",        "test_auc", r"all (28)"),
    ("susy",  "low_level",  "test_auc", r"low (8)"),
    ("susy",  "high_level", "test_auc", r"high (10)"),
    ("susy",  "all",        "test_auc", r"all (18)"),
    ("hepmass",      "all", "test_auc", r"all (27)"),
    ("forest_cover", "all", "test_acc", r"all (54)"),
    ("adult",        "all", "test_auc", r"all (14)"),
]


def cell_value(df: pd.DataFrame, dataset: str, setup: str, model: str,
               tag: str | None, metric: str) -> tuple[float, float, int]:
    sub = df[(df["dataset"] == dataset) & (df["setup"] == setup) &
             (df["model"] == model)]
    if tag is None:
        sub = sub  # any tag accepted
    elif tag == "":
        sub = sub[sub["tag"].isna() | (sub["tag"] == "")]
    else:
        sub = sub[sub["tag"] == tag]
    if sub.empty or sub[metric].isna().all():
        return float("nan"), float("nan"), 0
    return float(sub[metric].mean()), float(sub[metric].std(ddof=0)), int(len(sub))


def best_indices(values: list[float]) -> tuple[int | None, int | None]:
    """Return (best_idx, second_idx) for the largest two values, ignoring NaN."""
    pairs = [(v, i) for i, v in enumerate(values) if not np.isnan(v)]
    pairs.sort(reverse=True)
    best = pairs[0][1] if pairs else None
    second = pairs[1][1] if len(pairs) > 1 else None
    return best, second


def fmt(mu: float, std: float) -> str:
    if np.isnan(mu): return "—"
    if np.isnan(std) or std == 0: return f"{mu:.4f}"
    return f"{mu:.4f}{{\\tiny$\\pm${std:.4f}}}"


def render_table(df: pd.DataFrame) -> str:
    n_cols = len(COLUMNS)
    # column-major: list of values per column for best/second highlighting
    cell_grid = []  # rows = methods, cols = columns
    for model, tag, _ in METHOD_ORDER:
        row = []
        for dataset, setup, metric, _ in COLUMNS:
            mu, sd, _ = cell_value(df, dataset, setup, model, tag, metric)
            row.append((mu, sd))
        cell_grid.append(row)

    # find best/second per column
    bold = [None] * n_cols; second = [None] * n_cols
    for c in range(n_cols):
        col_vals = [cell_grid[r][c][0] for r in range(len(METHOD_ORDER))]
        bold[c], second[c] = best_indices(col_vals)

    # build LaTeX
    col_groups = (
        r"& \multicolumn{3}{c}{\textbf{HIGGS} (1M, AUC)}"
        r"& \multicolumn{3}{c}{\textbf{SUSY} (1M, AUC)}"
        r"& \textbf{HEPMASS} & \textbf{Forest} & \textbf{Adult} \\"
    )
    cmidrules = (
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}"
        r"\cmidrule(lr){8-8}\cmidrule(lr){9-9}\cmidrule(lr){10-10}"
    )
    col_header = "Method " + "".join(f"& {h} " for _, _, _, h in COLUMNS) + r"\\"
    metric_row = (
        r"& AUC & AUC & AUC & AUC & AUC & AUC & AUC & Acc & AUC \\"
    )

    body_lines = []
    for r, (_, _, name) in enumerate(METHOD_ORDER):
        cells = [name]
        for c in range(n_cols):
            mu, sd = cell_grid[r][c]
            cell = fmt(mu, sd)
            if r == bold[c]:
                cell = r"\textbf{" + cell + "}"
            elif r == second[c]:
                cell = r"\underline{" + cell + "}"
            cells.append(cell)
        body_lines.append(" & ".join(cells) + r" \\")

    midrule_indices = [4]  # midrule before PIFT block
    body = []
    for i, line in enumerate(body_lines):
        if i in midrule_indices: body.append(r"\midrule")
        body.append(line)

    tex = dedent(r"""
    \begin{table*}[t]
    \centering
    \caption{Test AUC / accuracy on HEP and control tabular benchmarks (mean$\pm$std over 3 seeds).
    HIGGS / SUSY admit three input \emph{setups} of increasing engineering: low-level raw kinematics, high-level
    Lorentz invariants, or all features.  Best per column in \textbf{bold}, second-best \underline{underlined};
    PIFT$^\dagger$ is the v1 untied baseline, kept for ablation.}
    \label{tab:main}
    \footnotesize
    \setlength{\tabcolsep}{4pt}
    \begin{tabular}{l ccc ccc c c c}
    \toprule
    """).lstrip()
    tex += col_groups + "\n" + cmidrules + "\n"
    tex += col_header + "\n" + metric_row + "\n"
    tex += r"\midrule" + "\n"
    tex += "\n".join(body) + "\n"
    tex += r"\bottomrule" + "\n"
    tex += r"\end{tabular}" + "\n"
    tex += r"\end{table*}" + "\n"
    return tex


def render_standalone(table_body: str) -> str:
    """Wrap the table in a minimal compilable .tex doc using twocolumn so table* spans both."""
    return dedent(r"""
    \documentclass[twocolumn]{article}
    \usepackage[margin=1in]{geometry}
    \usepackage{booktabs}
    \usepackage{multirow}
    \usepackage{amsmath,amssymb}
    \title{PIFT --- Main Results}
    \date{}
    \begin{document}
    \maketitle

    """).lstrip() + table_body + "\n" + r"\end{document}" + "\n"


def main() -> None:
    df = pd.read_csv("results/summary.csv")
    if "tag" not in df.columns:
        df["tag"] = ""
    table = render_table(df)
    Path("results/main_table.tex").write_text(table)
    Path("results/main_table_standalone.tex").write_text(render_standalone(table))
    print("[render] results/main_table.tex (snippet for inclusion)")
    print("[render] results/main_table_standalone.tex (compilable)")


if __name__ == "__main__":
    main()
