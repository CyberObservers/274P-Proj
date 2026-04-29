"""Render EXPERIMENT_REPORT.md from results/summary.csv.

Reads:  results/summary.csv  +  results/permutation_test.json (optional)
Writes: EXPERIMENT_REPORT.md  at repo root
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd


SETUP_LABEL = {"low_level": "low (21 features)", "high_level": "high (7 features)", "all": "all (28 features)"}
MODEL_LABEL = {
    "xgboost": "XGBoost", "mlp": "MLP", "resnet": "ResNet",
    "ft_transformer": "FT-Transformer", "pift": "**PIFT (ours)**",
}
DATASETS_HEP = ["higgs", "susy", "hepmass"]
DATASETS_CTRL = ["forest_cover", "adult"]


def fmt(v, std=None):
    if pd.isna(v): return "—"
    if std is not None and not pd.isna(std):
        return f"{v:.4f}±{std:.4f}"
    return f"{v:.4f}"


def main_table_for(df: pd.DataFrame, dataset: str, metric: str = "test_auc") -> str:
    sub = df[(df["dataset"] == dataset) & ((df["tag"].isna()) | (df["tag"] == ""))]
    if sub.empty: return f"_no runs for {dataset}_\n"
    setups = sorted(sub["setup"].unique(), key=lambda s: ["low_level", "high_level", "all"].index(s))
    models = ["xgboost", "mlp", "resnet", "ft_transformer", "pift"]
    rows = ["| model |" + "|".join(f" {SETUP_LABEL.get(s, s)} " for s in setups) + "|",
            "|---|" + "|".join("---" for _ in setups) + "|"]
    means = {(s, m): sub[(sub["setup"] == s) & (sub["model"] == m)][metric] for s in setups for m in models}
    # find best per setup
    best = {s: max((means[(s, m)].mean() for m in models if not means[(s, m)].empty), default=np.nan) for s in setups}
    for m in models:
        cells = [f" {MODEL_LABEL.get(m, m)} "]
        for s in setups:
            vals = means[(s, m)]
            if vals.empty:
                cells.append(" — ")
                continue
            mu, sd = vals.mean(), vals.std()
            cell = fmt(mu, sd if not pd.isna(sd) else None)
            if not pd.isna(mu) and not pd.isna(best[s]) and abs(mu - best[s]) < 1e-6:
                cell = f"**{cell}**"
            cells.append(f" {cell} ")
        rows.append("|".join(cells) + "|")
    return "\n".join(rows) + "\n"


def control_table(df: pd.DataFrame) -> str:
    sub = df[df["dataset"].isin(DATASETS_CTRL) & ((df["tag"].isna()) | (df["tag"] == ""))]
    if sub.empty: return "_no control runs_\n"
    rows = ["| dataset | model | metric | value |", "|---|---|---|---|"]
    for ds in DATASETS_CTRL:
        for m in ["xgboost", "ft_transformer"]:
            r = sub[(sub["dataset"] == ds) & (sub["model"] == m)]
            if r.empty: continue
            mname = "test_auc" if "test_auc" in r and not r["test_auc"].isna().all() else "test_acc"
            mu, sd = r[mname].mean(), r[mname].std()
            rows.append(f"| {ds} | {MODEL_LABEL.get(m, m)} | {mname} | {fmt(mu, sd)} |")
    return "\n".join(rows) + "\n"


def learning_curve_table(df: pd.DataFrame) -> str:
    sub = df[(df["dataset"] == "higgs") & (df["setup"] == "low_level")].copy()
    sub["n"] = sub["tag"].fillna("").str.extract(r"n(\d+)").astype(float)
    sub.loc[sub["tag"].fillna("") == "", "n"] = 1_000_000
    sub = sub[sub["n"].notna() & sub["model"].isin(["ft_transformer", "pift"])]
    if sub.empty: return "_no learning-curve runs_\n"
    rows = ["| n_train | FT-Transformer | PIFT | gain |", "|---|---|---|---|"]
    for n in sorted(sub["n"].unique()):
        r = sub[sub["n"] == n]
        ft = r[r["model"] == "ft_transformer"]["test_auc"]
        pi = r[r["model"] == "pift"]["test_auc"]
        ft_mu, ft_sd = ft.mean(), ft.std()
        pi_mu, pi_sd = pi.mean(), pi.std()
        gain = pi_mu - ft_mu
        rows.append(f"| {int(n):,} | {fmt(ft_mu, ft_sd)} | {fmt(pi_mu, pi_sd)} | {gain:+.4f} |")
    return "\n".join(rows) + "\n"


def ablation_table(df: pd.DataFrame) -> str:
    sub = df[(df["dataset"] == "higgs") & (df["setup"] == "low_level") & (df["model"] == "pift")].copy()
    sub["variant"] = sub["tag"].fillna("").map({
        "": "full", "abl_no_inv": "no_inv", "abl_no_sym": "no_sym", "abl_no_group": "no_group",
    })
    sub = sub[sub["variant"].notna()]
    if sub.empty: return "_no ablation runs_\n"
    rows = ["| variant | test_auc | Δ vs full |", "|---|---|---|"]
    full_mu = sub[sub["variant"] == "full"]["test_auc"].mean()
    for v in ["full", "no_inv", "no_sym", "no_group"]:
        r = sub[sub["variant"] == v]["test_auc"]
        if r.empty: continue
        mu, sd = r.mean(), r.std()
        delta = mu - full_mu if v != "full" else 0.0
        d_str = "—" if v == "full" else f"{delta:+.4f}"
        rows.append(f"| {v} | {fmt(mu, sd)} | {d_str} |")
    return "\n".join(rows) + "\n"


def fallback_story(df: pd.DataFrame) -> str:
    sub = df[(df["dataset"] == "higgs") & (df["setup"] == "low_level") &
             ((df["tag"].isna()) | (df["tag"] == ""))]
    ft = sub[sub["model"] == "ft_transformer"]["test_auc"].mean()
    pi = sub[sub["model"] == "pift"]["test_auc"].mean()
    if pd.isna(ft) or pd.isna(pi): return "_insufficient data to score story_"
    gap = pi - ft
    if gap >= 0.01:
        return (f"🟢 **GREEN** — PIFT 在 HIGGS low-level 上比 FT-T 高 **{gap:+.4f}** AUC，达成提案 §7 头条目标。\n"
                f"叙事建议：*PIFT: Physics Priors Close the Gap Between DL and XGBoost on Tabular HEP Data*.")
    if gap >= 0.003:
        return (f"🟡 **YELLOW** — gap **{gap:+.4f}** 不足 0.01，看 learning curve 上小数据是否更显著。\n"
                f"叙事建议：data efficiency story（提案 §7 中线）。")
    return (f"🔴 **RED** — gap **{gap:+.4f}** ≤ 0，落到提案 §7 兜底分支：诚实报告 + 消融解释为何 self-attn 已隐式学到部分物理。")


def main() -> None:
    df = pd.read_csv("results/summary.csv")
    if "tag" not in df: df["tag"] = ""
    perm = {}
    p = Path("results/permutation_test.json")
    if p.exists():
        perm = json.loads(p.read_text())
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        commit = "(no git)"

    md = dedent(f"""
    # PIFT — Experiment Report

    *CS 274P, Baldi.  Generated against commit `{commit}`. Dataset / hyperparameter
    settings follow [project_proposals.md](project_proposals.md) §2-§7.*

    ## §1 Setup recap

    - Datasets: HIGGS (1M train), SUSY (1M), HEPMASS (500K), Forest Cover (control), Adult (control).
    - Models: XGBoost / MLP / ResNet / FT-Transformer (rtdl_revisiting_models 0.0.2) / **PIFT (ours)**.
    - Setups for HIGGS / SUSY: low-level (raw kinematics), high-level (engineered invariants), all.
    - 3 seeds per cell; report mean±std over seeds.
    - Hardware: 2× RTX 3090, CUDA 13.0, torch 2.11+cu130; serial-per-GPU dispatcher.

    ## §2 Baseline reproduction (vs Baldi 2014)

    Baldi 2014 reported AUC ≈ 0.88 for HIGGS all-features with a 5-layer DNN; our
    FT-Transformer baseline on the same setting:

    """).strip() + "\n\n"

    md += main_table_for(df, "higgs") + "\n"

    md += dedent("""
    ## §3 Main HEP results

    ### HIGGS
    """).strip() + "\n\n" + main_table_for(df, "higgs") + "\n"
    md += "### SUSY\n\n" + main_table_for(df, "susy") + "\n"
    md += "### HEPMASS\n\n" + main_table_for(df, "hepmass") + "\n"

    md += dedent("""
    ## §4 Control datasets — verify PIFT does not regress non-physics tabular

    """).strip() + "\n\n" + control_table(df) + "\n"

    md += dedent("""
    ## §5 Learning curve — HIGGS low-level
    See `results/figures/learning_curve.png`.

    """).strip() + "\n\n" + learning_curve_table(df) + "\n"

    md += dedent("""
    ## §6 Ablation — PIFT components on HIGGS low-level
    See `results/figures/ablation.png`.

    """).strip() + "\n\n" + ablation_table(df) + "\n"

    if perm:
        md += dedent(f"""
        **Permutation invariance**: swapping jet_1 ↔ jet_3 indices on a 512-row
        test batch yields max\\|Δ\\| = **{perm['max_diff']:.2e}** (mean
        \\|Δ\\| = {perm['mean_diff']:.2e}); pass = `{perm['passed']}`.
        """).strip() + "\n\n"

    md += dedent("""
    ## §7 Conclusion

    """).strip() + "\n\n" + fallback_story(df) + "\n\n"

    md += dedent("""
    ## §8 Limitations

    - Only HEP datasets; not validated on chem/bio.
    - PIFT physics group config is hand-specified, not auto-discovered.
    - Learning curve max is 1M rows (proposal §6 caps here); not pre-training scale.
    - Only two control datasets (Forest Cover, Adult); cannot rule out negative
      effects on every non-physics tabular family.
    - Invariant list is a curated subset of Lorentz invariants, not exhaustive.

    ## §9 Reproduction

    ```bash
    git clone https://github.com/CyberObservers/274P-Proj.git
    cd 274P-Proj
    conda env create -f environment.yml && conda activate pift
    bash scripts/download_data.sh
    for cfg in higgs susy hepmass forest_cover adult; do
        python -m src.data.datasets --preprocess --config configs/${cfg}.yaml
    done
    bash scripts/dispatch.sh 0 scripts/queue_gpu0.txt &
    bash scripts/dispatch.sh 1 scripts/queue_gpu1.txt &
    wait
    bash scripts/dispatch.sh 0 scripts/queue_gpu0_lc.txt &
    bash scripts/dispatch.sh 1 scripts/queue_gpu1_lc.txt &
    wait
    python -m src.eval --root results --out results/summary.csv
    python -m src.plot
    python scripts/render_report.py
    ```
    """).strip() + "\n"

    Path("EXPERIMENT_REPORT.md").write_text(md)
    print("[render] EXPERIMENT_REPORT.md written")


if __name__ == "__main__":
    main()
