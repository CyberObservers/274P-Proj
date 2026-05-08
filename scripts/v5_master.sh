#!/usr/bin/env bash
# Master orchestrator for PIFT v5 (Phase 1: TabM × Fast-KAN × ChebyKAN-Edge).
#  1. Verify deps (fastkan, tabm) installed
#  2. Run pytest tests/test_v5.py
#  3. Dispatch dual-GPU queues (~45 runs, ~12-15h)
#  4. Aggregate metrics + render Top Tagging + v5 tables
#  5. Print summary block
set -uo pipefail

cd /home/lawrence/274P
source /home/lawrence/anaconda3/etc/profile.d/conda.sh
conda activate pift

LOG=logs/v5_master.log
mkdir -p logs
exec > >(tee -a "$LOG") 2>&1

echo "==[$(date)]== v5 master start"

# ---------------------------------------------------------------------------
# Phase 0: deps + tests
# ---------------------------------------------------------------------------
echo "==[$(date)]== Phase 0: verify deps + run pytest"
python -c "import fastkan, tabm; print('fastkan', 'tabm OK')" || { echo "deps missing; run pip install -r requirements_v5.txt"; exit 1; }
python -m pytest tests/test_v5.py tests/test_edges.py -q 2>&1 | tail -10 || { echo "tests FAIL; abort"; exit 1; }

# ---------------------------------------------------------------------------
# Phase 1: dispatch dual-GPU queues
# ---------------------------------------------------------------------------
echo "==[$(date)]== Phase 1: launching dual-GPU dispatchers"
mkdir -p logs/dispatch_gpu0_v5 logs/dispatch_gpu1_v5
nohup bash scripts/dispatch.sh 0 scripts/queue_v5_gpu0.txt > logs/v5_gpu0.log 2>&1 &
PID_GPU0=$!
sleep 2
nohup bash scripts/dispatch.sh 1 scripts/queue_v5_gpu1.txt > logs/v5_gpu1.log 2>&1 &
PID_GPU1=$!
echo "  GPU0 PID=$PID_GPU0  GPU1 PID=$PID_GPU1"

# Wait for both dispatchers
wait $PID_GPU0
RC_GPU0=$?
wait $PID_GPU1
RC_GPU1=$?
echo "==[$(date)]== GPU0 done (rc=$RC_GPU0), GPU1 done (rc=$RC_GPU1)"

# ---------------------------------------------------------------------------
# Phase 2: aggregate + render
# ---------------------------------------------------------------------------
echo "==[$(date)]== Phase 2: aggregate + render"
python -m src.eval --root results --out results/summary.csv | tail -5
python scripts/render_v5_table.py 2>&1 | tail -10

# ---------------------------------------------------------------------------
# Phase 3: print v5 summary block
# ---------------------------------------------------------------------------
echo "==[$(date)]== Phase 3: building results summary"
python - <<'EOF'
import json, glob, re
from pathlib import Path
import numpy as np

# Map (display label, dataset, model, tag) → expected at results/<dataset>/<model>__*__seed{N}_<tag>/
methods = {
    "HIGGS low — PIFT v2 (baseline)":          ("higgs", "pift", "low_level", ""),
    "HIGGS low — PIFT v2 + TabM":              ("higgs", "pift", "low_level", "v5_tabm"),
    "HIGGS low — PIFT v3 NSI (efficient-kan)": ("higgs", "pift", "low_level", ""),  # legacy v3 had no tag
    "HIGGS low — PIFT v3 NSI (Fast-KAN)":      ("higgs", "pift", "low_level", "v5_fastkan"),
    "HIGGS low — PIFT v2+TabM+FastKAN":        ("higgs", "pift", "low_level", "v5_tabm_fastkan"),
    "HIGGS all — PIFT v2 + TabM":              ("higgs", "pift", "all",       "v5_tabm"),
    "HIGGS all — PIFT v3 NSI (Fast-KAN)":      ("higgs", "pift", "all",       "v5_fastkan"),
    "SUSY low — PIFT v2 + TabM":               ("susy",  "pift", "low_level", "v5_tabm"),
    "SUSY all — PIFT v2 + TabM":               ("susy",  "pift", "all",       "v5_tabm"),
    "Top Tagging — PIFT-Edge baseline (v4)":   ("top_tagging", "pift", "all", "v4_edge"),
    "Top Tagging — PIFT-Edge + ChebyKAN":      ("top_tagging", "pift", "all", "v5_edge_cheby"),
    "Top Tagging — PIFT-Edge + TabM":          ("top_tagging", "pift", "all", "v5_edge_tabm"),
    "Top Tagging — PIFT-Edge+ChebyKAN+TabM":   ("top_tagging", "pift", "all", "v5_edge_cheby_tabm"),
    "Top Tagging — PIFT-Subjet + TabM":        ("top_tagging", "pift", "all", "v5_subjet_tabm"),
    "Adult — MLP + TabM":                      ("adult", "mlp", "all", "v5_tabm"),
    "Adult — ResNet + TabM":                   ("adult", "resnet", "all", "v5_tabm"),
    "Forest Cover — MLP + TabM":               ("forest_cover", "mlp", "all", "v5_tabm"),
    "Forest Cover — ResNet + TabM":            ("forest_cover", "resnet", "all", "v5_tabm"),
}

def collect(ds, model, setup, tag):
    aucs, accs, walls = [], [], []
    suffix = f"_{tag}" if tag else ""
    for d in glob.glob(f"results/{ds}/{model}__{setup}__seed*{suffix}"):
        m = re.match(rf"^{model}__{setup}__seed(\d+)(?:_(.+))?$", Path(d).name)
        if not m: continue
        if (m.group(2) or "") != tag: continue
        f = Path(d) / "metrics.json"
        if not f.exists(): continue
        with open(f) as fh: r = json.load(fh)
        aucs.append(r.get("test", {}).get("auc"))
        accs.append(r.get("test", {}).get("acc"))
        walls.append(r.get("wall_clock_s", 0))
    return aucs, accs, walls

print("\n## PIFT v5 summary (mean ± std over seeds)")
print(f"{'Method':52s} | {'AUC':16s} | {'Acc':16s} | wall(s) | n")
print("-" * 110)
for label, (ds, model, setup, tag) in methods.items():
    aucs, accs, walls = collect(ds, model, setup, tag)
    aucs = [a for a in aucs if a is not None]
    accs = [a for a in accs if a is not None]
    if aucs:
        m_auc, s_auc = np.mean(aucs), np.std(aucs, ddof=0)
        auc_s = f"{m_auc:.4f} ± {s_auc:.4f}"
    else:
        auc_s = "—"
    if accs:
        m_acc, s_acc = np.mean(accs), np.std(accs, ddof=0)
        acc_s = f"{m_acc:.4f} ± {s_acc:.4f}"
    else:
        acc_s = "—"
    wall_avg = int(np.mean(walls)) if walls else 0
    n = len(aucs) if aucs else len(accs)
    print(f"{label:52s} | {auc_s:16s} | {acc_s:16s} | {wall_avg:>7d} | {n}")
EOF

echo "==[$(date)]== v5 master DONE"
