#!/usr/bin/env bash
# Master orchestrator for v4 PIFT-Edge / PIFT-Subjet experiments.
#  1. Wait for preprocess to finish (test.npy appears)
#  2. Quick smoke train PIFT-Subjet 1 epoch
#  3. Dispatch dual-GPU queues (~27 runs, 3-4h)
#  4. Aggregate metrics + render Top Tagging table
#  5. Append summary block to EXPERIMENT_REPORT.md §10.4
set -uo pipefail

cd /home/lawrence/274P
source /home/lawrence/anaconda3/etc/profile.d/conda.sh
conda activate pift

LOG=logs/v4_master.log
exec > >(tee -a "$LOG") 2>&1

echo "==[$(date)]== v4 master start"

# ---------------------------------------------------------------------------
# Phase 1: wait for preprocess test.npy
# ---------------------------------------------------------------------------
echo "==[$(date)]== Phase 1: waiting for preprocess test.npy"
while [[ ! -f data/processed/top_tagging/test.npy ]]; do
    sleep 30
done
sleep 5
ls -lh data/processed/top_tagging/
echo "==[$(date)]== preprocess complete"

# ---------------------------------------------------------------------------
# Phase 2: smoke test PIFT-Subjet (1 epoch on first 50K jets)
# ---------------------------------------------------------------------------
echo "==[$(date)]== Phase 2: smoke test PIFT-Subjet"
CUDA_VISIBLE_DEVICES=0 python -m src.train --config configs/top_tagging.yaml \
    --model pift --setup all --seed 0 --max-train 50000 --tag v4_smoke --force 2>&1 | tail -30
SMOKE_RC=${PIPESTATUS[0]}
if [[ $SMOKE_RC -ne 0 ]]; then
    echo "==[$(date)]== smoke test FAILED (rc=$SMOKE_RC).  ABORT v4 dispatch."
    exit 1
fi
echo "==[$(date)]== smoke test PASSED"

# ---------------------------------------------------------------------------
# Phase 3: dispatch dual-GPU queues
# ---------------------------------------------------------------------------
echo "==[$(date)]== Phase 3: launching dual-GPU dispatchers"
mkdir -p logs/dispatch_gpu0_v4 logs/dispatch_gpu1_v4
nohup bash scripts/dispatch.sh 0 scripts/queue_v4_gpu0.txt > logs/v4_gpu0.log 2>&1 &
PID_GPU0=$!
sleep 2
nohup bash scripts/dispatch.sh 1 scripts/queue_v4_gpu1.txt > logs/v4_gpu1.log 2>&1 &
PID_GPU1=$!
echo "  GPU0 PID=$PID_GPU0  GPU1 PID=$PID_GPU1"

# Wait for both dispatchers
wait $PID_GPU0
RC_GPU0=$?
wait $PID_GPU1
RC_GPU1=$?
echo "==[$(date)]== GPU0 done (rc=$RC_GPU0), GPU1 done (rc=$RC_GPU1)"

# ---------------------------------------------------------------------------
# Phase 4: aggregate + render
# ---------------------------------------------------------------------------
echo "==[$(date)]== Phase 4: aggregate + render"
python -m src.eval --root results --out results/summary.csv | tail -5
python scripts/render_top_tagging_table.py
python scripts/render_table.py 2>&1 | tail -5

# ---------------------------------------------------------------------------
# Phase 5: print v4 summary block to a file (we will edit EXPERIMENT_REPORT manually)
# ---------------------------------------------------------------------------
echo "==[$(date)]== Phase 5: building results summary"
python - <<'EOF'
import json, glob, re
from pathlib import Path
import numpy as np

# Pull v4 / v4_subjet / v4_edge / v4_subjet_edge / v4_nsi rows for top_tagging
methods = {
    "XGBoost":        ("top_tagging", "xgboost",         ""),
    "MLP":            ("top_tagging", "mlp",              ""),
    "ResNet":         ("top_tagging", "resnet",          ""),
    "FT-Transformer": ("top_tagging", "ft_transformer",  ""),
    "PIFT-Subjet":    ("top_tagging", "pift",            "v4_subjet"),
    "PIFT-NSI":       ("top_tagging", "pift",            "v4_nsi"),
    "PIFT-Edge":      ("top_tagging", "pift",            "v4_edge"),
    "PIFT-Subjet+Edge": ("top_tagging", "pift",          "v4_subjet_edge"),
}
def collect(ds, model, tag):
    aucs = []
    for d in glob.glob(f"results/{ds}/{model}__*"):
        name = Path(d).name
        # parse out tag (anything after last _seedN_)
        m = re.match(r"^.+__seed(\d+)(?:_(.+))?$", name)
        if not m: continue
        if (m.group(2) or "") != tag: continue
        f = Path(d) / "metrics.json"
        if not f.exists(): continue
        with open(f) as fh: r = json.load(fh)
        aucs.append(r["test"]["auc"])
    return aucs

print("\n## Top Tagging results (mean ± std over seeds)")
print(f"{'Method':30s} | {'AUC':16s} | n_seeds")
print("-" * 60)
for label, (ds, model, tag) in methods.items():
    aucs = collect(ds, model, tag)
    if aucs:
        m = np.mean(aucs); s = np.std(aucs, ddof=0)
        print(f"{label:30s} | {m:.4f} ± {s:.4f}  | {len(aucs)}")
    else:
        print(f"{label:30s} | (no runs)        | 0")

# HIGGS sanity check for PIFT-Edge
print("\n## HIGGS low PIFT-Edge sanity check")
aucs_higgs = collect("higgs", "pift", "v4_edge")
if aucs_higgs:
    m = np.mean(aucs_higgs); s = np.std(aucs_higgs, ddof=0)
    print(f"PIFT-Edge HIGGS low: {m:.4f} ± {s:.4f} ({len(aucs_higgs)} seeds)")
EOF

echo "==[$(date)]== v4 master DONE"
