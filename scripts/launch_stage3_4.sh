#!/usr/bin/env bash
# Wait for both main-matrix dispatchers to finish, then launch learning-curve and ablation queues.
set -uo pipefail
cd "$(dirname "$0")/.."
source /home/lawrence/anaconda3/etc/profile.d/conda.sh
conda activate pift

# Wait until *both* main dispatchers have logged "queue done."
echo "[wait] for both main queues to finish ..."
until grep -q "queue done." logs/dispatch_gpu0/master.log 2>/dev/null \
   && grep -q "queue done." logs/dispatch_gpu1/master.log 2>/dev/null; do
    sleep 60
done
echo "[ok] main queues done."

# Aggregate intermediate results
python -m src.eval --root results --out results/summary.csv || true

# Stage 3 + 4: launch LC + ablation in parallel
echo "[launch] Stage 3+4 dispatchers"
nohup bash scripts/dispatch.sh 0 scripts/queue_gpu0_lc.txt > logs/dispatch_gpu0_lc.master.log 2>&1 &
g0=$!
nohup bash scripts/dispatch.sh 1 scripts/queue_gpu1_lc.txt > logs/dispatch_gpu1_lc.master.log 2>&1 &
g1=$!
echo "  gpu0 PID=$g0  gpu1 PID=$g1"

# Wait for both
wait $g0 $g1
echo "[ok] LC + ablation queues done."

# Stage 4b: permutation invariance check
python scripts/permutation_test.py || true

# Stage 5: aggregate + plots + report
python -m src.eval  --root results --out results/summary.csv
python -m src.plot  --summary results/summary.csv --out-dir results/figures
python scripts/render_report.py
echo "[done] all stages complete; see EXPERIMENT_REPORT.md"
