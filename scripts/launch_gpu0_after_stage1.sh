#!/usr/bin/env bash
# Wait for Stage 1 (HIGGS FT-T all-features seed 0) to finish, gate on AUC, then launch
# the main GPU 0 queue.
set -uo pipefail
cd "$(dirname "$0")/.."
source /home/lawrence/anaconda3/etc/profile.d/conda.sh
conda activate pift

GATE_FILE="results/higgs/ft_transformer__all__seed0/metrics.json"
echo "[wait] for $GATE_FILE ..."
until [[ -f "$GATE_FILE" ]]; do sleep 30; done

python scripts/check_baseline.py
rc=$?
if [[ $rc -ne 0 ]]; then
    echo "[abort] Stage 1 gate failed (rc=$rc); not launching main GPU 0 queue."
    exit $rc
fi

echo "[launch] GPU 0 main queue"
nohup bash scripts/dispatch.sh 0 scripts/queue_gpu0.txt > logs/dispatch_gpu0.master.log 2>&1 &
echo "[done] dispatcher PID=$!"
