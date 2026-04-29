#!/usr/bin/env bash
# Quick status snapshot of all running queues + completed runs.
cd "$(dirname "$0")/.."
echo "=== GPU usage ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo
echo "=== Active dispatcher processes ==="
ps -ef | grep -E "(dispatch.sh|src.train)" | grep -v grep | awk '{print $2, $9, $10, $11, $12, $13}' | head -10
echo
echo "=== Completed runs (metrics.json) ==="
find results -name metrics.json 2>/dev/null | wc -l
echo
echo "=== Per-dataset run count ==="
for ds in higgs susy hepmass forest_cover adult; do
    n=$(find "results/$ds" -name metrics.json 2>/dev/null | wc -l)
    echo "  $ds: $n"
done
echo
echo "=== Recent run_log entries ==="
[[ -f results/run_log.csv ]] && tail -10 results/run_log.csv
echo
echo "=== Failed runs ==="
[[ -f results/failed.txt ]] && tail -5 results/failed.txt || echo "(none)"
