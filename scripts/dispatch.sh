#!/usr/bin/env bash
# Serial-per-GPU dispatcher.
# Usage:  bash scripts/dispatch.sh <gpu_id> <queue.txt>
# Each line of <queue.txt> is a python invocation. Lines starting with '#' or empty are skipped.
# Already-completed runs (results/.../metrics.json present) are skipped via train.py --force absence.
set -uo pipefail

# Activate conda env so 'python' resolves with torch / rtdl / xgboost.
source /home/lawrence/anaconda3/etc/profile.d/conda.sh
conda activate pift

GPU="${1:?need gpu id}"
QUEUE="${2:?need queue file}"
LOG_DIR="logs/dispatch_gpu${GPU}"
mkdir -p "$LOG_DIR" results
RUN_LOG="results/run_log.csv"
[[ -f "$RUN_LOG" ]] || echo "ts,gpu,line_no,exit,wall_s,cmd" > "$RUN_LOG"

i=0
while IFS= read -r raw || [[ -n "$raw" ]]; do
    i=$((i+1))
    cmd="${raw%%#*}"
    cmd="$(echo -n "$cmd" | sed -e 's/[[:space:]]*$//' -e 's/^[[:space:]]*//')"
    [[ -z "$cmd" ]] && continue
    line_log="$LOG_DIR/line_${i}.log"
    echo "[gpu${GPU}|${i}] $cmd" | tee -a "$LOG_DIR/master.log"
    t0=$(date +%s)
    CUDA_VISIBLE_DEVICES="$GPU" PIFT_GPU=0 bash -c "$cmd" >>"$line_log" 2>&1
    rc=$?
    t1=$(date +%s); dt=$((t1-t0))
    ts=$(date -u +%FT%TZ)
    safe_cmd=$(echo "$cmd" | tr ',' ';' | tr '"' "'")
    echo "${ts},${GPU},${i},${rc},${dt},\"${safe_cmd}\"" >> "$RUN_LOG"
    if [[ $rc -ne 0 ]]; then
        echo "  [FAIL rc=$rc dt=${dt}s] -> $line_log" | tee -a "$LOG_DIR/master.log"
        echo "${ts}|gpu${GPU}|line${i}|rc${rc}|${cmd}" >> results/failed.txt
    else
        echo "  [ok   dt=${dt}s]" | tee -a "$LOG_DIR/master.log"
    fi
done < "$QUEUE"

echo "[gpu${GPU}] queue done." | tee -a "$LOG_DIR/master.log"
