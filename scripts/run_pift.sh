#!/usr/bin/env bash
# Main PIFT experiment: HIGGS + SUSY low-level (and all-features for sanity)
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${PIFT_GPU:-1}"

for cfg in configs/higgs.yaml configs/susy.yaml; do
    for setup in low_level all; do
        for seed in 0 1 2; do
            tag="pift_${setup}_s${seed}_$(basename "$cfg" .yaml)"
            log="logs/${tag}.log"
            mkdir -p logs
            echo "[run] ${tag}"
            PIFT_GPU=$GPU python -m src.train \
                --config "$cfg" --model pift --setup "$setup" --seed "$seed" \
                2>&1 | tee "$log"
        done
    done
done
