#!/usr/bin/env bash
# Data efficiency: HIGGS low-level, sweep training sizes, FT-T vs PIFT.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${PIFT_GPU:-1}"
CONFIG=configs/higgs.yaml

for n in 10000 50000 100000 500000 1000000; do
    for model in ft_transformer pift; do
        for seed in 0 1 2; do
            tag="lc_${model}_n${n}_s${seed}"
            log="logs/${tag}.log"
            mkdir -p logs
            echo "[run] ${tag}"
            PIFT_GPU=$GPU python -m src.train \
                --config "$CONFIG" --model "$model" --setup low_level \
                --max-train "$n" --seed "$seed" --tag "n${n}" \
                2>&1 | tee "$log"
        done
    done
done
