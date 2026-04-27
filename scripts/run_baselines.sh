#!/usr/bin/env bash
# Day 3-5 baseline sweep: XGBoost / MLP / ResNet / FT-T on HIGGS, all 3 setups, 3 seeds
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-configs/higgs.yaml}"
GPU="${PIFT_GPU:-1}"

for setup in low_level high_level all; do
    for model in xgboost mlp resnet ft_transformer; do
        for seed in 0 1 2; do
            tag="${model}_${setup}_s${seed}"
            log="logs/${tag}.log"
            mkdir -p logs
            echo "[run] ${tag}"
            PIFT_GPU=$GPU python -m src.train \
                --config "$CONFIG" --model "$model" --setup "$setup" --seed "$seed" \
                2>&1 | tee "$log"
        done
    done
done
