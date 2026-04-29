#!/usr/bin/env bash
# Post-hoc analysis pipeline for PIFT v3.  Run AFTER both dispatchers finish.
#   1. Aggregate results/ into summary.csv
#   2. Render LaTeX table (now with PIFT-v3 row)
#   3. Boost-invariance comparison (PIFT v2 tied vs v3) on HIGGS low seed 0
#   4. Symbolic readout on HIGGS all v3 seed 0
set -uo pipefail   # NOT -e: missing v2 checkpoints are expected (pre-edit runs)
source /home/lawrence/anaconda3/etc/profile.d/conda.sh
conda activate pift

echo "[1/4] aggregate metrics → summary.csv"
python -m src.eval --root results --out results/summary.csv

echo "[2/4] re-render main table (LaTeX)"
python scripts/render_table.py

# pick the first available checkpoint across seeds (first run per GPU was launched
# before checkpoint-saving was added to train.py, so seed 0 may lack model.pt)
find_ckpt() {
    local pat="$1"
    for s in 0 1 2; do
        local p="${pat//SEED/$s}"
        if [[ -f "$p" ]]; then
            echo "$p"
            return 0
        fi
    done
    return 0  # always succeed; caller checks for empty string
}
V2_CKPT="$(find_ckpt 'results/higgs/pift__low_level__seedSEED_tied/model.pt')"
[[ -z "$V2_CKPT" ]] && V2_CKPT="$(find_ckpt 'results/higgs/pift__low_level__seedSEED_tied_ckpt/model.pt')"
V3_LOW="$(find_ckpt 'results/higgs/pift__low_level__seedSEED_v3/model.pt')"
V3_ALL="$(find_ckpt 'results/higgs/pift__all__seedSEED_v3/model.pt')"

if [[ -f "$V2_CKPT" && -f "$V3_LOW" ]]; then
    echo "[3/4] boost-invariance comparison (HIGGS low)"
    mkdir -p results/v3_analysis
    python -m scripts.boost_invariance_test --ckpt "$V2_CKPT" \
        --config configs/higgs.yaml --setup low_level \
        --label "PIFT-v2-tied" --out results/v3_analysis/boost_v2_higgs_low.json
    python -m scripts.boost_invariance_test --ckpt "$V3_LOW" \
        --config configs/higgs.yaml --setup low_level \
        --label "PIFT-v3" --out results/v3_analysis/boost_v3_higgs_low.json
else
    echo "[3/4] SKIP — missing checkpoint(s):"
    [[ -f "$V2_CKPT" ]] || echo "    $V2_CKPT (PIFT v2 didn't save model.pt; only future runs do)"
    [[ -f "$V3_LOW" ]] || echo "    $V3_LOW"
fi

if [[ -f "$V3_ALL" ]]; then
    echo "[4/4] symbolic readout on HIGGS all v3"
    python -m scripts.nsi_symbolic_readout --ckpt "$V3_ALL" \
        --config configs/higgs.yaml --setup all \
        --out-dir results/higgs/
else
    echo "[4/4] SKIP — $V3_ALL not yet trained"
fi
echo "[done] all post-hoc analysis"
