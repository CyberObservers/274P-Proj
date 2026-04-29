#!/usr/bin/env bash
# Download all UCI datasets used in PIFT.
# Run from repo root: bash scripts/download_data.sh
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p data/raw

fetch() {
    local url="$1"; local out="$2"
    if [[ -f "$out" ]]; then
        echo "[skip] $out exists"
        return
    fi
    echo "[get ] $url -> $out"
    curl -L --fail -o "$out" "$url"
}

# HIGGS (~2.6 GB gz)
fetch "https://archive.ics.uci.edu/static/public/280/higgs.zip"        "data/raw/higgs.zip"
# SUSY (~900 MB gz)
fetch "https://archive.ics.uci.edu/static/public/279/susy.zip"         "data/raw/susy.zip"
# HEPMASS (~3 GB gz)
fetch "https://archive.ics.uci.edu/static/public/347/hepmass.zip"      "data/raw/hepmass.zip"
# Forest Cover
fetch "https://archive.ics.uci.edu/static/public/31/covertype.zip"     "data/raw/covertype.zip"
# Adult
fetch "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data" "data/raw/adult.data"
fetch "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test" "data/raw/adult.test"

# Top Quark Tagging Reference (Kasieczka 2019, ~1.7GB total) — for v4 PIFT-Edge / PIFT-Subjet
mkdir -p data/raw/top_tagging
fetch "https://zenodo.org/records/2603256/files/train.h5" "data/raw/top_tagging/train.h5"
fetch "https://zenodo.org/records/2603256/files/test.h5"  "data/raw/top_tagging/test.h5"
fetch "https://zenodo.org/records/2603256/files/val.h5"   "data/raw/top_tagging/val.h5"

echo "[unzip] expanding archives"
cd data/raw
for f in higgs.zip susy.zip hepmass.zip covertype.zip; do
    [[ -f "$f" ]] && unzip -n "$f"
done
cd ../..

echo "[next ] python -m src.data.datasets --preprocess --config configs/higgs.yaml"
