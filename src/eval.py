"""Aggregate metrics.json across runs into a result table.

Usage:
    python -m src.eval --root results/higgs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results")
    p.add_argument("--out",  default="results/summary.csv")
    args = p.parse_args()

    rows = []
    for f in Path(args.root).rglob("metrics.json"):
        with open(f) as fh:
            r = json.load(fh)
        flat = {
            "dataset": f.parent.parent.name,
            "model": r.get("model"),
            "setup": r.get("setup"),
            "seed": r.get("seed"),
            "params_M": r.get("params_M"),
            "best_val": r.get("best_val"),
            "best_epoch": r.get("best_epoch"),
            "wall_clock_s": r.get("wall_clock_s"),
        }
        flat.update({f"test_{k}": v for k, v in (r.get("test") or {}).items()})
        rows.append(flat)
    df = pd.DataFrame(rows).sort_values(["dataset", "setup", "model", "seed"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))
    print(f"\n[done] -> {args.out}")


if __name__ == "__main__":
    main()
