"""Load + preprocess UCI HEP datasets into parquet.

Usage:
    python -m src.data.datasets --preprocess --config configs/higgs.yaml
    python -m src.data.datasets --preprocess --config configs/susy.yaml
"""
from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _read_csv_gz(path: Path, n_rows: int | None = None) -> np.ndarray:
    """HIGGS / SUSY are headerless CSV.gz with ~7-8GB raw size. We stream them."""
    df = pd.read_csv(path, header=None, nrows=n_rows, dtype=np.float32, engine="c")
    return df.values


def preprocess_higgs_susy(cfg: dict) -> None:
    """HIGGS / SUSY share the same layout: col 0 = label, then features."""
    raw = Path(cfg["raw_path"])
    out = Path(cfg["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)

    n = cfg["n_samples"]
    n_test = cfg["test_split"]
    print(f"[{cfg['name']}] reading {raw} ({n + n_test} rows)")
    arr = _read_csv_gz(raw, n_rows=n + n_test)

    # Baldi 2014 protocol: last n_test rows = test set
    test = arr[-n_test:]
    train = arr[:-n_test]

    # subsample train if needed
    if train.shape[0] > n:
        rng = np.random.default_rng(0)
        idx = rng.choice(train.shape[0], size=n, replace=False)
        train = train[idx]

    np.save(out / "train.npy", train)
    np.save(out / "test.npy",  test)
    print(f"[{cfg['name']}] train {train.shape}  test {test.shape}  -> {out}")


def preprocess_forest_cover(cfg: dict) -> None:
    raw = Path(cfg["raw_path"])
    out = Path(cfg["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(raw, header=None)
    arr = df.values.astype(np.float32)
    # label is last column 1..7 -> 0..6
    arr[:, -1] -= 1
    rng = np.random.default_rng(0)
    rng.shuffle(arr)
    n_test = int(arr.shape[0] * cfg["test_split"])
    test, train = arr[:n_test], arr[n_test:]
    np.save(out / "train.npy", train)
    np.save(out / "test.npy",  test)
    print(f"[{cfg['name']}] train {train.shape}  test {test.shape}")


def preprocess_adult(cfg: dict) -> None:
    raise NotImplementedError("Adult requires categorical encoding — TODO member A")


PREPROCESSORS = {
    "higgs": preprocess_higgs_susy,
    "susy":  preprocess_higgs_susy,
    "forest_cover": preprocess_forest_cover,
    "adult": preprocess_adult,
}


def slice_features(x: np.ndarray, cfg: dict, setup: str) -> np.ndarray:
    """Apply low_level / high_level / all setup based on config."""
    n_low, n_high = cfg.get("n_low", 0), cfg.get("n_high", 0)
    if setup == "all" or n_low == 0:
        return x
    if setup == "low_level":
        return x[:, :n_low]
    if setup == "high_level":
        return x[:, n_low:n_low + n_high]
    raise ValueError(f"unknown setup {setup!r}")


def load_split(cfg: dict, setup: str = "all"):
    """Return (X_train, y_train, X_test, y_test) numpy arrays."""
    out = Path(cfg["processed_dir"])
    train = np.load(out / "train.npy")
    test  = np.load(out / "test.npy")
    target_col = cfg["target_col"]
    if target_col == -1:
        X_train, y_train = train[:, :-1], train[:, -1].astype(np.int64)
        X_test,  y_test  = test[:, :-1],  test[:, -1].astype(np.int64)
    else:
        X_train, y_train = train[:, target_col + 1:], train[:, target_col].astype(np.int64)
        X_test,  y_test  = test[:, target_col + 1:],  test[:, target_col].astype(np.int64)
    X_train = slice_features(X_train, cfg, setup)
    X_test  = slice_features(X_test,  cfg, setup)
    return X_train, y_train, X_test, y_test


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--preprocess", action="store_true")
    args = p.parse_args()
    cfg = _load_config(args.config)
    if args.preprocess:
        PREPROCESSORS[cfg["name"]](cfg)


if __name__ == "__main__":
    main()
