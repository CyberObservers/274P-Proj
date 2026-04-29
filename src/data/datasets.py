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
    """Adult: ordinal-encode categoricals, parse '?' as NaN -> mode-impute, label '>50K'->1."""
    out = Path(cfg["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    cols = list(range(15))
    df_train = pd.read_csv(cfg["raw_path"],      header=None, names=cols, na_values=" ?", skipinitialspace=True)
    df_test  = pd.read_csv(cfg["raw_path_test"], header=None, names=cols, na_values=" ?", skipinitialspace=True, skiprows=1)
    df_test[14] = df_test[14].astype(str).str.rstrip(".")
    label = lambda v: 1 if str(v).strip() in (">50K", ">50K.") else 0
    cat_cols = cfg.get("categorical_cols", [1, 3, 5, 6, 7, 8, 9, 13])
    for c in cat_cols:
        cats = pd.concat([df_train[c], df_test[c]]).astype(str).fillna("MISSING").unique()
        mapping = {v: i for i, v in enumerate(cats)}
        df_train[c] = df_train[c].astype(str).fillna("MISSING").map(mapping)
        df_test[c]  = df_test[c].astype(str).fillna("MISSING").map(mapping)
    for c in cfg.get("numeric_cols", [0, 2, 4, 10, 11, 12]):
        m = df_train[c].median()
        df_train[c] = df_train[c].fillna(m); df_test[c] = df_test[c].fillna(m)
    df_train[14] = df_train[14].apply(label)
    df_test[14]  = df_test[14].apply(label)
    train = df_train.values.astype(np.float32)
    test  = df_test.values.astype(np.float32)
    np.save(out / "train.npy", train); np.save(out / "test.npy", test)
    print(f"[adult] train {train.shape}  test {test.shape}")


def preprocess_hepmass(cfg: dict) -> None:
    """HEPMASS 1000_train header has 27 names but data has 28 cols; test similarly off by one
    AND has an extra mass column. We bypass pandas header inference."""
    out = Path(cfg["processed_dir"]); out.mkdir(parents=True, exist_ok=True)
    n = cfg["n_samples"]; n_test = cfg["test_split"]
    print(f"[hepmass] reading {cfg['raw_path']} ({n} rows)  +  {cfg['raw_path_test']} ({n_test} rows)")
    train = pd.read_csv(cfg["raw_path"], nrows=n, header=None, skiprows=1, dtype=np.float32, engine="c").values
    test  = pd.read_csv(cfg["raw_path_test"], nrows=n_test, header=None, skiprows=1, dtype=np.float32, engine="c").values
    # Drop trailing mass column from test if shape mismatch (1000_test has +1 col).
    if test.shape[1] == train.shape[1] + 1:
        test = test[:, :-1]
    assert train.shape[1] == test.shape[1], f"train {train.shape} vs test {test.shape}"
    np.save(out / "train.npy", train); np.save(out / "test.npy", test)
    print(f"[hepmass] train {train.shape}  test {test.shape}")


def preprocess_top_tagging(cfg: dict) -> None:
    """Top Tagging Reference (Kasieczka 2019): per-jet 200 constituents + truth.

    For each jet, run anti-kT/C/A clustering to K=8 subjets, output K*8=64 features
    per jet (8 features per subjet: E, px, py, pz, log_n, mass, width, log_pt).
    Saves train.npy / test.npy where col -1 is label (0=QCD, 1=top), prior cols are
    the 64-D flat subjet feature vector.
    """
    import pickle
    import tables
    from src.data.jet_clustering import cluster_subjets_batch

    raw_train = Path(cfg["raw_path_train"])
    raw_test  = Path(cfg["raw_path_test"])
    out = Path(cfg["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    K = cfg.get("n_subjets", 8)
    n_max_const = 200

    def _process(raw_path: Path, label: str) -> np.ndarray:
        f = tables.open_file(str(raw_path), "r")
        try:
            t = f.root.table.table
            N = t.nrows
            print(f"[top_tag] {label}: {N} jets")
            # Stream in chunks of 10K to keep memory bounded
            chunk = 10000
            all_feats: list[np.ndarray] = []
            all_labels: list[np.ndarray] = []
            for start in range(0, N, chunk):
                end = min(start + chunk, N)
                rows = t.read(start, end)
                v0 = rows["values_block_0"]   # (chunk, 804) float32
                v1 = rows["values_block_1"]   # (chunk, 2) int  (ttv, is_signal_new)
                # constituents: first 800 columns reshaped to (chunk, 200, 4)
                constituents = v0[:, : n_max_const * 4].reshape(-1, n_max_const, 4)
                labels = v1[:, 1].astype(np.int64)  # is_signal_new
                # cluster — slow O(seconds per 10K)
                sub = cluster_subjets_batch(constituents, n_subjets=K)   # (chunk, K, 8)
                feats = sub.reshape(-1, K * 8)
                all_feats.append(feats)
                all_labels.append(labels)
                if start % (chunk * 10) == 0:
                    print(f"  [{label}] {end}/{N}")
        finally:
            f.close()
        X = np.concatenate(all_feats, axis=0)
        y = np.concatenate(all_labels, axis=0)
        return np.concatenate([X, y[:, None].astype(np.float32)], axis=1)

    train_arr = _process(raw_train, "train")
    np.save(out / "train.npy", train_arr)
    print(f"[top_tag] saved train.npy {train_arr.shape}")
    test_arr = _process(raw_test, "test")
    np.save(out / "test.npy", test_arr)
    print(f"[top_tag] saved test.npy {test_arr.shape}")


PREPROCESSORS = {
    "higgs": preprocess_higgs_susy,
    "susy":  preprocess_higgs_susy,
    "hepmass": preprocess_hepmass,
    "forest_cover": preprocess_forest_cover,
    "adult": preprocess_adult,
    "top_tagging": preprocess_top_tagging,
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
