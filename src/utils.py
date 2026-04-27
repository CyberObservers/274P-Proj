"""Common utilities: seeding, device selection, metric computation."""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(prefer_gpu: int | None = None) -> torch.device:
    """Choose CUDA device. Defaults to GPU 1 (user occupies GPU 0 on the dev box)."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if prefer_gpu is None:
        prefer_gpu = int(os.environ.get("PIFT_GPU", "1"))
    prefer_gpu = min(prefer_gpu, torch.cuda.device_count() - 1)
    return torch.device(f"cuda:{prefer_gpu}")


def compute_metrics(y_true: np.ndarray, proba: np.ndarray, task: str) -> dict:
    if task == "binary_classification":
        if proba.ndim == 2:
            proba = proba[:, 1]
        return {
            "auc": float(roc_auc_score(y_true, proba)),
            "acc": float(accuracy_score(y_true, (proba > 0.5).astype(int))),
        }
    if task == "multiclass_classification":
        pred = proba.argmax(axis=1)
        return {"acc": float(accuracy_score(y_true, pred))}
    raise ValueError(task)


class StepTimer:
    def __init__(self): self.t0 = time.perf_counter()
    def __call__(self) -> float: return time.perf_counter() - self.t0


def save_run(out_dir: Path, payload: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2)
