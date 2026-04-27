"""XGBoost wrapper to expose the same fit/predict_proba API the train.py loop expects."""
from __future__ import annotations

import numpy as np

try:
    import xgboost as xgb
except ImportError:
    xgb = None


def build_xgb(task: str, *, n_estimators: int = 1000, max_depth: int = 8,
              lr: float = 0.05, n_jobs: int = -1, tree_method: str = "hist",
              device: str = "cuda"):
    if xgb is None:
        raise ImportError("xgboost not installed")
    if task == "binary_classification":
        return xgb.XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr,
            objective="binary:logistic", eval_metric="auc",
            tree_method=tree_method, device=device, n_jobs=n_jobs,
        )
    if task == "multiclass_classification":
        return xgb.XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr,
            objective="multi:softprob", eval_metric="mlogloss",
            tree_method=tree_method, device=device, n_jobs=n_jobs,
        )
    raise ValueError(task)
