"""Single entry-point training script.

Examples:
    python -m src.train --config configs/higgs.yaml --model ft_transformer --setup low_level --seed 0
    python -m src.train --config configs/higgs.yaml --model pift           --setup low_level --seed 0
    python -m src.train --config configs/higgs.yaml --model xgboost        --setup low_level --seed 0

The DL path uses AMP + cosine LR + early stop. XGBoost path uses early stopping rounds.
Outputs:
    results/<dataset>/<model>__<setup>__seed<seed>/metrics.json
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.data.datasets import load_split
from src.data.feature_groups import get_groups
from src.models import baselines as baselines_mod
from src.models import xgb as xgb_mod
from src.models.pift import PIFT
from src.utils import StepTimer, compute_metrics, pick_device, save_run, set_seed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--model", required=True,
                   choices=["xgboost", "mlp", "resnet", "ft_transformer", "pift"])
    p.add_argument("--setup", default="all", choices=["all", "low_level", "high_level"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--max-train", type=int, default=None,
                   help="Override n_samples for learning curves")
    p.add_argument("--tag", default="")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def build_model(args, cfg, d_in, d_out):
    name = args.model
    if name == "mlp":
        return baselines_mod.build_mlp(d_in, d_out)
    if name == "resnet":
        return baselines_mod.build_resnet(d_in, d_out)
    if name == "ft_transformer":
        return baselines_mod.build_ft_transformer(d_in, d_out)
    if name == "pift":
        spec = get_groups(cfg["name"], args.setup)
        if spec is None:
            raise ValueError(
                f"PIFT requires feature groups for {cfg['name']!r} setup={args.setup!r}; "
                f"register one in src/data/feature_groups.py")
        return PIFT(spec, d_out=d_out)
    raise ValueError(name)


# ---------------------------------------------------------------------------
# DL training loop
# ---------------------------------------------------------------------------
def train_dl(args, cfg, X_train, y_train, X_test, y_test, task: str) -> dict:
    device = pick_device(args.gpu)
    print(f"[train] device={device}  X_train={X_train.shape}  X_test={X_test.shape}")

    # standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    # train/val split
    rng = np.random.default_rng(args.seed)
    n = X_train.shape[0]
    val_n = max(1024, int(n * cfg.get("val_split", 0.05)))
    perm = rng.permutation(n)
    val_idx, tr_idx = perm[:val_n], perm[val_n:]
    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    val_ds   = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds  = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    bs = cfg["training"]["batch_size"]
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=2,
                          pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds, batch_size=bs * 4, num_workers=2, pin_memory=True)
    test_dl  = DataLoader(test_ds, batch_size=bs * 4, num_workers=2, pin_memory=True)

    n_classes = int(max(y_tr.max(), y_test.max())) + 1 if task == "multiclass_classification" else 1
    d_out = n_classes if task == "multiclass_classification" else 1
    model = build_model(args, cfg, d_in=X_tr.shape[1], d_out=d_out).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model={args.model}  params={n_params/1e6:.2f}M  d_in={X_tr.shape[1]} d_out={d_out}")

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    epochs = cfg["training"]["epochs"]
    warmup = cfg["training"].get("warmup_epochs", 0)

    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / max(1, warmup)
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    scaler_amp = torch.cuda.amp.GradScaler(enabled=cfg["training"].get("amp", True))

    if task == "binary_classification":
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    best_val, best_state, best_epoch, patience = -1.0, None, 0, 0
    timer = StepTimer()
    for epoch in range(epochs):
        model.train()
        running = 0.0; nb = 0
        for xb, yb in train_dl:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg["training"].get("amp", True)):
                logits = model(xb).squeeze(-1)
                if task == "binary_classification":
                    loss = loss_fn(logits, yb.float())
                else:
                    loss = loss_fn(logits, yb)
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optim)
            scaler_amp.update()
            running += loss.item(); nb += 1
        sched.step()
        val_metrics = _evaluate(model, val_dl, task, device)
        score = val_metrics.get("auc", val_metrics.get("acc"))
        print(f"  ep{epoch:02d}  loss={running/nb:.4f}  val={val_metrics}  lr={sched.get_last_lr()[0]:.2e}")
        if score > best_val:
            best_val, best_state, best_epoch, patience = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, epoch, 0
        else:
            patience += 1
            if patience >= cfg["training"].get("early_stop_patience", 10):
                print(f"  [early-stop] no improvement for {patience} epochs")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = _evaluate(model, test_dl, task, device)
    return {
        "model": args.model, "setup": args.setup, "seed": args.seed,
        "params_M": n_params / 1e6,
        "best_val": best_val, "best_epoch": best_epoch,
        "test": test_metrics, "wall_clock_s": timer(),
    }


@torch.no_grad()
def _evaluate(model, loader, task: str, device) -> dict:
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb).squeeze(-1)
        if task == "binary_classification":
            proba = torch.sigmoid(logits).cpu().numpy()
        else:
            proba = F.softmax(logits, dim=-1).cpu().numpy()
        ys.append(yb.numpy()); ps.append(proba)
    return compute_metrics(np.concatenate(ys), np.concatenate(ps), task)


# ---------------------------------------------------------------------------
# XGBoost path
# ---------------------------------------------------------------------------
def train_xgb(args, cfg, X_train, y_train, X_test, y_test, task: str) -> dict:
    timer = StepTimer()
    rng = np.random.default_rng(args.seed)
    n = X_train.shape[0]
    val_n = max(1024, int(n * cfg.get("val_split", 0.05)))
    perm = rng.permutation(n)
    val_idx, tr_idx = perm[:val_n], perm[val_n:]
    model = xgb_mod.build_xgb(task)
    model.fit(
        X_train[tr_idx], y_train[tr_idx],
        eval_set=[(X_train[val_idx], y_train[val_idx])],
        verbose=False,
    )
    proba = model.predict_proba(X_test)
    test_metrics = compute_metrics(y_test, proba, task)
    return {
        "model": "xgboost", "setup": args.setup, "seed": args.seed,
        "test": test_metrics, "wall_clock_s": timer(),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(args.seed)
    X_train, y_train, X_test, y_test = load_split(cfg, setup=args.setup)
    if args.max_train is not None and args.max_train < X_train.shape[0]:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(X_train.shape[0], size=args.max_train, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]

    task = cfg["task"]
    if args.model == "xgboost":
        result = train_xgb(args, cfg, X_train, y_train, X_test, y_test, task)
    else:
        result = train_dl(args, cfg, X_train, y_train, X_test, y_test, task)

    tag = f"_{args.tag}" if args.tag else ""
    out = Path("results") / cfg["name"] / f"{args.model}__{args.setup}__seed{args.seed}{tag}"
    save_run(out, result)
    print(f"[done] {result}")
    print(f"[done] -> {out}/metrics.json")


if __name__ == "__main__":
    main()
