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
    p.add_argument("--force", action="store_true",
                   help="Re-run even if metrics.json exists")
    p.add_argument("--pift-no-group", action="store_true",
                   help="Ablation: replace group emb with per-feature linear emb")
    p.add_argument("--pift-untied", action="store_true",
                   help="v1 behaviour: separate MLP per group (no weight tying)")
    p.add_argument("--pift-set-pool", action="store_true",
                   help="Use SetPooledGroupEmbedding (D): same-type particles "
                        "pooled via attention into a single token")
    # PIFT v3: Neural Symbolic Invariant Extractor
    p.add_argument("--pift-nsi", action="store_true",
                   help="PIFT v3: enable Neural Symbolic Invariant Extractor (KAN over 4-momenta)")
    p.add_argument("--nsi-k", type=int, default=16,
                   help="number of learned invariants K")
    p.add_argument("--nsi-lambda-inv", type=float, default=0.5,
                   help="weight on boost-invariance loss")
    p.add_argument("--nsi-beta-max", type=float, default=0.3,
                   help="max longitudinal beta for boost augmentation")
    p.add_argument("--nsi-warmup-epochs", type=int, default=5,
                   help="linear warmup epochs for lambda_inv")
    p.add_argument("--nsi-no-boost-reg", action="store_true",
                   help="ablation: disable boost-invariance regularizer (lambda=0)")
    # PIFT v4: pairwise Lorentz scalar edge tokens
    p.add_argument("--pift-edge", action="store_true",
                   help="PIFT v4: enable pairwise Lorentz scalar edge tokens")
    p.add_argument("--edge-k", type=int, default=32,
                   help="number of edge tokens kept per event (top-K by selection)")
    p.add_argument("--edge-select", default="kt", choices=["kt", "mass", "all"],
                   help="edge selection score: k_T, mass, or no selection")
    # PIFT v5: KAN backend selection + ChebyKAN edge + TabM ensemble
    p.add_argument("--kan-impl", default="efficient",
                   choices=["efficient", "fast", "cheby"],
                   help="v5: KAN backend for v3 NSI (default efficient = v3 behaviour)")
    p.add_argument("--edge-kan", default="none", choices=["none", "cheby"],
                   help="v5: ChebyKAN tokenizer for v4 PIFT-Edge (default none = MLP)")
    p.add_argument("--use-tabm", action="store_true",
                   help="v5: enable TabM-light BatchEnsemble (k members, light variant)")
    p.add_argument("--tabm-k", type=int, default=32,
                   help="TabM ensemble size k (default 32 per Yandex paper)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def build_model(args, cfg, d_in, d_out):
    name = args.model
    use_tabm = getattr(args, "use_tabm", False)
    tabm_k = getattr(args, "tabm_k", 32)
    if name == "mlp":
        return baselines_mod.build_mlp(d_in, d_out, use_tabm=use_tabm, tabm_k=tabm_k)
    if name == "resnet":
        return baselines_mod.build_resnet(d_in, d_out, use_tabm=use_tabm, tabm_k=tabm_k)
    if name == "ft_transformer":
        return baselines_mod.build_ft_transformer(d_in, d_out, use_tabm=use_tabm, tabm_k=tabm_k)
    if name == "pift":
        spec = get_groups(cfg["name"], args.setup)
        if spec is None:
            raise ValueError(
                f"PIFT requires feature groups for {cfg['name']!r} setup={args.setup!r}; "
                f"register one in src/data/feature_groups.py")
        return PIFT(
            spec, d_out=d_out,
            use_group_emb=not args.pift_no_group,
            weight_tied_groups=not args.pift_untied,
            set_pool=args.pift_set_pool,
            use_nsi=args.pift_nsi,
            nsi_K=args.nsi_k,
            nsi_kan_impl=args.kan_impl,
            feature_dim=d_in,
            use_edge_tokens=args.pift_edge,
            edge_k=args.edge_k,
            edge_select=args.edge_select,
            edge_kan=args.edge_kan,
            use_tabm=use_tabm,
            tabm_k=tabm_k,
        )
    raise ValueError(name)


# ---------------------------------------------------------------------------
# DL training loop
# ---------------------------------------------------------------------------
def train_dl(args, cfg, X_train, y_train, X_test, y_test, task: str, out_dir: "Path | None" = None) -> dict:
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

    # PIFT v3: install StandardScaler stats into NSI so it can recover physical (pT, eta, phi)
    if args.model == "pift" and getattr(model, "use_nsi", False):
        model.nsi.set_scaler_stats(
            torch.from_numpy(scaler.mean_).to(device),
            torch.from_numpy(scaler.scale_).to(device),
        )
    # PIFT v4: same for edge tokenizer (needs raw 4-momenta to compute pairwise scalars)
    if args.model == "pift" and getattr(model, "use_edge_tokens", False):
        model.set_edge_scaler_stats(
            torch.from_numpy(scaler.mean_).to(device),
            torch.from_numpy(scaler.scale_).to(device),
        )

    # PIFT v3: KAN params get 3× LR — efficient-kan splines have ~3× smaller gradients
    if args.model == "pift" and getattr(model, "use_nsi", False):
        kan_params, other_params = [], []
        for n, p in model.named_parameters():
            (kan_params if "nsi.kan" in n else other_params).append(p)
        optim = torch.optim.AdamW(
            [{"params": other_params, "lr": cfg["training"]["lr"]},
             {"params": kan_params,   "lr": cfg["training"]["lr"] * 3.0}],
            weight_decay=cfg["training"]["weight_decay"],
        )
        print(f"[train] NSI on, KAN params={sum(p.numel() for p in kan_params)/1e3:.1f}k @ 3x lr")
    else:
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
    use_amp = cfg["training"].get("amp", True) and device.type == "cuda"
    scaler_amp = torch.amp.GradScaler("cuda", enabled=use_amp)

    if task == "binary_classification":
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    use_nsi = (args.model == "pift" and getattr(model, "use_nsi", False))
    use_boost_reg = use_nsi and not args.nsi_no_boost_reg
    best_val, best_state, best_epoch, patience = -1.0, None, 0, 0
    timer = StepTimer()
    grid_updated = False
    for epoch in range(epochs):
        model.train()
        running = 0.0; running_inv = 0.0; nb = 0
        # Linear warmup of lambda_inv from 0 to nsi_lambda_inv over warmup_epochs
        lam_inv = args.nsi_lambda_inv * min(1.0, epoch / max(1, args.nsi_warmup_epochs)) \
                  if use_boost_reg else 0.0
        for step, (xb, yb) in enumerate(train_dl):
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)

            # Refresh KAN spline grids on first epoch every 200 steps
            if use_nsi and not grid_updated and epoch == 0 and step in (0, 200, 400):
                model.nsi.update_kan_grid(xb)
                if step == 400:
                    grid_updated = True

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(xb).squeeze(-1)
                if task == "binary_classification":
                    if logits.dim() == 2:  # TabM: (B, k) — per-member BCE then mean
                        target = yb.float().unsqueeze(1).expand_as(logits)
                        loss_cls = loss_fn(logits, target)
                    else:
                        loss_cls = loss_fn(logits, yb.float())
                else:
                    # Multiclass. Plain: logits (B, n_classes); TabM: (B, k, n_classes).
                    if logits.dim() == 3:
                        B, k, C = logits.shape
                        # Per-member CE then mean: flatten (B, k, C) -> (B*k, C), tile y
                        loss_cls = loss_fn(
                            logits.reshape(B * k, C),
                            yb.unsqueeze(1).expand(B, k).reshape(B * k),
                        )
                    else:
                        loss_cls = loss_fn(logits, yb)

            if use_boost_reg and lam_inv > 0:
                # boost-invariance regularizer in fp32
                with torch.amp.autocast("cuda", enabled=False):
                    beta = torch.empty((), device=device).uniform_(0, args.nsi_beta_max)
                    z_orig  = model.nsi(xb)
                    xb_b    = model.nsi.boost(xb, beta)
                    z_boost = model.nsi(xb_b)
                    denom = z_orig.detach().std(dim=0).mean().clamp_min(1e-3)
                    loss_inv = F.mse_loss(z_orig, z_boost) / denom
                loss = loss_cls + lam_inv * loss_inv.to(loss_cls.dtype)
                running_inv += loss_inv.item()
            else:
                loss = loss_cls

            scaler_amp.scale(loss).backward()
            scaler_amp.step(optim)
            scaler_amp.update()
            running += loss.item(); nb += 1
        sched.step()
        val_metrics = _evaluate(model, val_dl, task, device)
        score = val_metrics.get("auc", val_metrics.get("acc"))
        log_extra = f"  inv={running_inv/nb:.4f} (λ={lam_inv:.2f})" if use_boost_reg else ""
        print(f"  ep{epoch:02d}  loss={running/nb:.4f}{log_extra}  val={val_metrics}  lr={sched.get_last_lr()[0]:.2e}")
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

    # Save checkpoint for PIFT (needed for symbolic readout / boost-invariance analysis)
    if out_dir is not None and args.model == "pift":
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "state_dict": model.state_dict(),
            "scaler_mean": scaler.mean_.astype(np.float32),
            "scaler_scale": scaler.scale_.astype(np.float32),
            "args_cli": vars(args),
            "config_name": cfg["name"],
        }
        torch.save(ckpt, out_dir / "model.pt")

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
            proba = torch.sigmoid(logits)
            if proba.dim() == 2:  # TabM: (B, k) — average prob across ensemble
                proba = proba.mean(dim=1)
            proba = proba.cpu().numpy()
        else:
            if logits.dim() == 3:  # TabM multiclass: (B, k, n_classes)
                logits = logits.mean(dim=1)
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

    tag = f"_{args.tag}" if args.tag else ""
    out = Path("results") / cfg["name"] / f"{args.model}__{args.setup}__seed{args.seed}{tag}"
    if (out / "metrics.json").exists() and not args.force:
        print(f"[skip] {out}/metrics.json exists; pass --force to rerun")
        return

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
        result = train_dl(args, cfg, X_train, y_train, X_test, y_test, task, out_dir=out)

    save_run(out, result)
    print(f"[done] {result}")
    print(f"[done] -> {out}/metrics.json")


if __name__ == "__main__":
    main()
