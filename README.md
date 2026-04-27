# PIFT: Physics-Informed Feature Tokenization for Tabular Deep Learning

CS 274P (Baldi) 期末 project 主推方案 —— 在 HIGGS / SUSY / HEPMASS 等 HEP 表格 benchmark 上把物理先验嵌入 FT-Transformer。

## 目录结构

```
274P/
├── environment.yml          conda 环境
├── configs/                 数据集 / 实验 YAML
├── src/
│   ├── data/                加载与 feature group
│   ├── models/              MLP / ResNet / FT-T / PIFT / XGB
│   ├── train.py             训练入口
│   ├── eval.py              评测
│   └── utils.py             seeding / metrics / logging
├── scripts/                 数据下载、批量跑 sweep
├── data/                    raw + processed (gitignored)
├── results/                 metric tables / figures
└── logs/                    training logs
```

## 环境（conda）

```bash
conda env create -f environment.yml
conda activate pift
# 按本机 CUDA 调整 torch；2× RTX 3090 + CUDA 13.0
# pip install --index-url https://download.pytorch.org/whl/cu124 torch
```

## 数据下载

```bash
bash scripts/download_data.sh
```

UCI 三件套 (HIGGS / SUSY / HEPMASS) + Forest Cover + Adult。HIGGS 原始 ~7.5GB（gz），下载后按 `configs/*.yaml` 中的 `n_samples` 子采样落盘成 parquet。

## 快速跑一次

```bash
# baseline FT-Transformer，HIGGS low-level
python -m src.train --config configs/higgs.yaml --model ft_transformer --setup low_level --seed 0

# PIFT
python -m src.train --config configs/higgs.yaml --model pift --setup low_level --seed 0

# XGBoost
python -m src.train --config configs/higgs.yaml --model xgboost --setup low_level --seed 0
```

## Headline target

> HIGGS low-level 21 features 上 PIFT AUC ≥ **0.87**（FT-T baseline ~0.85，Baldi 2014 high-level 0.88）。

## 主实验矩阵

| 数据集 | n_samples | low / high / all | seeds |
|---|---|---|---|
| HIGGS | 1M | ✓ ✓ ✓ | 3 |
| SUSY  | 1M | ✓ ✓ ✓ | 3 |
| HEPMASS | 500K | — — ✓ | 3 |
| Forest Cover | 581K | — — ✓ | 3（对照） |
| Adult | 48K | — — ✓ | 3（对照） |

模型：XGBoost / MLP / ResNet / FT-Transformer / **PIFT (ours)**。

## 分工

新手 → 成员 A（数据 + baseline）；老手 1 → PIFT 模块（本仓 [src/models/pift.py](src/models/pift.py)）。

## 当前状态

- [x] 仓库骨架
- [x] PIFT 模块草稿（三子模块 stub）
- [x] HIGGS / SUSY 的 feature group config
- [ ] 数据下载与子采样脚本验证
- [ ] FT-Transformer baseline 复现到 Baldi 2014 ±0.005
- [ ] PIFT 主实验
- [ ] 消融 + learning curve
- [ ] 报告
