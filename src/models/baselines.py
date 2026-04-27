"""DL baselines: MLP / ResNet / FT-Transformer.

Thin wrappers around `rtdl_revisiting_models` so member A / B can drop them in.
The package exposes `MLP`, `ResNet`, `FTTransformer` with from_default factories.
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    from rtdl_revisiting_models import MLP as _RTDLMLP
    from rtdl_revisiting_models import ResNet as _RTDLResNet
    from rtdl_revisiting_models import FTTransformer as _RTDLFTT
    _HAS_RTDL = True
except ImportError:  # rtdl not installed yet
    _HAS_RTDL = False


def _need_rtdl() -> None:
    if not _HAS_RTDL:
        raise ImportError(
            "rtdl_revisiting_models not installed. "
            "Run: pip install rtdl-revisiting-models"
        )


def build_mlp(d_in: int, d_out: int, *, n_blocks: int = 3, d_hidden: int = 256,
              dropout: float = 0.1) -> nn.Module:
    _need_rtdl()
    return _RTDLMLP(
        d_in=d_in,
        d_out=d_out,
        n_blocks=n_blocks,
        d_block=d_hidden,
        dropout=dropout,
    )


def build_resnet(d_in: int, d_out: int, *, n_blocks: int = 3, d_main: int = 256,
                 d_hidden_factor: float = 2.0, dropout1: float = 0.25,
                 dropout2: float = 0.0) -> nn.Module:
    _need_rtdl()
    return _RTDLResNet(
        d_in=d_in,
        d_out=d_out,
        n_blocks=n_blocks,
        d_block=d_main,
        d_hidden_multiplier=d_hidden_factor,
        dropout1=dropout1,
        dropout2=dropout2,
    )


def build_ft_transformer(n_num_features: int, d_out: int, *, n_blocks: int = 3,
                         d_token: int = 128, attention_dropout: float = 0.1,
                         ffn_dropout: float = 0.1) -> nn.Module:
    _need_rtdl()
    return _RTDLFTT.make_baseline(
        n_num_features=n_num_features,
        cat_cardinalities=None,
        d_out=d_out,
        n_blocks=n_blocks,
        d_block=d_token,
        attention_n_heads=8,
        attention_dropout=attention_dropout,
        ffn_d_hidden_multiplier=4 / 3,
        ffn_dropout=ffn_dropout,
        residual_dropout=0.0,
    )
