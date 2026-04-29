"""DL baselines: MLP / ResNet / FT-Transformer.

Wrappers around rtdl_revisiting_models 0.0.2.

  - `MLP(d_in, d_out, n_blocks, d_block, dropout)`
  - `ResNet(d_in, d_out, n_blocks, d_block, d_hidden_multiplier, dropout1, dropout2)`
  - `FTTransformer(n_cont_features, cat_cardinalities, **backbone_kwargs)`,
     where `backbone_kwargs` come from `FTTransformer.get_default_kwargs(n_blocks)`.
"""
from __future__ import annotations

import torch.nn as nn

try:
    from rtdl_revisiting_models import MLP as _RTDLMLP
    from rtdl_revisiting_models import ResNet as _RTDLResNet
    from rtdl_revisiting_models import FTTransformer as _RTDLFTT
    _HAS_RTDL = True
except ImportError:
    _HAS_RTDL = False


def _need_rtdl() -> None:
    if not _HAS_RTDL:
        raise ImportError("rtdl_revisiting_models not installed. pip install rtdl-revisiting-models")


def build_mlp(d_in: int, d_out: int, *, n_blocks: int = 3, d_block: int = 256,
              dropout: float = 0.1) -> nn.Module:
    _need_rtdl()
    return _RTDLMLP(d_in=d_in, d_out=d_out, n_blocks=n_blocks,
                    d_block=d_block, dropout=dropout)


def build_resnet(d_in: int, d_out: int, *, n_blocks: int = 3, d_block: int = 256,
                 d_hidden_multiplier: float = 2.0, dropout1: float = 0.25,
                 dropout2: float = 0.0) -> nn.Module:
    _need_rtdl()
    return _RTDLResNet(d_in=d_in, d_out=d_out, n_blocks=n_blocks,
                       d_block=d_block, d_hidden_multiplier=d_hidden_multiplier,
                       dropout1=dropout1, dropout2=dropout2)


class _FTTAdapter(nn.Module):
    """rtdl FTTransformer.forward(x_cont, x_cat) -> uniform forward(x)."""

    def __init__(self, ftt: nn.Module):
        super().__init__()
        self.ftt = ftt

    def forward(self, x):
        return self.ftt(x, None)


def build_ft_transformer(n_cont_features: int, d_out: int, *, n_blocks: int = 3) -> nn.Module:
    """Use rtdl 0.0.2 default kwargs."""
    _need_rtdl()
    backbone_kwargs = _RTDLFTT.get_default_kwargs(n_blocks=n_blocks)
    backbone_kwargs["d_out"] = d_out
    ftt = _RTDLFTT(
        n_cont_features=n_cont_features,
        cat_cardinalities=[],
        **backbone_kwargs,
    )
    return _FTTAdapter(ftt)
