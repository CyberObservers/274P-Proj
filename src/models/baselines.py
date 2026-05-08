"""DL baselines: MLP / ResNet / FT-Transformer.

Wrappers around rtdl_revisiting_models 0.0.2.

  - `MLP(d_in, d_out, n_blocks, d_block, dropout)`
  - `ResNet(d_in, d_out, n_blocks, d_block, d_hidden_multiplier, dropout1, dropout2)`
  - `FTTransformer(n_cont_features, cat_cardinalities, **backbone_kwargs)`,
     where `backbone_kwargs` come from `FTTransformer.get_default_kwargs(n_blocks)`.

v5: optional TabM-light wrapping via `_TabMHead` — replaces the final Linear
head with `LinearBatchEnsemble(d_in, d_out, k)`, broadcasting the penultimate
hidden through `EnsembleView`.  Used on Adult / Forest Cover (non-physics
tabular) where PIFT's group spec doesn't apply.
"""
from __future__ import annotations

import torch
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


class _TabMHead(nn.Module):
    """Wrap a backbone whose forward returns (B, d_out): replace head with a
    BatchEnsemble of k members.  Forward returns (B, k, d_out).

    Strategy for course-grade simplicity: replace the model's final Linear
    `head` (or `output_layer`) with a `LinearBatchEnsemble` driven from the
    penultimate features.  We rely on the rtdl backbones' standard structure:
    most expose `.head` (MLP/ResNet) or have a final transformer projection
    (FT-T)."""

    def __init__(self, backbone: nn.Module, d_pen: int, d_out: int, k: int):
        super().__init__()
        from tabm import LinearBatchEnsemble, EnsembleView
        self.backbone = backbone
        self.k = k
        self.ensemble_view = EnsembleView(k=k)
        self.tabm_head = LinearBatchEnsemble(
            d_pen, d_out, k=k, scaling_init="random-signs",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Strip backbone's final head, get penultimate features
        h_pen = self._extract_penultimate(x)            # (B, d_pen)
        h = self.ensemble_view(h_pen)                   # (B, k, d_pen)
        out = self.tabm_head(h)                         # (B, k, d_out)
        return out

    def _extract_penultimate(self, x: torch.Tensor) -> torch.Tensor:
        # Subclass overrides this for backbone-specific surgery
        raise NotImplementedError


class _MLPTabM(_TabMHead):
    def _extract_penultimate(self, x: torch.Tensor) -> torch.Tensor:
        # rtdl MLP: blocks → output (Linear). Manually run blocks; skip output.
        h = x
        for blk in self.backbone.blocks:
            h = blk(h)
        return h


class _ResNetTabM(_TabMHead):
    def _extract_penultimate(self, x: torch.Tensor) -> torch.Tensor:
        # rtdl 0.0.2 ResNet: input_projection → blocks → output (Sequential).
        # `output` is Sequential(Norm, Linear) — we strip the final Linear and
        # keep its preceding norm by running output[:-1] (the norm).
        h = self.backbone.input_projection(x)
        for blk in self.backbone.blocks:
            h = blk(h)
        # Apply the pre-head normalization but skip the final Linear projection
        for layer in list(self.backbone.output)[:-1]:
            h = layer(h)
        return h


def build_mlp(d_in: int, d_out: int, *, n_blocks: int = 3, d_block: int = 256,
              dropout: float = 0.1, use_tabm: bool = False, tabm_k: int = 32) -> nn.Module:
    _need_rtdl()
    backbone = _RTDLMLP(d_in=d_in, d_out=d_out, n_blocks=n_blocks,
                        d_block=d_block, dropout=dropout)
    if not use_tabm:
        return backbone
    return _MLPTabM(backbone, d_pen=d_block, d_out=d_out, k=tabm_k)


def build_resnet(d_in: int, d_out: int, *, n_blocks: int = 3, d_block: int = 256,
                 d_hidden_multiplier: float = 2.0, dropout1: float = 0.25,
                 dropout2: float = 0.0, use_tabm: bool = False,
                 tabm_k: int = 32) -> nn.Module:
    _need_rtdl()
    backbone = _RTDLResNet(d_in=d_in, d_out=d_out, n_blocks=n_blocks,
                           d_block=d_block, d_hidden_multiplier=d_hidden_multiplier,
                           dropout1=dropout1, dropout2=dropout2)
    if not use_tabm:
        return backbone
    return _ResNetTabM(backbone, d_pen=d_block, d_out=d_out, k=tabm_k)


class _FTTAdapter(nn.Module):
    """rtdl FTTransformer.forward(x_cont, x_cat) -> uniform forward(x)."""

    def __init__(self, ftt: nn.Module):
        super().__init__()
        self.ftt = ftt

    def forward(self, x):
        return self.ftt(x, None)


def build_ft_transformer(n_cont_features: int, d_out: int, *, n_blocks: int = 3,
                         use_tabm: bool = False, tabm_k: int = 32) -> nn.Module:
    """Use rtdl 0.0.2 default kwargs.

    Note: TabM head wrap is NOT supported for rtdl FT-Transformer in this
    version because rtdl's internal API doesn't expose a stable penultimate
    hook (no `feature_tokenizer` attribute in 0.0.2).  Use MLP/ResNet+TabM on
    non-physics tabular instead, or use PIFT+TabM on physics tabular.
    """
    if use_tabm:
        raise NotImplementedError(
            "ft_transformer + TabM is not implemented; use mlp/resnet + TabM "
            "for non-physics tabular ablation, or pift + TabM for physics tabular."
        )
    _need_rtdl()
    backbone_kwargs = _RTDLFTT.get_default_kwargs(n_blocks=n_blocks)
    backbone_kwargs["d_out"] = d_out
    ftt = _RTDLFTT(
        n_cont_features=n_cont_features,
        cat_cardinalities=[],
        **backbone_kwargs,
    )
    return _FTTAdapter(ftt)
