"""MViTv2-S model trainer — config and HuggingFace Trainer wrapper.

Provides :class:`MViTv2TrainerConfig` (training hyperparameters) and
:class:`MViTv2ModelTrainer` (handles model loading, layer freezing,
label smoothing, and HuggingFace Trainer integration).

MViTv2-S (Multiscale Vision Transformer v2, Small variant) is loaded
from ``torchvision.models.video``.  It uses pooling attention instead
of 3D convolutions for temporal modelling, achieving strong accuracy
with comparable FLOPs to R(2+1)D-18.

Supports arbitrary spatial sizes via relative-position-embedding
interpolation.  The default competition pipeline produces 112×112
inputs; the pretrained 224×224 Kinetics-400 weights are adapted by
bicubic interpolation of the ``rel_pos_h`` / ``rel_pos_w`` tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torchvision.models.video.mvit import MSBlockConfig, MViT, MViT_V2_S_Weights

from lpcv.models.base import (
    BaseForClassification,
    BaseModelTrainer,
    BaseTrainerConfig,
    log_freeze_stats,
)

_BLOCK_SETTING_CONFIG: dict[str, list[Any]] = {
    "num_heads": [1, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8],
    "input_channels": [
        96,
        96,
        192,
        192,
        384,
        384,
        384,
        384,
        384,
        384,
        384,
        384,
        384,
        384,
        384,
        768,
    ],
    "output_channels": [
        96,
        192,
        192,
        384,
        384,
        384,
        384,
        384,
        384,
        384,
        384,
        384,
        384,
        384,
        768,
        768,
    ],
    "kernel_q": [[3, 3, 3]] * 16,
    "kernel_kv": [[3, 3, 3]] * 16,
    "stride_q": (
        [[1, 1, 1], [1, 2, 2], [1, 1, 1], [1, 2, 2]] + [[1, 1, 1]] * 10 + [[1, 2, 2], [1, 1, 1]]
    ),
    "stride_kv": (
        [[1, 8, 8], [1, 4, 4], [1, 4, 4], [1, 2, 2]] + [[1, 2, 2]] * 10 + [[1, 1, 1], [1, 1, 1]]
    ),
}
"""MViTv2-S block configuration matching ``torchvision.models.video.mvit_v2_s``."""


def _build_block_setting() -> list[MSBlockConfig]:
    """Build the MViTv2-S block setting list."""
    cfg = _BLOCK_SETTING_CONFIG
    return [
        MSBlockConfig(
            num_heads=cfg["num_heads"][i],
            input_channels=cfg["input_channels"][i],
            output_channels=cfg["output_channels"][i],
            kernel_q=cfg["kernel_q"][i],
            kernel_kv=cfg["kernel_kv"][i],
            stride_q=cfg["stride_q"][i],
            stride_kv=cfg["stride_kv"][i],
        )
        for i in range(16)
    ]


def _interpolate_rel_pos(
    pretrained_sd: dict[str, torch.Tensor],
    target_sd: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Adapt pretrained weights to a different spatial resolution.

    Relative positional embedding tables (``rel_pos_h``, ``rel_pos_w``)
    are resized via bilinear interpolation when the target model has a
    different spatial size.  All other parameters are copied as-is.

    Parameters
    ----------
    pretrained_sd
        State dict from the pretrained (224×224) model.
    target_sd
        State dict from the target-resolution model (used to determine
        expected tensor shapes).

    Returns
    -------
    dict[str, torch.Tensor]
        Adapted state dict ready for ``load_state_dict``.
    """
    new_sd: dict[str, torch.Tensor] = {}
    for k, v in pretrained_sd.items():
        if k in target_sd and target_sd[k].shape != v.shape and "rel_pos" in k:
            new_len = target_sd[k].shape[0]
            v_interp = (
                F.interpolate(
                    v.float().unsqueeze(0).unsqueeze(0),
                    size=(new_len, v.shape[1]),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .squeeze(0)
            )
            new_sd[k] = v_interp
        else:
            new_sd[k] = v
    return new_sd


def _build_mvitv2(
    num_classes: int,
    spatial_size: int = 112,
    temporal_size: int = 16,
    pretrained: bool = True,
) -> MViT:
    """Build an MViTv2-S model at an arbitrary spatial resolution.

    When *pretrained* is ``True`` and *spatial_size* differs from 224,
    the Kinetics-400 weights are loaded and the relative positional
    embedding tables are interpolated to match the target resolution.

    Parameters
    ----------
    num_classes
        Number of output classes.
    spatial_size
        Spatial input size (square).
    temporal_size
        Number of temporal frames.
    pretrained
        Load Kinetics-400 pretrained weights.

    Returns
    -------
    MViT
        Model instance ready for fine-tuning.
    """
    block_setting = _build_block_setting()
    model = MViT(
        spatial_size=(spatial_size, spatial_size),
        temporal_size=temporal_size,
        block_setting=block_setting,
        residual_pool=True,
        residual_with_cls_embed=False,
        rel_pos_embed=True,
        proj_after_attn=True,
        stochastic_depth_prob=0.2,
        num_classes=num_classes,
    )

    if pretrained:
        pretrained_sd: dict[str, torch.Tensor] = dict(
            MViT_V2_S_Weights.KINETICS400_V1.get_state_dict(progress=True, check_hash=True)
        )
        if spatial_size != 224:
            pretrained_sd = _interpolate_rel_pos(pretrained_sd, model.state_dict())
            logger.info(
                f"Interpolated rel_pos embeddings from 224×224 to {spatial_size}×{spatial_size}"
            )
        head_key = "head.1.weight"
        bias_key = "head.1.bias"
        if pretrained_sd[head_key].shape[0] != num_classes:
            del pretrained_sd[head_key]
            del pretrained_sd[bias_key]
            model.load_state_dict(pretrained_sd, strict=False)
        else:
            model.load_state_dict(pretrained_sd)
    return model


@dataclass
class MViTv2TrainerConfig(BaseTrainerConfig):
    """Hyperparameters for an MViTv2-S training run.

    Inherits all common fields from :class:`BaseTrainerConfig` and adds
    MViTv2-specific settings.

    Attributes
    ----------
    num_classes
        Number of output classes.  When ``0``, inferred from the dataset.
    num_frames
        Number of frames sampled per video.
    crop_size
        Spatial crop size (square).
    label_smoothing
        Label smoothing factor for cross-entropy loss.
    """

    num_classes: int = 0
    num_frames: int = 16
    crop_size: int = 112
    label_smoothing: float = 0.1
    learning_rate: float = 1e-4
    freeze_strategy: str = "partial"


class MViTv2ForClassification(BaseForClassification):
    """MViTv2-S wrapper compatible with HuggingFace Trainer.

    Bridges torchvision's MViTv2-S with the HuggingFace Trainer's
    expected interface (keyword ``pixel_values``, ``labels``, and a
    ``.logits`` attribute on the output).

    Parameters
    ----------
    num_classes
        Number of output classes.
    crop_size
        Spatial input size (square).
    pretrained
        Load Kinetics-400 pretrained weights (with rel_pos interpolation
        when *crop_size* differs from 224).
    label_smoothing
        Label smoothing for cross-entropy loss.
    """

    def __init__(
        self,
        num_classes: int,
        crop_size: int = 112,
        pretrained: bool = True,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone: nn.Module = _build_mvitv2(
            num_classes=num_classes,
            spatial_size=crop_size,
            pretrained=pretrained,
        )
        self.num_classes = num_classes
        self.crop_size = crop_size
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def _extra_save_meta(self) -> dict[str, Any]:
        return {"crop_size": self.crop_size}

    @classmethod
    def load_pretrained(cls, path: str | Path) -> MViTv2ForClassification:
        """Load a saved model from a directory.

        Parameters
        ----------
        path
            Directory containing ``model.pt``.

        Returns
        -------
        MViTv2ForClassification
            Loaded model in eval mode.
        """
        path = Path(path)
        checkpoint = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        model = cls(
            num_classes=checkpoint["num_classes"],
            crop_size=checkpoint.get("crop_size", 112),
            pretrained=False,
        )
        model.backbone.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model


class MViTv2ModelTrainer(BaseModelTrainer):
    """High-level wrapper around HuggingFace ``Trainer`` for MViTv2-S fine-tuning.

    Parameters
    ----------
    config
        A :class:`MViTv2TrainerConfig` with all hyperparameters.
    train_dataset
        Training dataset.
    eval_dataset
        Evaluation dataset.
    val_transform_config
        Validation transform config to save alongside the model.
    """

    model_display_name = "MViTv2-S"
    model: MViTv2ForClassification

    def _init_model(self) -> MViTv2ForClassification:
        num_classes = self.config.num_classes if self.config.num_classes > 0 else self.num_labels
        logger.info(
            f"Initializing MViTv2-S trainer: classes={num_classes}, "
            f"crop={self.config.crop_size}, "
            f"epochs={self.config.num_train_epochs}, freeze={self.config.freeze_strategy}"
        )
        return MViTv2ForClassification(
            num_classes=num_classes,
            crop_size=self.config.crop_size,
            pretrained=True,
            label_smoothing=self.config.label_smoothing,
        )

    def _apply_freeze_strategy(self, strategy: str) -> None:
        """Freeze model parameters according to *strategy*.

        Parameters
        ----------
        strategy
            One of:

            - ``"none"`` — all parameters trainable.
            - ``"backbone"`` — freeze everything except the classification
              head (``head``).
            - ``"partial"`` — freeze everything except the last 4 transformer
              blocks (``blocks.12``–``blocks.15``), the final norm, and the head.
        """
        if strategy == "none":
            return

        if strategy == "backbone":
            for name, param in self.model.backbone.named_parameters():
                if not name.startswith("head"):
                    param.requires_grad = False
        elif strategy == "partial":
            trainable_prefixes = (
                "blocks.12",
                "blocks.13",
                "blocks.14",
                "blocks.15",
                "head",
                "norm",
            )
            for name, param in self.model.backbone.named_parameters():
                if not any(name.startswith(p) for p in trainable_prefixes):
                    param.requires_grad = False
        else:
            logger.warning(f"Unknown freeze strategy '{strategy}', skipping")
            return

        log_freeze_stats(self.model, strategy)
