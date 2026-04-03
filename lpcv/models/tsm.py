"""TSM (Temporal Shift Module) model trainer — standalone, no mmaction2.

Implements the Temporal Shift Module from `TSM: Temporal Shift Module for
Efficient Video Understanding <https://arxiv.org/abs/1811.08383>`_.

Uses torchvision's ``resnet18`` / ``resnet50`` as the 2D backbone with
temporal shift inserted into selected residual blocks.  All operations are
standard Conv2d — no Conv3d ops, making it efficient on Qualcomm NPU.

The temporal shift uses ``torch.roll`` instead of slice+pad+concat to
minimise reshape and data-movement ops that are expensive on NPU.

Provides :class:`TSMTrainerConfig`, :class:`TSMForClassification`, and
:class:`TSMModelTrainer` following the same patterns as the R(2+1)D
implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from loguru import logger
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50

from lpcv.models.base import (
    BaseForClassification,
    BaseModelTrainer,
    BaseTrainerConfig,
    ModelOutput,
    log_freeze_stats,
)

TSM_BACKBONES: dict[str, dict[str, Any]] = {
    "resnet18": {
        "factory": resnet18,
        "weights": ResNet18_Weights.IMAGENET1K_V1,
        "fc_in_features": 512,
    },
    "resnet50": {
        "factory": resnet50,
        "weights": ResNet50_Weights.IMAGENET1K_V2,
        "fc_in_features": 2048,
    },
}
"""Supported 2D backbones for TSM."""

ALL_LAYER_NAMES: tuple[str, ...] = ("layer1", "layer2", "layer3", "layer4")
"""Ordered ResNet layer group names."""

DEFAULT_SHIFT_LAST_N: int = 2
"""Default: inject temporal shift into the last 2 layer groups (layer3, layer4)."""


def _resolve_shift_layers(shift_last_n: int) -> tuple[str, ...]:
    """Return the last *shift_last_n* ResNet layer names.

    Parameters
    ----------
    shift_last_n
        Number of layer groups (counted from the end) to inject temporal
        shift into.  ``0`` disables shift; ``4`` shifts all layers.

    Returns
    -------
    tuple[str, ...]
        Layer names to inject temporal shift into.
    """
    n = max(0, min(shift_last_n, len(ALL_LAYER_NAMES)))
    if n == 0:
        return ()
    return ALL_LAYER_NAMES[-n:]


def temporal_shift(x: torch.Tensor, num_segments: int, shift_div: int = 8) -> torch.Tensor:
    """Perform temporal shift on feature maps using ``torch.roll``.

    Shifts 1/shift_div of channels forward in time and 1/shift_div backward
    using ``torch.roll``, leaving the rest unchanged.  This wraps boundary
    frames rather than zero-padding, producing a single efficient op on NPU.

    Parameters
    ----------
    x
        Input tensor of shape ``(B*T, C, H, W)``.
    num_segments
        Number of temporal segments ``T``.
    shift_div
        Fraction of channels to shift (default 8 → shift 1/8 forward, 1/8 back).

    Returns
    -------
    torch.Tensor
        Shifted tensor with the same shape as input.
    """
    bt, c, h, w = x.shape
    b = bt // num_segments
    x = x.view(b, num_segments, c, h, w)

    fold = c // shift_div

    out = x.clone()
    out[:, :, :fold] = torch.roll(x[:, :, :fold], shifts=1, dims=1)
    out[:, :, fold : 2 * fold] = torch.roll(x[:, :, fold : 2 * fold], shifts=-1, dims=1)

    return out.view(bt, c, h, w)


class TemporalShiftWrapper(nn.Module):
    """Wraps a residual block to apply temporal shift before the first conv.

    Parameters
    ----------
    block
        A residual block (e.g. ``BasicBlock`` or ``Bottleneck``).
    num_segments
        Number of temporal segments.
    shift_div
        Fraction of channels to shift.
    """

    def __init__(self, block: nn.Module, num_segments: int, shift_div: int = 8) -> None:
        super().__init__()
        self.block = block
        self.num_segments = num_segments
        self.shift_div = shift_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply temporal shift then the original block."""
        x = temporal_shift(x, self.num_segments, self.shift_div)
        return self.block(x)


@dataclass
class TSMTrainerConfig(BaseTrainerConfig):
    """Hyperparameters for a TSM training run.

    Inherits all common fields from :class:`BaseTrainerConfig` and adds
    TSM-specific settings.

    Attributes
    ----------
    backbone
        Backbone name: ``"resnet18"`` or ``"resnet50"``.
    num_classes
        Number of output classes.  When ``0``, inferred from the dataset.
    num_frames
        Number of temporal segments sampled per video.
    shift_div
        Fraction of channels to shift (default 8).
    shift_last_n
        Number of layer groups (from the end) to inject temporal shift into.
        ``0`` disables shift; ``4`` shifts all layers.  Default ``2`` shifts
        ``layer3`` and ``layer4``.
    label_smoothing
        Label smoothing factor for cross-entropy loss.
    """

    backbone: str = "resnet50"
    num_classes: int = 0
    num_frames: int = 8
    shift_div: int = 8
    shift_last_n: int = DEFAULT_SHIFT_LAST_N
    label_smoothing: float = 0.1
    learning_rate: float = 1e-2
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    freeze_strategy: str = "partial"
    lr_scheduler_type: str = "cosine"


class TSMForClassification(BaseForClassification):
    """TSM wrapper compatible with HuggingFace Trainer.

    Bridges a torchvision ResNet (with temporal shift) with the HuggingFace
    Trainer's expected interface (keyword ``pixel_values``, ``labels``, and
    a ``.logits`` attribute on the output).

    The model expects input of shape ``(B, C, T, H, W)`` and internally
    reshapes to ``(B*T, C, H, W)`` for the 2D backbone, then averages
    temporal logits to produce the final prediction.

    Parameters
    ----------
    num_classes
        Number of output classes.
    backbone_name
        Key in :data:`TSM_BACKBONES` (``"resnet18"`` or ``"resnet50"``).
    num_frames
        Number of temporal segments.
    shift_div
        Fraction of channels to shift.
    shift_last_n
        Number of layer groups (from the end) to inject temporal shift into.
    pretrained
        Load ImageNet pretrained weights for the backbone.
    label_smoothing
        Label smoothing for cross-entropy loss.
    """

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "resnet50",
        num_frames: int = 8,
        shift_div: int = 8,
        shift_last_n: int = DEFAULT_SHIFT_LAST_N,
        pretrained: bool = True,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if backbone_name not in TSM_BACKBONES:
            available = ", ".join(sorted(TSM_BACKBONES))
            raise ValueError(f"Unknown TSM backbone {backbone_name!r}. Available: {available}")

        spec = TSM_BACKBONES[backbone_name]
        weights = spec["weights"] if pretrained else None
        self.backbone = spec["factory"](weights=weights)
        self.backbone.fc = nn.Linear(spec["fc_in_features"], num_classes)

        shift_layers = _resolve_shift_layers(shift_last_n)
        self._inject_temporal_shift(
            num_segments=num_frames, shift_div=shift_div, shift_layers=shift_layers
        )

        self.num_classes = num_classes
        self.backbone_name = backbone_name
        self.num_frames = num_frames
        self.shift_div = shift_div
        self.shift_last_n = shift_last_n
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def _inject_temporal_shift(
        self,
        num_segments: int,
        shift_div: int = 8,
        shift_layers: tuple[str, ...] = (),
    ) -> None:
        """Inject temporal shift into selected residual blocks of the backbone.

        Wraps each block in the specified layers with
        :class:`TemporalShiftWrapper`.

        Parameters
        ----------
        num_segments
            Number of temporal segments.
        shift_div
            Fraction of channels to shift.
        shift_layers
            Layer names to inject temporal shift into.
        """
        for layer_name in shift_layers:
            layer = getattr(self.backbone, layer_name, None)
            if layer is None:
                continue
            blocks = list(layer.children())
            wrapped = [TemporalShiftWrapper(b, num_segments, shift_div) for b in blocks]
            setattr(self.backbone, layer_name, nn.Sequential(*wrapped))

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> ModelOutput:
        """Forward pass.

        Parameters
        ----------
        pixel_values
            Input tensor of shape ``(B, C, T, H, W)``.
        labels
            Ground-truth class indices of shape ``(B,)``.

        Returns
        -------
        ModelOutput
            Output with ``.loss`` and ``.logits`` attributes.
        """
        b, c, t, h, w = pixel_values.shape
        x = pixel_values.permute(0, 2, 1, 3, 4).contiguous().reshape(b * t, c, h, w)

        logits = self.backbone(x)
        logits = logits.view(b, t, -1).mean(dim=1)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return ModelOutput(loss=loss, logits=logits)

    def _extra_save_meta(self) -> dict[str, Any]:
        return {
            "backbone_name": self.backbone_name,
            "num_frames": self.num_frames,
            "shift_div": self.shift_div,
            "shift_last_n": self.shift_last_n,
        }

    @classmethod
    def load_pretrained(cls, path: str | Path) -> TSMForClassification:
        """Load a saved model from a directory.

        Parameters
        ----------
        path
            Directory containing ``model.pt``.

        Returns
        -------
        TSMForClassification
            Loaded model in eval mode.
        """
        path = Path(path)
        checkpoint = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        model = cls(
            num_classes=checkpoint["num_classes"],
            backbone_name=checkpoint.get("backbone_name", "resnet50"),
            num_frames=checkpoint.get("num_frames", 8),
            shift_div=checkpoint.get("shift_div", 8),
            shift_last_n=checkpoint.get("shift_last_n", DEFAULT_SHIFT_LAST_N),
            pretrained=False,
        )
        model.backbone.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model


class TSMModelTrainer(BaseModelTrainer):
    """High-level wrapper around HuggingFace ``Trainer`` for TSM fine-tuning.

    Parameters
    ----------
    config
        A :class:`TSMTrainerConfig` with all hyperparameters.
    train_dataset
        Training dataset.
    eval_dataset
        Evaluation dataset.
    val_transform_config
        Validation transform config to save alongside the model.
    """

    model_display_name = "TSM"
    model: TSMForClassification

    def _init_model(self) -> TSMForClassification:
        num_classes = self.config.num_classes if self.config.num_classes > 0 else self.num_labels
        shift_layers = _resolve_shift_layers(self.config.shift_last_n)
        logger.info(
            f"Initializing TSM ({self.config.backbone}) trainer: classes={num_classes}, "
            f"frames={self.config.num_frames}, shift_div={self.config.shift_div}, "
            f"shift_last_n={self.config.shift_last_n} ({shift_layers}), "
            f"epochs={self.config.num_train_epochs}, freeze={self.config.freeze_strategy}"
        )
        return TSMForClassification(
            num_classes=num_classes,
            backbone_name=self.config.backbone,
            num_frames=self.config.num_frames,
            shift_div=self.config.shift_div,
            shift_last_n=self.config.shift_last_n,
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
            - ``"backbone"`` — freeze everything except ``fc``.
            - ``"partial"`` — freeze everything except ``layer4`` and ``fc``.
        """
        if strategy == "none":
            return

        if strategy == "backbone":
            for name, param in self.model.backbone.named_parameters():
                if not name.startswith("fc"):
                    param.requires_grad = False
        elif strategy == "partial":
            for name, param in self.model.backbone.named_parameters():
                if not (name.startswith("layer4") or name.startswith("fc")):
                    param.requires_grad = False
        else:
            logger.warning(f"Unknown freeze strategy '{strategy}', skipping")
            return

        log_freeze_stats(self.model, strategy)
