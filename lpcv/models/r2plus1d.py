"""R(2+1)D-18 model trainer — config and HuggingFace Trainer wrapper.

Provides :class:`R2Plus1DTrainerConfig` (training hyperparameters) and
:class:`R2Plus1DModelTrainer` (handles model loading, layer freezing,
label smoothing, and HuggingFace Trainer integration).

Optimised for few-epoch fine-tuning on large video datasets with a
frozen backbone, unfrozen ``layer4`` + ``fc``, and cosine LR schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18

from lpcv.models.base import (
    BaseForClassification,
    BaseModelTrainer,
    BaseTrainerConfig,
    log_freeze_stats,
)


@dataclass
class R2Plus1DTrainerConfig(BaseTrainerConfig):
    """Hyperparameters for an R(2+1)D-18 training run.

    Inherits all common fields from :class:`BaseTrainerConfig` and adds
    R(2+1)D-specific settings.

    Attributes
    ----------
    num_classes
        Number of output classes (overrides the pretrained head).
        When ``0``, inferred from the dataset.
    num_frames
        Number of frames sampled per video.
    label_smoothing
        Label smoothing factor for cross-entropy loss.
    """

    num_classes: int = 0
    num_frames: int = 16
    label_smoothing: float = 0.1
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 24
    per_device_eval_batch_size: int = 24
    learning_rate: float = 1e-2
    freeze_strategy: str = "partial"


class R2Plus1DForClassification(BaseForClassification):
    """R(2+1)D-18 wrapper compatible with HuggingFace Trainer.

    Bridges torchvision's R(2+1)D-18 with the HuggingFace Trainer's
    expected interface (keyword ``pixel_values``, ``labels``, and a
    ``.logits`` attribute on the output).

    Parameters
    ----------
    num_classes
        Number of output classes.
    pretrained
        Load Kinetics-400 pretrained weights.
    label_smoothing
        Label smoothing for cross-entropy loss.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        weights = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
        self.backbone = r2plus1d_18(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        self.num_classes = num_classes
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    @classmethod
    def load_pretrained(cls, path: str | Path) -> R2Plus1DForClassification:
        """Load a saved model from a directory.

        Parameters
        ----------
        path
            Directory containing ``model.pt``.

        Returns
        -------
        R2Plus1DForClassification
            Loaded model in eval mode.
        """
        path = Path(path)
        checkpoint = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        model = cls(num_classes=checkpoint["num_classes"], pretrained=False)
        model.backbone.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model


class R2Plus1DModelTrainer(BaseModelTrainer):
    """High-level wrapper around HuggingFace ``Trainer`` for R(2+1)D-18 fine-tuning.

    Handles model initialisation, parameter freezing, label smoothing,
    custom collation, metrics, and checkpoint saving.

    Parameters
    ----------
    config
        A :class:`R2Plus1DTrainerConfig` with all hyperparameters.
    train_dataset
        Training dataset.  Must expose a ``features`` attribute with a
        ``"label"`` key that has a ``names`` list.
    eval_dataset
        Evaluation dataset with the same schema.

    Raises
    ------
    ValueError
        If the dataset does not have a ``"label"`` feature.
    """

    model_display_name = "R2Plus1D-18"
    model: R2Plus1DForClassification

    def _init_model(self) -> R2Plus1DForClassification:
        num_classes = self.config.num_classes if self.config.num_classes > 0 else self.num_labels
        logger.info(
            f"Initializing R2Plus1D-18 trainer: classes={num_classes}, "
            f"epochs={self.config.num_train_epochs}, freeze={self.config.freeze_strategy}"
        )
        return R2Plus1DForClassification(
            num_classes=num_classes,
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
