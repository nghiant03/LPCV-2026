"""VideoMAE model trainer — config and HuggingFace Trainer wrapper.

Provides :class:`VideoMAETrainerConfig` (training hyperparameters) and
:class:`VideoMAEModelTrainer` (handles model loading, freeze strategies,
custom collation, metrics, and checkpoint saving).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger
from transformers import Trainer, VideoMAEForVideoClassification

from lpcv.models.base import (
    BaseModelTrainer,
    BaseTrainerConfig,
    collate_for_video,
    log_freeze_stats,
)

if TYPE_CHECKING:
    from pathlib import Path

    import torch


@dataclass
class VideoMAETrainerConfig(BaseTrainerConfig):
    """Hyperparameters for a VideoMAE training run.

    Inherits all common fields from :class:`BaseTrainerConfig` and adds
    VideoMAE-specific settings.

    Attributes
    ----------
    model_name
        HuggingFace model identifier or local path.
    num_frames
        Number of frames sampled per video.
    """

    model_name: str = "MCG-NJU/videomae-base"
    num_frames: int = 16
    num_train_epochs: int = 15
    learning_rate: float = 5e-5
    weight_decay: float = 0.05
    lr_scheduler_type: str = "linear"


class VideoMAEModelTrainer(BaseModelTrainer):
    """HuggingFace ``Trainer`` wrapper for VideoMAE fine-tuning.

    Parameters
    ----------
    config
        A :class:`VideoMAETrainerConfig` with all hyperparameters.
    train_dataset
        Training dataset.  Must expose a ``features`` attribute with a
        ``"label"`` key that has a ``names`` list (HuggingFace ``ClassLabel``
        or the :class:`~lpcv.datasets.base.DatasetFeatures` shim).
    eval_dataset
        Evaluation dataset with the same schema.
    val_transform_config
        Validation transform config to save alongside the model.

    Raises
    ------
    ValueError
        If the dataset does not have a ``"label"`` feature.
    """

    model_display_name = "VideoMAE"
    model: VideoMAEForVideoClassification

    def _init_model(self) -> VideoMAEForVideoClassification:  # type: ignore[override]
        label2id = {name: i for i, name in enumerate(self.label_names)}
        id2label = {i: name for i, name in enumerate(self.label_names)}

        logger.info(
            f"Initializing VideoMAE trainer: model={self.config.model_name}, "
            f"labels={self.num_labels}, epochs={self.config.num_train_epochs}"
        )

        return VideoMAEForVideoClassification.from_pretrained(
            self.config.model_name,
            num_labels=self.num_labels,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,
        )

    def _apply_freeze_strategy(self, strategy: str) -> None:
        """Freeze model parameters according to *strategy*.

        Parameters
        ----------
        strategy
            One of:

            - ``"none"`` — all parameters trainable.
            - ``"backbone"`` — freeze everything except the classifier head.
            - ``"partial"`` — freeze all except the last 2 encoder layers and
              the classifier head.
        """
        if strategy == "none":
            return

        if strategy == "backbone":
            for name, param in self.model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False
        elif strategy == "partial":
            num_layers = len(self.model.videomae.encoder.layer)
            freeze_until = num_layers - 2
            for name, param in self.model.named_parameters():
                if "classifier" in name:
                    continue
                if "encoder.layer." in name:
                    layer_idx = int(name.split("encoder.layer.")[1].split(".")[0])
                    if layer_idx < freeze_until:
                        param.requires_grad = False
                elif "videomae." in name:
                    param.requires_grad = False
        else:
            logger.warning(f"Unknown freeze strategy '{strategy}', skipping")
            return

        log_freeze_stats(self.model, strategy)

    @staticmethod
    def _collate_fn(examples: list[dict]) -> dict[str, torch.Tensor]:
        """Collate samples without layout permutation (VideoMAE expects BTCHW)."""
        return collate_for_video(examples, permute_to_cthw=False)

    def _save_model(self, trainer: Trainer, path: Path) -> None:
        trainer.save_model(str(path))
