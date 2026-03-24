"""X3D model trainer — config and HuggingFace Trainer wrapper.

Provides :class:`X3DTrainerConfig` (training hyperparameters) and
:class:`X3DModelTrainer` (handles model loading, layer freezing,
label smoothing, and HuggingFace Trainer integration).

X3D models are loaded from ``facebookresearch/pytorchvideo`` via
``torch.hub``.  Available presets: ``x3d_xs``, ``x3d_s``, ``x3d_m``,
``x3d_l``.

Each preset has a default spatial crop size and temporal frame count
that matches the architecture's design; these can be overridden via
the config or CLI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from loguru import logger

from lpcv.models.base import BaseForClassification, BaseModelTrainer, log_freeze_stats

X3D_PRESET_DEFAULTS: dict[str, dict[str, int]] = {
    "x3d_xs": {"num_frames": 4, "crop_size": 182},
    "x3d_s": {"num_frames": 13, "crop_size": 182},
    "x3d_m": {"num_frames": 16, "crop_size": 256},
    "x3d_l": {"num_frames": 16, "crop_size": 312},
}
"""Default num_frames and crop_size for each X3D variant from pytorchvideo."""


@dataclass
class X3DTrainerConfig:
    """All hyperparameters for an X3D training run.

    Attributes
    ----------
    preset
        X3D variant: ``"x3d_xs"``, ``"x3d_s"``, ``"x3d_m"``, or ``"x3d_l"``.
    num_classes
        Number of output classes.  When ``0``, inferred from the dataset.
    num_frames
        Number of frames sampled per video.  When ``0``, uses the preset default.
    crop_size
        Spatial crop size.  When ``0``, uses the preset default.
    output_dir
        Directory for checkpoints and the final saved model.
    num_train_epochs
        Total training epochs.
    per_device_train_batch_size
        Batch size per GPU during training.
    per_device_eval_batch_size
        Batch size per GPU during evaluation.
    learning_rate
        Peak learning rate for the optimiser.
    warmup_ratio
        Fraction of total steps used for linear warmup.
    weight_decay
        L2 regularisation coefficient.
    label_smoothing
        Label smoothing factor for cross-entropy loss.
    logging_steps
        Log metrics every N optimiser steps.
    eval_strategy
        When to evaluate: ``"epoch"``, ``"steps"``, or ``"no"``.
    save_strategy
        When to save checkpoints: ``"epoch"``, ``"steps"``, or ``"no"``.
    save_total_limit
        Maximum number of checkpoints to keep on disk.
    load_best_model_at_end
        Whether to reload the best checkpoint after training.
    metric_for_best_model
        Metric name used for best-model selection.
    fp16
        Enable FP16 mixed precision.
    bf16
        Enable BF16 mixed precision.
    dataloader_num_workers
        Number of data-loading worker processes.
    dataloader_pin_memory
        Pin memory for faster host-to-device transfer.
    dataloader_persistent_workers
        Keep workers alive between epochs.
    dataloader_prefetch_factor
        Number of batches to prefetch per worker.
    remove_unused_columns
        Whether the Trainer should drop columns not used by the model.
    resume_from_checkpoint
        Path to a checkpoint directory to resume from.
    gradient_accumulation_steps
        Number of forward passes before an optimiser step.
    max_steps
        Stop after N optimiser steps (overrides *num_train_epochs* when > 0).
    lr_scheduler_type
        Learning rate scheduler type.
    torch_compile
        Use ``torch.compile`` for fused kernels.
    tf32
        Enable TF32 math on Ampere+ GPUs.
    freeze_strategy
        Parameter freeze strategy: ``"none"``, ``"backbone"``, or ``"partial"``.

        - ``"none"`` — all parameters trainable.
        - ``"backbone"`` — freeze everything except the final projection head.
        - ``"partial"`` — freeze everything except the last two blocks and head.
    extra_args
        Additional keyword arguments forwarded to ``TrainingArguments``.
    """

    preset: str = "x3d_m"
    num_classes: int = 0
    num_frames: int = 0
    crop_size: int = 0
    output_dir: str = "output"
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    learning_rate: float = 5e-3
    warmup_ratio: float = 0.1
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    logging_steps: int = 10
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "accuracy"
    fp16: bool = False
    bf16: bool = False
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = False
    dataloader_prefetch_factor: int | None = None
    remove_unused_columns: bool = False
    resume_from_checkpoint: str | None = None
    gradient_accumulation_steps: int = 1
    max_steps: int = -1
    lr_scheduler_type: str = "cosine"
    torch_compile: bool = False
    tf32: bool = False
    freeze_strategy: str = "partial"
    extra_args: dict[str, Any] = field(default_factory=dict)

    def resolved_num_frames(self) -> int:
        """Return num_frames, falling back to the preset default."""
        if self.num_frames > 0:
            return self.num_frames
        return X3D_PRESET_DEFAULTS[self.preset]["num_frames"]

    def resolved_crop_size(self) -> int:
        """Return crop_size, falling back to the preset default."""
        if self.crop_size > 0:
            return self.crop_size
        return X3D_PRESET_DEFAULTS[self.preset]["crop_size"]


class X3DForClassification(BaseForClassification):
    """X3D wrapper compatible with HuggingFace Trainer.

    Bridges pytorchvideo's X3D with the HuggingFace Trainer's expected
    interface (keyword ``pixel_values``, ``labels``, and a ``.logits``
    attribute on the output).

    Parameters
    ----------
    num_classes
        Number of output classes.
    preset
        X3D variant name (``"x3d_xs"``, ``"x3d_s"``, ``"x3d_m"``, ``"x3d_l"``).
    pretrained
        Load Kinetics-400 pretrained weights.
    label_smoothing
        Label smoothing for cross-entropy loss.
    """

    def __init__(
        self,
        num_classes: int,
        preset: str = "x3d_m",
        pretrained: bool = True,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if preset not in X3D_PRESET_DEFAULTS:
            available = ", ".join(sorted(X3D_PRESET_DEFAULTS))
            raise ValueError(f"Unknown X3D preset {preset!r}. Available: {available}")

        backbone: nn.Module = torch.hub.load(  # pyright: ignore[reportAssignmentType]
            "facebookresearch/pytorchvideo", preset, pretrained=pretrained
        )
        self.backbone = backbone

        head_block: Any = backbone.blocks[5]  # type: ignore[index]
        in_features: int = head_block.proj.in_features
        head_block.proj = nn.Linear(in_features, num_classes)
        head_block.activation = None

        self.num_classes = num_classes
        self.preset = preset
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def _extra_save_meta(self) -> dict[str, Any]:
        return {"preset": self.preset}

    @classmethod
    def load_pretrained(cls, path: str | Path) -> X3DForClassification:
        """Load a saved model from a directory.

        Parameters
        ----------
        path
            Directory containing ``model.pt``.

        Returns
        -------
        X3DForClassification
            Loaded model in eval mode.
        """
        path = Path(path)
        checkpoint = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        model = cls(
            num_classes=checkpoint["num_classes"],
            preset=checkpoint["preset"],
            pretrained=False,
        )
        model.backbone.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model


class X3DModelTrainer(BaseModelTrainer):
    """High-level wrapper around HuggingFace ``Trainer`` for X3D fine-tuning.

    Parameters
    ----------
    config
        A :class:`X3DTrainerConfig` with all hyperparameters.
    train_dataset
        Training dataset.
    eval_dataset
        Evaluation dataset.
    val_transform_config
        Validation transform config to save alongside the model.
    """

    model_display_name = "X3D"

    def _init_model(self) -> X3DForClassification:
        num_classes = self.config.num_classes if self.config.num_classes > 0 else self.num_labels
        logger.info(
            f"Initializing X3D ({self.config.preset}) trainer: classes={num_classes}, "
            f"epochs={self.config.num_train_epochs}, freeze={self.config.freeze_strategy}"
        )
        return X3DForClassification(
            num_classes=num_classes,
            preset=self.config.preset,
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
            - ``"backbone"`` — freeze everything except the final projection
              head (``blocks.5``).
            - ``"partial"`` — freeze everything except the last two blocks
              (``blocks.4`` and ``blocks.5``).
        """
        if strategy == "none":
            return

        if strategy == "backbone":
            for name, param in self.model.backbone.named_parameters():
                if not name.startswith("blocks.5"):
                    param.requires_grad = False
        elif strategy == "partial":
            for name, param in self.model.backbone.named_parameters():
                if not (name.startswith("blocks.4") or name.startswith("blocks.5")):
                    param.requires_grad = False
        else:
            logger.warning(f"Unknown freeze strategy '{strategy}', skipping")
            return

        log_freeze_stats(self.model, strategy)
