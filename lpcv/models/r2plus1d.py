"""R(2+1)D-18 model trainer — config and HuggingFace Trainer wrapper.

Provides :class:`R2Plus1DTrainerConfig` (training hyperparameters) and
:class:`R2Plus1DModelTrainer` (handles model loading, layer freezing,
label smoothing, and HuggingFace Trainer integration).

Optimised for few-epoch fine-tuning on large video datasets with a
frozen backbone, unfrozen ``layer4`` + ``fc``, and cosine LR schedule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from loguru import logger
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18
from transformers import Trainer, TrainingArguments

if TYPE_CHECKING:
    from transformers import EvalPrediction

    from lpcv.datasets.base import VideoDataset


@dataclass
class R2Plus1DTrainerConfig:
    """All hyperparameters for an R(2+1)D-18 training run.

    Attributes
    ----------
    num_classes
        Number of output classes (overrides the pretrained head).
        When ``0``, inferred from the dataset.
    num_frames
        Number of frames sampled per video.
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
        - ``"backbone"`` — freeze everything except ``fc``.
        - ``"partial"`` — freeze everything except ``layer4`` and ``fc``.
    extra_args
        Additional keyword arguments forwarded to ``TrainingArguments``.
    """

    num_classes: int = 0
    num_frames: int = 16
    output_dir: str = "output"
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 24
    per_device_eval_batch_size: int = 24
    learning_rate: float = 1e-2
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


class _R2Plus1DOutput:
    """Minimal output wrapper matching the HuggingFace Trainer interface."""

    __slots__ = ("loss", "logits")

    def __init__(self, loss: torch.Tensor | None, logits: torch.Tensor) -> None:
        self.loss = loss
        self.logits = logits


class R2Plus1DForClassification(nn.Module):
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

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> _R2Plus1DOutput:
        """Forward pass.

        Parameters
        ----------
        pixel_values
            Input tensor of shape ``(B, C, T, H, W)``.
        labels
            Ground-truth class indices of shape ``(B,)``.

        Returns
        -------
        _R2Plus1DOutput
            Output with ``.loss`` and ``.logits`` attributes.
        """
        logits = self.backbone(pixel_values)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return _R2Plus1DOutput(loss=loss, logits=logits)

    def save_pretrained(self, path: str | Path) -> None:
        """Save the model state dict and config to a directory.

        Parameters
        ----------
        path
            Directory to save into.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.backbone.state_dict(),
                "num_classes": self.num_classes,
            },
            path / "model.pt",
        )

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


class R2Plus1DModelTrainer:
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

    def __init__(
        self,
        config: R2Plus1DTrainerConfig,
        train_dataset: VideoDataset,
        eval_dataset: VideoDataset,
        val_transform_config: list[dict[str, Any]] | None = None,
    ):
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.val_transform_config = val_transform_config

        label_feature = train_dataset.features.get("label")
        if label_feature is None:
            raise ValueError("Dataset must have a 'label' feature for label metadata")
        self.label_names: list[str] = label_feature.names
        self.num_labels = len(self.label_names)

        num_classes = config.num_classes if config.num_classes > 0 else self.num_labels
        logger.info(
            f"Initializing R2Plus1D-18 trainer: classes={num_classes}, "
            f"epochs={config.num_train_epochs}, freeze={config.freeze_strategy}"
        )

        self.model = R2Plus1DForClassification(
            num_classes=num_classes,
            pretrained=True,
            label_smoothing=config.label_smoothing,
        )

        self._apply_freeze_strategy(config.freeze_strategy)

        if config.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("TF32 enabled for matmul and cuDNN")

        torch.backends.cudnn.benchmark = True

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

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"Freeze strategy '{strategy}': {trainable:,}/{total:,} params trainable "
            f"({trainable / total * 100:.1f}%)"
        )

    def _compute_metrics(self, eval_pred: EvalPrediction) -> dict[str, float]:
        """Compute top-1 and top-5 accuracy from evaluation predictions.

        Parameters
        ----------
        eval_pred
            Predictions and label ids from the Trainer.

        Returns
        -------
        dict[str, float]
            ``{"accuracy": <top1>, "top5_accuracy": <top5>}`` as percentages.
        """
        from lpcv.evaluation import topk_accuracy

        logits_t = torch.as_tensor(eval_pred.predictions)
        labels_t = torch.as_tensor(eval_pred.label_ids)
        acc1, acc5 = topk_accuracy(logits_t, labels_t, topk=(1, 5))
        return {"accuracy": acc1.item(), "top5_accuracy": acc5.item()}

    def _collate_fn(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        """Collate samples into a batch.

        Expects each sample to have ``"pixel_values"`` in ``(T, C, H, W)``
        layout and permutes to ``(C, T, H, W)`` for R(2+1)D's expected
        ``(B, C, T, H, W)`` input.
        """
        pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
        if pixel_values.ndim == 5 and pixel_values.shape[1] != 3:
            pixel_values = pixel_values.permute(0, 2, 1, 3, 4)
        labels = torch.tensor([ex["labels"] for ex in examples], dtype=torch.long)
        return {"pixel_values": pixel_values, "labels": labels}

    def train(self) -> Path:
        """Run the full training loop and save the best model.

        Returns
        -------
        Path
            Path to the saved ``best_model`` directory.
        """
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            eval_strategy=self.config.eval_strategy,
            save_strategy=self.config.save_strategy,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            dataloader_num_workers=self.config.dataloader_num_workers,
            dataloader_pin_memory=self.config.dataloader_pin_memory,
            dataloader_persistent_workers=self.config.dataloader_persistent_workers,
            dataloader_prefetch_factor=self.config.dataloader_prefetch_factor,
            remove_unused_columns=self.config.remove_unused_columns,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            max_steps=self.config.max_steps,
            lr_scheduler_type=self.config.lr_scheduler_type,
            torch_compile=self.config.torch_compile,
            tf32=self.config.tf32,
            report_to="none",
            **self.config.extra_args,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self._compute_metrics,
            data_collator=self._collate_fn,
        )

        logger.info("Starting R2Plus1D-18 training...")
        trainer.train(resume_from_checkpoint=self.config.resume_from_checkpoint)

        output_path = Path(self.config.output_dir) / "best_model"
        self.model.save_pretrained(output_path)

        if self.val_transform_config is not None:
            from lpcv.transforms import save_val_transform_config

            save_val_transform_config(self.val_transform_config, output_path / "val_transform.json")

        logger.info(f"Model saved to {output_path}")
        return output_path
