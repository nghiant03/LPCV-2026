"""Shared model components — output wrapper, metrics, collation, base classes.

Provides building blocks reused across model implementations:

- :class:`ModelOutput` — lightweight output wrapper for HuggingFace Trainer.
- :func:`compute_metrics` — top-1 / top-5 accuracy computation.
- :func:`collate_for_video` — video batch collation with optional layout permutation.
- :class:`BaseForClassification` — ``nn.Module`` base for custom classification wrappers.
- :class:`BaseModelTrainer` — shared HuggingFace Trainer integration for non-HF models.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from loguru import logger
from transformers import Trainer, TrainingArguments

if TYPE_CHECKING:
    from transformers import EvalPrediction

    from lpcv.datasets.base import VideoDataset


class ModelOutput:
    """Minimal output wrapper matching the HuggingFace Trainer interface.

    Attributes
    ----------
    loss
        Scalar loss tensor, or ``None`` during inference.
    logits
        Classification logits of shape ``(B, num_classes)``.
    """

    __slots__ = ("loss", "logits")

    def __init__(self, loss: torch.Tensor | None, logits: torch.Tensor) -> None:
        self.loss = loss
        self.logits = logits


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
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


def collate_for_video(
    examples: list[dict],
    *,
    permute_to_cthw: bool = False,
) -> dict[str, torch.Tensor]:
    """Collate a list of sample dicts into a batched dict.

    Parameters
    ----------
    examples
        List of dicts with ``"pixel_values"`` and ``"labels"`` keys.
    permute_to_cthw
        If ``True``, permute ``(B, T, C, H, W)`` → ``(B, C, T, H, W)``
        for models expecting channel-first temporal layout (e.g. R2+1D, X3D).

    Returns
    -------
    dict[str, torch.Tensor]
        Batched ``pixel_values`` and ``labels`` tensors.
    """
    pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
    if permute_to_cthw and pixel_values.ndim == 5 and pixel_values.shape[1] != 3:
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4)
    labels = torch.tensor([ex["labels"] for ex in examples], dtype=torch.long)
    return {"pixel_values": pixel_values, "labels": labels}


def log_freeze_stats(model: nn.Module, strategy: str) -> None:
    """Log the number of trainable vs total parameters after freezing.

    Parameters
    ----------
    model
        The model whose parameters to inspect.
    strategy
        Name of the freeze strategy (for the log message).
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Freeze strategy '{strategy}': {trainable:,}/{total:,} params trainable "
        f"({trainable / total * 100:.1f}%)"
    )


class BaseForClassification(nn.Module):
    """Base ``nn.Module`` for torchvision / torch-hub classification wrappers.

    Subclasses must set :attr:`backbone`, :attr:`num_classes`, and
    :attr:`loss_fn` before calling ``super().__init__()`` or in their own
    ``__init__``.  The shared :meth:`forward`, :meth:`save_pretrained`,
    and helper methods are then inherited.

    Attributes
    ----------
    backbone
        The wrapped backbone network.
    num_classes
        Number of output classes.
    loss_fn
        Loss function (typically ``nn.CrossEntropyLoss``).
    """

    backbone: nn.Module
    num_classes: int
    loss_fn: nn.Module

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
        logits = self.backbone(pixel_values)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return ModelOutput(loss=loss, logits=logits)

    def save_pretrained(self, path: str | Path) -> None:
        """Save the model state dict and metadata to a directory.

        Parameters
        ----------
        path
            Directory to save into.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta: dict[str, Any] = {
            "state_dict": self.backbone.state_dict(),
            "num_classes": self.num_classes,
        }
        meta.update(self._extra_save_meta())
        torch.save(meta, path / "model.pt")

    def _extra_save_meta(self) -> dict[str, Any]:
        """Return additional metadata to include in the checkpoint.

        Override in subclasses to persist extra fields (e.g. preset name).
        """
        return {}


class BaseModelTrainer:
    """Shared HuggingFace Trainer wrapper for non-HF video classification models.

    Handles label extraction, model initialisation (via :meth:`_init_model`),
    parameter freezing (via :meth:`_apply_freeze_strategy`), metric computation,
    collation, and the full training loop.

    Subclasses must override :meth:`_init_model` and
    :meth:`_apply_freeze_strategy`.

    Parameters
    ----------
    config
        Trainer config dataclass with training hyperparameters.
    train_dataset
        Training dataset with a ``features["label"]`` attribute.
    eval_dataset
        Evaluation dataset with the same schema.
    val_transform_config
        Validation transform config to save alongside the model.
    """

    model_display_name: str = "Model"

    def __init__(
        self,
        config: Any,
        train_dataset: VideoDataset,
        eval_dataset: VideoDataset,
        val_transform_config: list[dict[str, Any]] | None = None,
    ) -> None:
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.val_transform_config = val_transform_config

        label_feature = train_dataset.features.get("label")
        if label_feature is None:
            raise ValueError("Dataset must have a 'label' feature for label metadata")
        self.label_names: list[str] = label_feature.names
        self.num_labels = len(self.label_names)

        self.model = self._init_model()
        self._apply_freeze_strategy(config.freeze_strategy)

        if config.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("TF32 enabled for matmul and cuDNN")

        torch.backends.cudnn.benchmark = True

    def _init_model(self) -> BaseForClassification:
        """Create and return the model instance.

        Subclasses must override this method.
        """
        raise NotImplementedError

    def _apply_freeze_strategy(self, strategy: str) -> None:
        """Freeze model parameters according to *strategy*.

        Subclasses must override this method with model-specific logic.
        """
        raise NotImplementedError

    @staticmethod
    def _collate_fn(examples: list[dict]) -> dict[str, torch.Tensor]:
        """Collate samples, permuting ``(T,C,H,W)`` → ``(C,T,H,W)``."""
        return collate_for_video(examples, permute_to_cthw=True)

    @staticmethod
    def _compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
        """Compute top-1 and top-5 accuracy."""
        return compute_metrics(eval_pred)

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

        logger.info(f"Starting {self.model_display_name} training...")
        trainer.train(resume_from_checkpoint=self.config.resume_from_checkpoint)

        output_path = Path(self.config.output_dir) / "best_model"
        self.model.save_pretrained(output_path)

        if self.val_transform_config is not None:
            from lpcv.transforms import save_val_transform_config

            save_val_transform_config(self.val_transform_config, output_path / "val_transform.json")

        logger.info(f"Model saved to {output_path}")
        return output_path
