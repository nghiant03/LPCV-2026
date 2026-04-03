"""Shared model components — output wrapper, metrics, collation, base classes.

Provides building blocks reused across model implementations:

- :class:`ModelOutput` — lightweight output wrapper for HuggingFace Trainer.
- :func:`compute_metrics` — top-1 / top-5 accuracy computation.
- :func:`collate_for_video` — video batch collation with optional layout permutation.
- :class:`BaseTrainerConfig` — common training hyperparameters.
- :class:`BaseForClassification` — ``nn.Module`` base for custom classification wrappers.
- :class:`BaseModelTrainer` — shared HuggingFace Trainer integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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

    Supports attribute access (``output.loss``), dict-style access
    (``output["loss"]``), and integer indexing (``output[0]``) so the
    HuggingFace Trainer can consume it in all code paths.

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

    def __getitem__(self, key: str | int | slice) -> Any:
        if isinstance(key, (int, slice)):
            return (self.loss, self.logits)[key]
        return getattr(self, key)


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


@dataclass
class BaseTrainerConfig:
    """Common training hyperparameters shared by all model trainers.

    Subclasses add model-specific fields and may override defaults.

    Attributes
    ----------
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
    extra_args
        Additional keyword arguments forwarded to ``TrainingArguments``.
    """

    output_dir: str = "output"
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    learning_rate: float = 5e-3
    warmup_ratio: float = 0.1
    weight_decay: float = 1e-4
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
    dataloader_persistent_workers: bool = True
    dataloader_prefetch_factor: int | None = 2
    remove_unused_columns: bool = False
    resume_from_checkpoint: str | None = None
    gradient_accumulation_steps: int = 1
    max_steps: int = -1
    lr_scheduler_type: str = "cosine"
    torch_compile: bool = False
    tf32: bool = False
    freeze_strategy: str = "none"
    extra_args: dict[str, Any] = field(default_factory=dict)


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
    """Shared HuggingFace Trainer wrapper for video classification models.

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

    def _init_model(self) -> BaseForClassification | nn.Module:
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
        """Collate samples into a batch.

        Default permutes ``(T,C,H,W)`` → ``(C,T,H,W)`` for models
        expecting channel-first temporal layout.  Override for models
        that expect ``(B,T,C,H,W)`` (e.g. VideoMAE).
        """
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
            **self._extra_training_args(),
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
        self._save_model(trainer, output_path)

        if self.val_transform_config is not None:
            from lpcv.transforms import save_val_transform_config

            save_val_transform_config(self.val_transform_config, output_path / "val_transform.json")

        logger.info(f"Model saved to {output_path}")
        return output_path

    def _extra_training_args(self) -> dict[str, Any]:
        """Return additional kwargs for ``TrainingArguments``.

        Override in subclasses to inject model-specific arguments
        (e.g. ``gradient_checkpointing``, ``ddp_find_unused_parameters``).
        """
        return {"ddp_find_unused_parameters": True}

    def _save_model(self, trainer: Trainer, path: Path) -> None:
        """Save the trained model to *path*.

        Default calls ``self.model.save_pretrained(path)`` for custom
        ``BaseForClassification`` models.  Override for HuggingFace models
        that use ``trainer.save_model()``.
        """
        self.model.save_pretrained(path)  # type: ignore[union-attr]


class DecomposedDepthwiseConv3d(nn.Module):
    """Depthwise 3D convolution decomposed into spatial 2D + temporal 1D.

    Equivalent to ``Conv3d(C, C, kernel, stride, padding, groups=C)`` but
    expressed as a ``Conv2d`` (spatial) followed by a ``Conv1d`` (temporal),
    both depthwise.  This avoids the unsupported depthwise-3D-conv op on
    Qualcomm AI Hub while preserving the learned weights.

    Parameters
    ----------
    channels
        Number of input/output channels (same as ``groups``).
    kernel_size
        3-element list ``[k_t, k_h, k_w]``.
    stride
        3-element list ``[s_t, s_h, s_w]``.
    padding
        3-element list ``[p_t, p_h, p_w]``.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: list[int],
        stride: list[int],
        padding: list[int],
    ) -> None:
        super().__init__()
        k_t, k_h, k_w = kernel_size
        s_t, s_h, s_w = stride
        p_t, p_h, p_w = padding
        self.channels = channels

        self.spatial = nn.Conv2d(
            channels,
            channels,
            kernel_size=(k_h, k_w),
            stride=(s_h, s_w),
            padding=(p_h, p_w),
            groups=channels,
            bias=False,
        )
        self.temporal = nn.Conv1d(
            channels,
            channels,
            kernel_size=k_t,
            stride=s_t,
            padding=p_t,
            groups=channels,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial conv then temporal conv.

        Parameters
        ----------
        x
            Tensor of shape ``(B, C, T, H, W)``.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(B, C, T', H', W')``.
        """
        bx, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(bx * t, c, h, w)
        x = self.spatial(x)
        h2, w2 = x.shape[2], x.shape[3]
        x = x.view(bx, t, c, h2, w2).permute(0, 3, 4, 2, 1).reshape(bx * h2 * w2, c, t)
        x = self.temporal(x)
        t2 = x.shape[2]
        x = x.view(bx, h2, w2, c, t2).permute(0, 3, 4, 1, 2)
        return x

    @classmethod
    def from_conv3d(cls, conv3d: nn.Conv3d) -> DecomposedDepthwiseConv3d:
        """Build from an existing ``Conv3d`` and copy its weights.

        The 3D kernel ``(C, 1, k_t, k_h, k_w)`` is decomposed by summing
        over the temporal axis for the spatial kernel and over spatial axes
        for the temporal kernel, then rescaling so the product approximates
        the original.

        Parameters
        ----------
        conv3d
            Source depthwise ``Conv3d`` with ``groups == in_channels``.

        Returns
        -------
        DecomposedDepthwiseConv3d
            Initialised module with weights derived from *conv3d*.
        """
        c = conv3d.in_channels
        k = [int(x) for x in conv3d.kernel_size]
        s = [int(x) for x in conv3d.stride]
        p = [int(x) for x in conv3d.padding]  # type: ignore[union-attr]
        mod = cls(channels=c, kernel_size=k, stride=s, padding=p)

        with torch.no_grad():
            w3d = conv3d.weight  # (C, 1, k_t, k_h, k_w)
            w_spatial = w3d.sum(dim=2)  # (C, 1, k_h, k_w)
            w_temporal = w3d.sum(dim=(3, 4))  # (C, 1, k_t)
            scale = (1.0 / (k[0] * k[1] * k[2])) ** 0.5
            mod.spatial.weight.copy_(w_spatial * (scale * k[0] ** 0.5))
            mod.temporal.weight.copy_(w_temporal * (scale * (k[1] * k[2]) ** 0.5))

        return mod


def decompose_depthwise_conv3d(model: nn.Module) -> int:
    """Replace all depthwise ``Conv3d`` layers in *model* with 2D+1D decompositions.

    Walks every sub-module and swaps any ``Conv3d`` where
    ``groups == in_channels == out_channels`` with a
    :class:`DecomposedDepthwiseConv3d`.

    Parameters
    ----------
    model
        Any ``nn.Module`` tree.

    Returns
    -------
    int
        Number of ``Conv3d`` layers replaced.
    """
    replaced = 0
    for parent_name, parent in list(model.named_modules()):
        for attr_name, child in list(parent.named_children()):
            if (
                isinstance(child, nn.Conv3d)
                and child.groups == child.in_channels
                and child.in_channels == child.out_channels
            ):
                setattr(parent, attr_name, DecomposedDepthwiseConv3d.from_conv3d(child))
                replaced += 1
                full_name = f"{parent_name}.{attr_name}" if parent_name else attr_name
                logger.info(f"Decomposed depthwise Conv3d: {full_name}")
    return replaced
