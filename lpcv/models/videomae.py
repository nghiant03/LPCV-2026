"""VideoMAE model trainer — config and HuggingFace Trainer wrapper.

Provides :class:`VideoMAETrainerConfig` (training hyperparameters) and
:class:`VideoMAEModelTrainer` (handles model loading, freeze strategies,
custom collation, metrics, and checkpoint saving).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from loguru import logger
from transformers import (
    EvalPrediction,
    Trainer,
    TrainingArguments,
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
)

if TYPE_CHECKING:
    from datasets import Dataset
    from torch.utils.data import Dataset as TorchDataset


@dataclass
class VideoMAETrainerConfig:
    """All hyperparameters for a VideoMAE training run.

    Attributes
    ----------
    model_name
        HuggingFace model identifier or local path.
    num_frames
        Number of frames sampled per video.
    image_size
        Spatial resolution (height = width) after preprocessing.
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
    gradient_checkpointing
        Trade compute for memory by recomputing activations.
    max_steps
        Stop after N optimiser steps (overrides *num_train_epochs* when > 0).
    freeze_strategy
        Parameter freeze strategy: ``"none"``, ``"backbone"``, or ``"partial"``.
    lr_scheduler_type
        Learning rate scheduler: ``"linear"``, ``"cosine"``,
        ``"cosine_with_restarts"``, ``"polynomial"``, ``"constant"``,
        ``"constant_with_warmup"``, or ``"inverse_sqrt"``.
    torch_compile
        Use ``torch.compile`` for fused kernels.
    tf32
        Enable TF32 math on Ampere+ GPUs.
    extra_args
        Additional keyword arguments forwarded to ``TrainingArguments``.
    """

    model_name: str = "MCG-NJU/videomae-base"
    num_frames: int = 16
    image_size: int = 224
    output_dir: str = "output"
    num_train_epochs: int = 15
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.05
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
    gradient_checkpointing: bool = False
    max_steps: int = -1
    freeze_strategy: str = "none"
    lr_scheduler_type: str = "linear"
    torch_compile: bool = False
    tf32: bool = False
    extra_args: dict[str, Any] = field(default_factory=dict)


class VideoMAEModelTrainer:
    """High-level wrapper around HuggingFace ``Trainer`` for VideoMAE fine-tuning.

    Handles model initialisation, parameter freezing, custom collation,
    metric computation, and saving.

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

    Raises
    ------
    ValueError
        If the dataset does not have a ``"label"`` feature.
    """

    def __init__(
        self,
        config: VideoMAETrainerConfig,
        train_dataset: Dataset | TorchDataset,
        eval_dataset: Dataset | TorchDataset,
    ):
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        label_feature = train_dataset.features.get("label")  # type: ignore[union-attr]
        if label_feature is None:
            raise ValueError("Dataset must have a 'label' feature for label metadata")
        self.label_names: list[str] = label_feature.names
        self.num_labels = len(self.label_names)
        self.label2id = {name: i for i, name in enumerate(self.label_names)}
        self.id2label = {i: name for i, name in enumerate(self.label_names)}

        logger.info(
            f"Initializing VideoMAE trainer: model={config.model_name}, "
            f"labels={self.num_labels}, epochs={config.num_train_epochs}"
        )

        self.processor = VideoMAEImageProcessor.from_pretrained(
            config.model_name,
            do_resize=True,
            size={"shortest_edge": config.image_size},
            do_center_crop=True,
            crop_size={"height": config.image_size, "width": config.image_size},
        )

        self.model = VideoMAEForVideoClassification.from_pretrained(
            config.model_name,
            num_labels=self.num_labels,
            label2id=self.label2id,
            id2label=self.id2label,
            ignore_mismatched_sizes=True,
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
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            logger.info(
                f"Froze backbone: {trainable:,}/{total:,} params trainable "
                f"({trainable / total * 100:.1f}%)"
            )
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
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            logger.info(
                f"Partial freeze (last 2 encoder layers + classifier): "
                f"{trainable:,}/{total:,} params trainable ({trainable / total * 100:.1f}%)"
            )
        else:
            logger.warning(f"Unknown freeze strategy '{strategy}', skipping")

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
        logits = eval_pred.predictions
        labels = eval_pred.label_ids
        logits_t = torch.as_tensor(logits)
        predictions = logits_t.argmax(dim=-1)
        labels_t = torch.as_tensor(labels)

        acc1 = (predictions == labels_t).float().mean().item() * 100.0

        top5_pred = logits_t.topk(min(5, logits_t.shape[-1]), dim=-1).indices
        correct_top5 = top5_pred.eq(labels_t.unsqueeze(-1)).any(dim=-1).float().mean().item()
        acc5 = correct_top5 * 100.0

        return {"accuracy": acc1, "top5_accuracy": acc5}

    def _collate_fn(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        """Collate a list of sample dicts into a batched dict.

        Parameters
        ----------
        examples
            List of dicts with ``"pixel_values"`` and ``"labels"`` keys.

        Returns
        -------
        dict[str, torch.Tensor]
            Batched ``pixel_values`` and ``labels`` tensors.
        """
        pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
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
            gradient_checkpointing=self.config.gradient_checkpointing,
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

        logger.info("Starting training...")
        trainer.train(resume_from_checkpoint=self.config.resume_from_checkpoint)

        output_path = Path(self.config.output_dir) / "best_model"
        trainer.save_model(str(output_path))
        self.processor.save_pretrained(str(output_path))

        logger.info(f"Model saved to {output_path}")
        return output_path
