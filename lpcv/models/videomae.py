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

from lpcv.datasets.utils import subsample

if TYPE_CHECKING:
    from datasets import DatasetDict

DEFAULT_MODEL_NAME = "MCG-NJU/videomae-base"
DEFAULT_NUM_FRAMES = 16
DEFAULT_IMAGE_SIZE = 224
DEFAULT_FREEZE_STRATEGY = "none"


@dataclass
class VideoMAETrainerConfig:
    model_name: str = DEFAULT_MODEL_NAME
    num_frames: int = DEFAULT_NUM_FRAMES
    image_size: int = DEFAULT_IMAGE_SIZE
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
    remove_unused_columns: bool = False
    resume_from_checkpoint: str | None = None
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    freeze_strategy: str = DEFAULT_FREEZE_STRATEGY
    torch_compile: bool = False
    tf32: bool = False
    extra_args: dict[str, Any] = field(default_factory=dict)


class VideoMAEModelTrainer:
    def __init__(
        self,
        config: VideoMAETrainerConfig,
        dataset: DatasetDict,
    ):
        self.config = config
        self.dataset = dataset

        first_split = next(iter(dataset))
        self.label_names: list[str] = dataset[first_split].features["label"].names
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

    def _preprocess(self, examples: dict) -> dict:
        videos = examples["video"]
        num_frames = self.config.num_frames

        all_pixel_values = []
        for video in videos:
            sampled = subsample(list(video), num_frames)
            pixel_values = self.processor(sampled, return_tensors="pt")["pixel_values"]
            all_pixel_values.append(pixel_values.squeeze(0))

        examples["pixel_values"] = all_pixel_values
        examples["labels"] = examples["label"]
        return examples

    def _compute_metrics(self, eval_pred: EvalPrediction) -> dict[str, float]:
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
        pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
        labels = torch.tensor([ex["labels"] for ex in examples], dtype=torch.long)
        return {"pixel_values": pixel_values, "labels": labels}

    @property
    def _is_precomputed(self) -> bool:
        first_split = next(iter(self.dataset))
        return "pixel_values" in self.dataset[first_split].column_names

    def train(self) -> Path:
        if self._is_precomputed:
            logger.info("Using precomputed dataset")
            processed = dict(self.dataset)
        else:
            logger.info("Setting up lazy preprocessing...")
            processed = {}
            for split in self.dataset:
                ds = self.dataset[split]
                ds.set_transform(self._preprocess)
                processed[split] = ds

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
            remove_unused_columns=self.config.remove_unused_columns,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            gradient_checkpointing=self.config.gradient_checkpointing,
            torch_compile=self.config.torch_compile,
            tf32=self.config.tf32,
            report_to="none",
            **self.config.extra_args,
        )

        train_ds = processed["train"]
        eval_ds = processed.get("val", processed.get("validation", processed.get("test")))

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
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
