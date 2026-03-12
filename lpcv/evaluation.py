from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import torch
from loguru import logger

if TYPE_CHECKING:
    from pathlib import Path


def topk_accuracy(
    preds: torch.Tensor,
    targets: torch.Tensor,
    topk: tuple[int, ...] = (1, 5),
) -> list[torch.Tensor]:
    maxk = max(topk)
    _, pred = preds.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    return [
        (correct[:k].reshape(-1).float().sum(0) / preds.size(0)) * 100.0
        for k in topk
    ]


def load_logits_h5(h5_path: str | Path) -> np.ndarray:
    import h5py

    h5_path = str(h5_path)
    logits = []
    with h5py.File(h5_path, "r") as f:
        grp = f["data/0"]
        sorted_keys = sorted(grp.keys(), key=lambda x: int(x.split("_")[1]))
        for k in sorted_keys:
            logits.append(grp[k][...].squeeze())
    return np.stack(logits, axis=0)


def load_labels_from_manifest(
    manifest_path: str | Path,
    class_to_idx: dict[str, int],
) -> list[int]:
    labels: list[int] = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            label = record["label"]
            if label not in class_to_idx:
                raise KeyError(
                    f"Label '{label}' from manifest not found in class map."
                )
            labels.append(class_to_idx[label])
    return labels


def evaluate_h5(
    h5_path: str | Path,
    manifest_path: str | Path,
    class_map_path: str | Path,
    verbose: bool = False,
) -> dict[str, float]:
    with open(class_map_path, encoding="utf-8") as f:
        class_to_idx: dict[str, int] = json.load(f)
    idx_to_class = {i: cls for cls, i in class_to_idx.items()}

    raw_logits = load_logits_h5(h5_path)
    labels = load_labels_from_manifest(manifest_path, class_to_idx)

    logits = torch.as_tensor(raw_logits, dtype=torch.float32)
    probs = torch.softmax(logits, dim=1)
    label_tensor = torch.tensor(labels, dtype=torch.int64)

    n_logits = probs.shape[0]
    n_labels = label_tensor.shape[0]
    if n_labels > n_logits:
        logger.warning(
            f"H5 has {n_logits} results but manifest has {n_labels} labels. "
            f"Truncating labels to first {n_logits}."
        )
        label_tensor = label_tensor[:n_logits]
    elif n_logits > n_labels:
        raise ValueError(
            f"H5 has more results ({n_logits}) than manifest labels ({n_labels})."
        )

    if verbose:
        pred_indices = torch.argmax(probs, dim=1)
        n_show = min(10, pred_indices.shape[0])
        for i in range(n_show):
            p = int(pred_indices[i].item())
            g = int(label_tensor[i].item())
            marker = "✓" if p == g else "✗"
            logger.info(
                f"  [{i}] {marker}  pred={idx_to_class[p]} ({p})  "
                f"gt={idx_to_class[g]} ({g})"
            )

    acc1, acc5 = topk_accuracy(probs, label_tensor, topk=(1, 5))
    return {"top1_accuracy": acc1.item(), "top5_accuracy": acc5.item()}


def evaluate_model(
    model_path: str | Path,
    data_dir: str | Path,
    num_frames: int = 16,
    batch_size: int = 8,
    num_workers: int = 4,
) -> dict[str, float]:
    from datasets import DatasetDict, load_dataset
    from transformers import (
        VideoMAEForVideoClassification,
        VideoMAEImageProcessor,
    )

    processor = VideoMAEImageProcessor.from_pretrained(str(model_path))
    model = VideoMAEForVideoClassification.from_pretrained(str(model_path))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # type: ignore[assignment]

    ds = load_dataset("videofolder", data_dir=str(data_dir))
    if not isinstance(ds, DatasetDict):
        raise ValueError("Expected a DatasetDict with train/val splits.")

    eval_split_name = "val" if "val" in ds else "validation" if "validation" in ds else "test"
    if eval_split_name not in ds:
        raise ValueError(f"No evaluation split found in dataset. Available: {list(ds.keys())}")

    eval_ds = ds[eval_split_name]

    def preprocess(examples: dict) -> dict:
        videos = examples["video"]
        all_pixel_values = []
        for video in videos:
            frames = list(video)
            total = len(frames)
            if total >= num_frames:
                indices = torch.linspace(0, total - 1, num_frames).long().tolist()
                sampled = [frames[i] for i in indices]
            else:
                sampled = frames + [frames[-1]] * (num_frames - total)
            pixel_values = processor(sampled, return_tensors="pt")["pixel_values"]
            all_pixel_values.append(pixel_values.squeeze(0))
        examples["pixel_values"] = all_pixel_values
        examples["labels"] = examples["label"]
        return examples

    logger.info(f"Preprocessing {len(eval_ds)} evaluation samples...")
    processed = eval_ds.map(preprocess, batched=True, batch_size=4, remove_columns=["video"])
    processed.set_format("torch")

    all_logits = []
    all_labels = []

    logger.info("Running inference...")
    for i in range(0, len(processed), batch_size):
        batch_items = [processed[j] for j in range(i, min(i + batch_size, len(processed)))]
        pixel_values = torch.stack([item["pixel_values"] for item in batch_items]).to(device)
        labels = torch.tensor(
            [item["labels"] for item in batch_items], dtype=torch.long
        )

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)

        all_logits.append(outputs.logits.cpu())
        all_labels.append(labels)

    logits_t = torch.cat(all_logits, dim=0)
    labels_t = torch.cat(all_labels, dim=0)
    probs = torch.softmax(logits_t, dim=1)

    acc1, acc5 = topk_accuracy(probs, labels_t, topk=(1, 5))

    results = {"top1_accuracy": acc1.item(), "top5_accuracy": acc5.item()}
    logger.info(f"Top-1 Accuracy: {results['top1_accuracy']:.2f}%")
    logger.info(f"Top-5 Accuracy: {results['top5_accuracy']:.2f}%")

    return results
