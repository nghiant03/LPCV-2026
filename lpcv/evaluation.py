"""Evaluation utilities — top-k accuracy, H5 logit evaluation, model evaluation.

Provides:

- :func:`topk_accuracy` — generic top-k accuracy on tensors.
- :func:`evaluate_h5` — evaluate precomputed logits in HDF5 format.
- :func:`evaluate_model` — end-to-end inference + evaluation on a dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from loguru import logger
from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Callable


def topk_accuracy(
    preds: torch.Tensor,
    targets: torch.Tensor,
    topk: tuple[int, ...] = (1, 5),
) -> list[torch.Tensor]:
    """Compute top-k classification accuracy.

    Parameters
    ----------
    preds
        Logits or probabilities of shape ``(N, C)``.
    targets
        Ground-truth class indices of shape ``(N,)``.
    topk
        Tuple of k values to compute accuracy for.

    Returns
    -------
    list[torch.Tensor]
        List of scalar tensors, one per k, as percentages (0–100).
    """
    maxk = max(topk)
    _, pred = preds.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    return [(correct[:k].reshape(-1).float().sum(0) / preds.size(0)) * 100.0 for k in topk]


def load_logits_h5(h5_path: str | Path) -> np.ndarray:
    """Load logits from an HDF5 file.

    Expects logits under the ``data/0/`` group with keys ``sample_0``,
    ``sample_1``, etc.

    Parameters
    ----------
    h5_path
        Path to the HDF5 file.

    Returns
    -------
    np.ndarray
        Stacked logit array of shape ``(N, C)``.
    """
    import h5py

    h5_path = str(h5_path)
    logits = []
    with h5py.File(h5_path, "r") as f:
        grp = f["data/0"]
        assert isinstance(grp, h5py.Group)
        sorted_keys = sorted(grp.keys(), key=lambda x: int(x.split("_")[1]))
        for k in sorted_keys:
            ds = grp[k]
            assert isinstance(ds, h5py.Dataset)
            logits.append(np.asarray(ds[...]).squeeze())
    return np.stack(logits, axis=0)


def load_labels_from_manifest(
    manifest_path: str | Path,
    class_to_idx: dict[str, int],
) -> list[int]:
    """Load ground-truth labels from a JSONL manifest.

    Each line must be a JSON object with at least a ``"label"`` field whose
    value matches a key in *class_to_idx*.

    Parameters
    ----------
    manifest_path
        Path to the JSONL manifest file.
    class_to_idx
        Mapping from class name to integer index.

    Returns
    -------
    list[int]
        Integer labels in manifest order.

    Raises
    ------
    KeyError
        If a label in the manifest is not found in *class_to_idx*.
    """
    labels: list[int] = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            label = record["label"]
            if label not in class_to_idx:
                raise KeyError(f"Label '{label}' from manifest not found in class map.")
            labels.append(class_to_idx[label])
    return labels


def evaluate_h5(
    h5_path: str | Path,
    manifest_path: str | Path,
    class_map_path: str | Path,
    verbose: bool = False,
) -> dict[str, float]:
    """Evaluate precomputed logits in HDF5 format against a manifest.

    Parameters
    ----------
    h5_path
        Path to the HDF5 logits file.
    manifest_path
        Path to the JSONL manifest with ground-truth labels.
    class_map_path
        Path to a JSON file mapping class names to integer indices.
    verbose
        If ``True``, log the first 10 predictions with match markers.

    Returns
    -------
    dict[str, float]
        ``{"top1_accuracy": <float>, "top5_accuracy": <float>}`` as
        percentages.

    Raises
    ------
    ValueError
        If the H5 file has more results than the manifest has labels.
    """
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
        raise ValueError(f"H5 has more results ({n_logits}) than manifest labels ({n_labels}).")

    if verbose:
        pred_indices = torch.argmax(probs, dim=1)
        n_show = min(10, pred_indices.shape[0])
        for i in range(n_show):
            p = int(pred_indices[i].item())
            g = int(label_tensor[i].item())
            marker = "✓" if p == g else "✗"
            logger.info(
                f"  [{i}] {marker}  pred={idx_to_class[p]} ({p})  gt={idx_to_class[g]} ({g})"
            )

    acc1, acc5 = topk_accuracy(probs, label_tensor, topk=(1, 5))
    return {"top1_accuracy": acc1.item(), "top5_accuracy": acc5.item()}


def evaluate_model(
    model_path: str | Path,
    data_dir: str | Path,
    num_frames: int = 16,
    batch_size: int = 8,
    num_workers: int = 4,
    augmentation: Callable[[list[Image.Image]], list[Image.Image]] | None = None,
) -> dict[str, float]:
    """Run end-to-end inference on a dataset and compute accuracy.

    Loads a saved ``VideoMAEForVideoClassification`` checkpoint, preprocesses
    the evaluation split, runs batched forward passes, and returns top-1 /
    top-5 accuracy.

    Parameters
    ----------
    model_path
        Path to the saved model directory (contains config + weights).
    data_dir
        Root directory of the QEVD dataset (videofolder or saved DatasetDict).
    num_frames
        Number of frames to sample per video.
    batch_size
        Inference batch size.
    num_workers
        Dataloader workers (unused currently — batching is manual).
    augmentation
        Optional callable applied to sampled PIL frames before the
        processor.  Receives and returns a list of PIL images.

    Returns
    -------
    dict[str, float]
        ``{"top1_accuracy": <float>, "top5_accuracy": <float>}`` as
        percentages.

    Raises
    ------
    ValueError
        If the loaded dataset is not a ``DatasetDict`` or has no evaluation
        split.
    """
    from datasets import DatasetDict, load_dataset, load_from_disk
    from transformers import (
        VideoMAEForVideoClassification,
        VideoMAEImageProcessor,
    )

    processor = VideoMAEImageProcessor.from_pretrained(str(model_path))
    model = VideoMAEForVideoClassification.from_pretrained(str(model_path))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # type: ignore[assignment]

    data_path = Path(data_dir) if not isinstance(data_dir, Path) else data_dir
    try:
        ds = load_from_disk(str(data_path))
    except Exception:
        ds = load_dataset("videofolder", data_dir=str(data_path))
    if not isinstance(ds, DatasetDict):
        raise ValueError("Expected a DatasetDict with train/val splits.")

    eval_split_name = "val" if "val" in ds else "validation" if "validation" in ds else "test"
    if eval_split_name not in ds:
        raise ValueError(f"No evaluation split found in dataset. Available: {list(ds.keys())}")

    eval_ds = ds[eval_split_name]
    is_decoded = "frames" in eval_ds.column_names

    def _frames_to_pil(frames_array: np.ndarray) -> list[Image.Image]:
        return [Image.fromarray(frames_array[i]) for i in range(frames_array.shape[0])]

    def preprocess(examples: dict) -> dict:
        from lpcv.datasets.utils import subsample

        all_pixel_values = []
        sources = examples["frames"] if is_decoded else examples["video"]

        for source in sources:
            frames = _frames_to_pil(np.array(source)) if is_decoded else list(source)

            sampled = subsample(frames, num_frames, mode="uniform")

            if augmentation is not None:
                sampled = augmentation(sampled)

            pixel_values = processor(sampled, return_tensors="pt")["pixel_values"]
            all_pixel_values.append(pixel_values.squeeze(0))

        examples["pixel_values"] = all_pixel_values
        examples["labels"] = examples["label"]
        return examples

    logger.info(f"Preprocessing {len(eval_ds)} evaluation samples...")
    cols_to_remove = ["frames"] if is_decoded else ["video"]
    processed = eval_ds.map(preprocess, batched=True, batch_size=4, remove_columns=cols_to_remove)
    processed.set_format("torch")

    all_logits = []
    all_labels = []

    logger.info("Running inference...")
    for i in range(0, len(processed), batch_size):
        batch_items = [processed[j] for j in range(i, min(i + batch_size, len(processed)))]
        pixel_values = torch.stack([item["pixel_values"] for item in batch_items]).to(device)
        labels = torch.tensor([item["labels"] for item in batch_items], dtype=torch.long)

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
