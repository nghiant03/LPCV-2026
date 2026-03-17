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
from tqdm import tqdm

if TYPE_CHECKING:
    from lpcv.datasets.base import VideoDataset


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
    eval_ds: VideoDataset,
    batch_size: int = 8,
    clips_per_video: int = 1,
) -> dict[str, float]:
    """Run end-to-end inference on a dataset and compute accuracy.

    Loads a saved ``VideoMAEForVideoClassification`` checkpoint, runs batched
    forward passes on the provided evaluation dataset, and returns clip-level
    and video-level top-1 / top-5 accuracy.

    When *clips_per_video* > 1 the dataset is sampled multiple times per video.
    The softmax probabilities of all clips belonging to the same video are
    **summed** (matching the LPCVC Track 2 reference evaluation) and accuracy
    is computed on the aggregated predictions.

    Parameters
    ----------
    model_path
        Path to the saved model directory (contains config + weights).
    eval_ds
        A :class:`~lpcv.datasets.base.VideoDataset` (or any PyTorch
        ``Dataset``) yielding ``{"pixel_values": Tensor, "labels": int}``
        dicts.
    batch_size
        Inference batch size (in clips).
    clips_per_video
        Number of clips to extract from each video.  When ``1`` (default),
        behaviour is identical to single-sample evaluation.

    Returns
    -------
    dict[str, float]
        ``{"clip_top1_accuracy", "clip_top5_accuracy",
        "video_top1_accuracy", "video_top5_accuracy"}`` as percentages.
    """
    from torch.utils.data import DataLoader
    from transformers import VideoMAEForVideoClassification

    model_path = Path(model_path)
    model = VideoMAEForVideoClassification.from_pretrained(str(model_path))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # type: ignore[assignment]

    loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    logger.info(f"Running inference on {len(eval_ds)} samples...")
    for batch in tqdm(loader, desc="Evaluating", unit="batch"):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"]

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)

        all_logits.append(outputs.logits.cpu())
        all_labels.append(labels)

    logits_t = torch.cat(all_logits, dim=0)
    labels_t = torch.cat(all_labels, dim=0)
    clip_probs = torch.softmax(logits_t, dim=1)

    clip_acc1, clip_acc5 = topk_accuracy(clip_probs, labels_t, topk=(1, 5))

    num_videos = len(eval_ds) // clips_per_video
    num_classes = clip_probs.shape[1]
    agg_probs = torch.zeros((num_videos, num_classes), dtype=torch.float32)
    agg_labels = torch.zeros(num_videos, dtype=torch.long)

    for idx in range(clip_probs.shape[0]):
        vid = idx // clips_per_video
        agg_probs[vid] += clip_probs[idx]
        agg_labels[vid] = labels_t[idx]

    video_acc1, video_acc5 = topk_accuracy(agg_probs, agg_labels, topk=(1, 5))

    return {
        "clip_top1_accuracy": clip_acc1.item(),
        "clip_top5_accuracy": clip_acc5.item(),
        "video_top1_accuracy": video_acc1.item(),
        "video_top5_accuracy": video_acc5.item(),
    }
