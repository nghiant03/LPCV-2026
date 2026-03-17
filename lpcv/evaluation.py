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
    clips_per_video: int = 1,
    augmentation: Callable[[list[Image.Image]], list[Image.Image]] | None = None,
) -> dict[str, float]:
    """Run end-to-end inference on a dataset and compute accuracy.

    Loads a saved ``VideoMAEForVideoClassification`` checkpoint, preprocesses
    the evaluation split, runs batched forward passes, and returns clip-level
    and video-level top-1 / top-5 accuracy.

    When *clips_per_video* > 1 each video is sampled multiple times using
    uniformly-spaced starting offsets.  The softmax probabilities of all clips
    belonging to the same video are **summed** (matching the reference
    evaluation from the LPCVC Track 2 sample solution) and accuracy is
    computed on the aggregated predictions.

    Parameters
    ----------
    model_path
        Path to the saved model directory (contains config + weights).
    data_dir
        Root directory of the QEVD dataset (videofolder or saved DatasetDict).
    num_frames
        Number of frames to sample per clip.
    batch_size
        Inference batch size (in clips).
    num_workers
        Dataloader workers (unused currently — batching is manual).
    clips_per_video
        Number of clips to extract from each video.  When ``1`` (default),
        behaviour is identical to single-sample evaluation.
    augmentation
        Optional callable applied to sampled PIL frames before the
        processor.  Receives and returns a list of PIL images.

    Returns
    -------
    dict[str, float]
        ``{"clip_top1_accuracy", "clip_top5_accuracy",
        "video_top1_accuracy", "video_top5_accuracy"}`` as percentages.

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

    model_path = Path(model_path)
    model = VideoMAEForVideoClassification.from_pretrained(str(model_path))
    model.eval()

    processor_config = model_path / "preprocessor_config.json"
    if processor_config.exists():
        processor = VideoMAEImageProcessor.from_pretrained(str(model_path))
    else:
        logger.info(
            "No preprocessor_config.json in {}; constructing processor with ImageNet stats",
            model_path,
        )
        processor = VideoMAEImageProcessor(
            do_resize=True,
            size={"shortest_edge": 224},
            do_center_crop=True,
            crop_size={"height": 224, "width": 224},
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
        )

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

    def _sample_clips(
        frames: list[Image.Image],
        n_clips: int,
        n_frames: int,
    ) -> list[list[Image.Image]]:
        """Sample *n_clips* clips of *n_frames* from *frames*.

        Clips are uniformly spaced across the video.  Each clip is a
        contiguous window of *n_frames* frames (or uniform-subsampled if
        the video is shorter than ``n_clips * n_frames``).
        """
        total = len(frames)
        if total == 0:
            return []

        if n_clips == 1:
            from lpcv.datasets.utils import subsample

            return [subsample(frames, n_frames, mode="uniform")]

        clips: list[list[Image.Image]] = []
        if total <= n_frames:
            clip = frames + [frames[-1]] * (n_frames - total)
            return [clip] * n_clips

        max_start = total - n_frames
        starts = np.linspace(0, max_start, n_clips).astype(int).tolist()

        for s in starts:
            clip = frames[s : s + n_frames]
            clips.append(clip)
        return clips

    def preprocess(examples: dict) -> dict:
        all_pixel_values: list[torch.Tensor] = []
        all_labels: list[int] = []
        all_video_indices: list[int] = []

        sources = examples["frames"] if is_decoded else examples["video"]

        for video_idx_in_batch, source in enumerate(sources):
            frames = _frames_to_pil(np.array(source)) if is_decoded else list(source)
            clips = _sample_clips(frames, clips_per_video, num_frames)

            for clip_frames in clips:
                if augmentation is not None:
                    clip_frames = augmentation(clip_frames)

                pixel_values = processor(clip_frames, return_tensors="pt")["pixel_values"]
                all_pixel_values.append(pixel_values.squeeze(0))
                all_labels.append(examples["label"][video_idx_in_batch])
                all_video_indices.append(video_idx_in_batch)

        examples["pixel_values"] = all_pixel_values
        examples["labels"] = all_labels
        examples["video_index"] = all_video_indices
        return examples

    logger.info(
        f"Preprocessing {len(eval_ds)} evaluation samples ({clips_per_video} clip(s) per video)..."
    )
    cols_to_remove = ["frames"] if is_decoded else ["video"]
    processed = eval_ds.map(
        preprocess,
        batched=True,
        batch_size=4,
        remove_columns=cols_to_remove,
    )
    processed.set_format("torch")

    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_video_indices: list[torch.Tensor] = []

    logger.info("Running inference...")
    for i in range(0, len(processed), batch_size):
        batch_items = [processed[j] for j in range(i, min(i + batch_size, len(processed)))]
        pixel_values = torch.stack([item["pixel_values"] for item in batch_items]).to(device)
        labels = torch.tensor([item["labels"] for item in batch_items], dtype=torch.long)
        video_indices = torch.tensor(
            [item["video_index"] for item in batch_items], dtype=torch.long
        )

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)

        all_logits.append(outputs.logits.cpu())
        all_labels.append(labels)
        all_video_indices.append(video_indices)

    logits_t = torch.cat(all_logits, dim=0)
    labels_t = torch.cat(all_labels, dim=0)
    video_indices_t = torch.cat(all_video_indices, dim=0)
    clip_probs = torch.softmax(logits_t, dim=1)

    clip_acc1, clip_acc5 = topk_accuracy(clip_probs, labels_t, topk=(1, 5))

    num_videos = int(video_indices_t.max().item()) + 1
    num_classes = clip_probs.shape[1]
    agg_probs = torch.zeros((num_videos, num_classes), dtype=torch.float32)
    agg_labels = torch.zeros(num_videos, dtype=torch.long)

    for idx in range(clip_probs.shape[0]):
        vid = int(video_indices_t[idx].item())
        agg_probs[vid] += clip_probs[idx]
        agg_labels[vid] = labels_t[idx]

    video_acc1, video_acc5 = topk_accuracy(agg_probs, agg_labels, topk=(1, 5))

    results = {
        "clip_top1_accuracy": clip_acc1.item(),
        "clip_top5_accuracy": clip_acc5.item(),
        "video_top1_accuracy": video_acc1.item(),
        "video_top5_accuracy": video_acc5.item(),
    }
    logger.info(f"Clip  Acc@1: {results['clip_top1_accuracy']:.2f}%")
    logger.info(f"Clip  Acc@5: {results['clip_top5_accuracy']:.2f}%")
    logger.info(f"Video Acc@1: {results['video_top1_accuracy']:.2f}%")
    logger.info(f"Video Acc@5: {results['video_top5_accuracy']:.2f}%")

    return results
