"""VideoDataset and loader for the videofolder layout.

Provides a PyTorch ``Dataset`` that delegates frame decoding to an injected
:class:`~lpcv.datasets.decoder.VideoDecoder`, plus a factory function
(:func:`load_video_dataset`) that builds train/val splits from disk.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger
from torch.utils.data import Dataset

from lpcv.datasets.info import TARGET_LABEL_FILE_NAME, VIDEO_EXTENSIONS

if TYPE_CHECKING:
    from torchvision.transforms import Compose

    from lpcv.datasets.decoder import VideoDecoder


class VideoDataset(Dataset):
    """Unified video dataset that delegates frame decoding to an injected decoder.

    Each sample is returned as a dict with keys ``"pixel_values"`` (a float
    tensor of shape ``(T, C, H, W)``) and ``"labels"`` (an integer class id).

    Parameters
    ----------
    video_paths
        Ordered list of video file paths.
    labels
        Corresponding integer class ids (same length as *video_paths*).
    label_names
        Ordered list of class label strings.
    decoder
        A :class:`~lpcv.datasets.decoder.VideoDecoder` instance used to
        extract frames.
    transform
        Optional ``torchvision.transforms.Compose`` applied to the decoded
        tensor **after** decoding.
    num_frames
        Number of frames to sample per video.
    label_names
        Ordered list of class label strings, exposed directly for trainer
        metadata and artifact generation.
    """

    def __init__(
        self,
        video_paths: list[Path],
        labels: list[int],
        label_names: list[str],
        decoder: VideoDecoder,
        transform: Compose | None = None,
        num_frames: int = 16,
    ) -> None:
        if len(video_paths) != len(labels):
            raise ValueError("video_paths and labels must have the same length")

        self.video_paths = video_paths
        self.labels = labels
        self.label_names = label_names
        self.decoder = decoder
        self.transform = transform
        self.num_frames = num_frames

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return the *idx*-th sample."""
        path = self.video_paths[idx]
        label = self.labels[idx]

        video = self.decoder.decode(path, self.num_frames)

        if self.transform is not None:
            video = self.transform(video)

        return {"pixel_values": video, "labels": label}


def _stratified_subsample(
    paths: list[Path],
    labels: list[int],
    fraction: float,
    seed: int = 42,
) -> tuple[list[Path], list[int]]:
    """Subsample *paths* and *labels* keeping the original class ratio.

    Parameters
    ----------
    paths
        Video file paths.
    labels
        Corresponding integer class ids.
    fraction
        Fraction of data to keep (0.0, 1.0].
    seed
        Random seed for reproducibility.

    Returns
    -------
    tuple[list[Path], list[int]]
        Subsampled ``(paths, labels)``.
    """
    by_class: dict[int, list[int]] = {}
    for idx, lbl in enumerate(labels):
        by_class.setdefault(lbl, []).append(idx)

    rng = random.Random(seed)
    selected: list[int] = []
    for cls_id in sorted(by_class):
        indices = by_class[cls_id]
        k = max(1, math.ceil(len(indices) * fraction))
        selected.extend(rng.sample(indices, k))

    selected.sort()
    return [paths[i] for i in selected], [labels[i] for i in selected]


def load_video_dataset(
    data_dir: str | Path,
    decoder: VideoDecoder,
    train_transform: Compose | None = None,
    val_transform: Compose | None = None,
    num_frames: int = 16,
    data_percent: float = 100.0,
) -> tuple[VideoDataset, VideoDataset]:
    """Build train and val :class:`VideoDataset` from a videofolder layout.

    Expected directory structure::

        data_dir/
          class_labels.json      # JSON list of class names
          train/<class>/*.mp4
          val/<class>/*.mp4

    Parameters
    ----------
    data_dir
        Root directory containing ``class_labels.json`` and split folders.
    decoder
        A :class:`~lpcv.datasets.decoder.VideoDecoder` used for frame
        extraction.
    train_transform
        Transform pipeline applied to training samples.
    val_transform
        Transform pipeline applied to validation samples.
    num_frames
        Number of frames to sample per video.
    data_percent
        Percentage of data to use (0–100]. Stratified sampling preserves
        the original class ratio. Applies to both train and val splits.

    Returns
    -------
    tuple[VideoDataset, VideoDataset]
        ``(train_dataset, val_dataset)``.

    Raises
    ------
    FileNotFoundError
        If the label file or a required split directory is missing.
    ValueError
        If *data_percent* is not in (0, 100].
    """
    if not (0.0 < data_percent <= 100.0):
        raise ValueError(f"data_percent must be in (0, 100], got {data_percent}")
    data_dir = Path(data_dir)
    label_file = data_dir / TARGET_LABEL_FILE_NAME
    if not label_file.is_file():
        raise FileNotFoundError(f"Label file not found: {label_file}")

    with open(label_file) as f:
        label_names: list[str] = json.load(f)
    label2id = {name: i for i, name in enumerate(label_names)}

    splits: dict[str, VideoDataset] = {}
    for split in ("train", "val"):
        split_dir = data_dir / split
        if not split_dir.is_dir():
            continue

        paths: list[Path] = []
        labels: list[int] = []

        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            if class_name not in label2id:
                logger.warning(f"Skipping unknown class directory: {class_name}")
                continue
            class_id = label2id[class_name]

            for video_file in sorted(class_dir.iterdir()):
                if video_file.is_file() and video_file.suffix.lower() in VIDEO_EXTENSIONS:
                    paths.append(video_file)
                    labels.append(class_id)

        if data_percent < 100.0:
            total = len(paths)
            paths, labels = _stratified_subsample(paths, labels, data_percent / 100.0)
            logger.info(f"[{split}] Subsampled {data_percent:.1f}%: {total} → {len(paths)} videos")

        transform = train_transform if split == "train" else val_transform
        ds = VideoDataset(
            video_paths=paths,
            labels=labels,
            label_names=label_names,
            decoder=decoder,
            transform=transform,
            num_frames=num_frames,
        )
        splits[split] = ds
        logger.info(f"[{type(decoder).__name__}] {split}: {len(ds)} videos")

    if "train" not in splits:
        raise FileNotFoundError(f"No train split found in {data_dir}")
    if "val" not in splits:
        raise FileNotFoundError(f"No val split found in {data_dir}")

    return splits["train"], splits["val"]
