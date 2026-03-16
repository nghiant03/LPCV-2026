"""VideoDataset and loader for the videofolder layout.

Provides a PyTorch ``Dataset`` that delegates frame decoding to an injected
:class:`~lpcv.datasets.decoder.VideoDecoder`, plus a factory function
(:func:`load_video_dataset`) that builds train/val splits from disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from loguru import logger
from torch.utils.data import Dataset

from lpcv.datasets.info import TARGET_LABEL_FILE_NAME, VIDEO_EXTENSIONS

if TYPE_CHECKING:
    from torchvision.transforms import Compose

    from lpcv.datasets.decoder import VideoDecoder


class LabelFeature:
    """Mimics the HuggingFace ``dataset.features["label"]`` interface.

    Exposes a ``names`` attribute so that ``VideoMAEModelTrainer`` can read
    label metadata without changes.

    Parameters
    ----------
    label_names
        Ordered list of class label strings.
    """

    def __init__(self, label_names: list[str]) -> None:
        self.names = label_names


class DatasetFeatures:
    """Thin wrapper exposing a ``get("label")`` API compatible with ``VideoMAEModelTrainer``.

    Parameters
    ----------
    label_names
        Ordered list of class label strings.
    """

    def __init__(self, label_names: list[str]) -> None:
        self._label = LabelFeature(label_names)

    def get(self, key: str) -> LabelFeature | None:
        """Return the feature for *key*, or ``None`` if not found."""
        if key == "label":
            return self._label
        return None


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
        self.decoder = decoder
        self.transform = transform
        self.num_frames = num_frames
        self.features = DatasetFeatures(label_names)

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return the *idx*-th sample.

        If decoding fails a zero tensor is returned so that training can
        continue without crashing on individual corrupt files.
        """
        path = self.video_paths[idx]
        label = self.labels[idx]

        try:
            video = self.decoder.decode(path, self.num_frames)
        except Exception:
            logger.warning(f"Failed to decode {path}, returning zeros")
            video = torch.zeros(self.num_frames, 3, 224, 224)

        if self.transform is not None:
            video = self.transform(video)

        return {"pixel_values": video, "labels": label}


def load_video_dataset(
    data_dir: str | Path,
    decoder: VideoDecoder,
    train_transform: Compose | None = None,
    val_transform: Compose | None = None,
    num_frames: int = 16,
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

    Returns
    -------
    tuple[VideoDataset, VideoDataset]
        ``(train_dataset, val_dataset)``.

    Raises
    ------
    FileNotFoundError
        If the label file or a required split directory is missing.
    """
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
