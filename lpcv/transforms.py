"""Registry-based video transform system.

Transforms are registered by name via the :func:`register` decorator and
instantiated from configuration dicts using :func:`build_transform`.  All
spatial/temporal transforms operate on ``torch.Tensor`` with shape
``(T, C, H, W)``.

Four built-in presets are provided:

- ``TRAIN_PRESET`` / ``VAL_PRESET`` — include temporal subsampling (for
  HuggingFace ``videofolder`` pipeline).
- ``DECODE_TRAIN_PRESET`` / ``DECODE_VAL_PRESET`` — exclude temporal
  subsampling (decoder handles it).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose

TransformFactory = Callable[..., Callable[[torch.Tensor], torch.Tensor]]
"""Type alias for a callable that produces a transform function."""

_REGISTRY: dict[str, TransformFactory] = {}


def register(name: str):
    """Class decorator that registers a transform under *name*.

    Parameters
    ----------
    name
        Unique name used to look up the transform in configuration dicts.

    Examples
    --------
    >>> @register("MyTransform")
    ... class MyTransform:
    ...     def __call__(self, video: torch.Tensor) -> torch.Tensor:
    ...         return video
    """

    def _decorator(cls: TransformFactory) -> TransformFactory:
        _REGISTRY[name] = cls
        return cls

    return _decorator


def get(name: str) -> TransformFactory:
    """Look up a registered transform factory by *name*.

    Parameters
    ----------
    name
        The registered name.

    Returns
    -------
    TransformFactory
        The callable class / factory.

    Raises
    ------
    KeyError
        If *name* is not registered.
    """
    if name not in _REGISTRY:
        raise KeyError(f"Unknown transform '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def build_transform(steps: list[dict[str, Any]]) -> Compose:
    """Build a ``torchvision.transforms.Compose`` pipeline from config dicts.

    Each dict must contain a ``"name"`` key whose value matches a registered
    transform.  Remaining keys are forwarded as keyword arguments to the
    transform constructor.

    Parameters
    ----------
    steps
        List of ``{"name": "<registered_name>", **kwargs}`` dicts.

    Returns
    -------
    Compose
        Composed transform pipeline.
    """
    transforms: list[Callable[[torch.Tensor], torch.Tensor]] = []
    for step in steps:
        step = dict(step)
        name = step.pop("name")
        transforms.append(get(name)(**step))
    return Compose(transforms)


class VideoTransformCallable:
    """Wraps a ``Compose`` pipeline for use with HuggingFace ``set_transform``.

    Applies the transform to each video in a batched examples dict and
    copies the ``"label"`` column to ``"labels"`` (expected by the Trainer).

    Parameters
    ----------
    transform
        A ``torchvision.transforms.Compose`` pipeline.
    """

    def __init__(self, transform: Compose):
        self.transform = transform

    def __call__(self, examples: dict) -> dict:
        """Apply the transform to each video in the batch."""
        examples["pixel_values"] = [self.transform(video) for video in examples["video"]]
        examples["labels"] = examples["label"]
        return examples


# ---------------------------------------------------------------------------
# Format transforms
# ---------------------------------------------------------------------------


@register("FromVideo")
class FromVideo:
    """Convert a sequence of PIL images or numpy arrays to a ``(T, C, H, W)`` tensor.

    Handles the common case where raw frames are ``(H, W, C)`` numpy arrays
    or PIL images.  The output is a float tensor with values in ``[0, 255]``.
    """

    def __call__(self, source: Any) -> torch.Tensor:
        frames = list(source)
        t = torch.stack([torch.from_numpy(np.array(f)) for f in frames]).float()
        if t.ndim == 4 and t.shape[-1] in (1, 3, 4):
            t = t.permute(0, 3, 1, 2)
        return t


# ---------------------------------------------------------------------------
# Temporal transforms
# ---------------------------------------------------------------------------


@register("UniformTemporalSubsample")
class UniformTemporalSubsample:
    """Uniformly subsample *num_frames* frames along the temporal axis.

    Parameters
    ----------
    num_frames
        Number of output frames.
    """

    def __init__(self, num_frames: int):
        self.num_frames = num_frames

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        t = video.shape[0]
        if t == self.num_frames:
            return video
        indices = torch.linspace(0, t - 1, self.num_frames).long()
        return video[indices]


@register("ScalePixels")
class ScalePixels:
    """Divide pixel values by a constant (default 255).

    Parameters
    ----------
    scale
        Divisor applied to the entire tensor.
    """

    def __init__(self, scale: float = 255.0):
        self.scale = scale

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        return video / self.scale


@register("Normalize")
class Normalize:
    """Per-channel mean/std normalization.

    Parameters
    ----------
    mean
        Channel means, e.g. ``[0.485, 0.456, 0.406]``.
    std
        Channel standard deviations, e.g. ``[0.229, 0.224, 0.225]``.
    """

    def __init__(self, mean: list[float], std: list[float]):
        self.mean = mean
        self.std = std

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.mean, dtype=video.dtype, device=video.device).view(1, -1, 1, 1)
        std = torch.tensor(self.std, dtype=video.dtype, device=video.device).view(1, -1, 1, 1)
        return (video - mean) / std


@register("RandomShortSideScale")
class RandomShortSideScale:
    """Randomly scale so the short side falls within ``[min_size, max_size]``.

    Parameters
    ----------
    min_size
        Minimum short-side length after scaling.
    max_size
        Maximum short-side length after scaling.
    """

    def __init__(self, min_size: int = 256, max_size: int = 320):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        _, _, h, w = video.shape
        short_side = min(h, w)
        target = int(torch.randint(self.min_size, self.max_size + 1, (1,)).item())
        scale = target / short_side
        new_h = int(h * scale)
        new_w = int(w * scale)
        return F.interpolate(video, size=(new_h, new_w), mode="bilinear", align_corners=False)


@register("ShortSideScale")
class ShortSideScale:
    """Scale so the short side equals exactly *size*.

    Parameters
    ----------
    size
        Target short-side length.
    """

    def __init__(self, size: int = 256):
        self.size = size

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        _, _, h, w = video.shape
        short_side = min(h, w)
        if short_side == self.size:
            return video
        scale = self.size / short_side
        new_h = int(h * scale)
        new_w = int(w * scale)
        return F.interpolate(video, size=(new_h, new_w), mode="bilinear", align_corners=False)


@register("RandomCrop")
class RandomCrop:
    """Random spatial crop.

    Parameters
    ----------
    height
        Crop height in pixels.
    width
        Crop width in pixels.  Defaults to *height* (square crop).
    """

    def __init__(self, height: int, width: int | None = None):
        self.height = height
        self.width = width if width is not None else height

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        _, _, h, w = video.shape
        top = int(torch.randint(0, max(h - self.height, 0) + 1, (1,)).item())
        left = int(torch.randint(0, max(w - self.width, 0) + 1, (1,)).item())
        return video[:, :, top : top + self.height, left : left + self.width]


@register("CenterCrop")
class CenterCrop:
    """Deterministic center crop.

    Parameters
    ----------
    height
        Crop height in pixels.
    width
        Crop width in pixels.  Defaults to *height*.
    """

    def __init__(self, height: int, width: int | None = None):
        self.height = height
        self.width = width if width is not None else height

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        _, _, h, w = video.shape
        top = (h - self.height) // 2
        left = (w - self.width) // 2
        return video[:, :, top : top + self.height, left : left + self.width]


@register("RandomHorizontalFlip")
class RandomHorizontalFlip:
    """Randomly flip the video horizontally with probability *p*.

    Parameters
    ----------
    p
        Flip probability (default ``0.5``).
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            return video.flip(-1)
        return video


@register("Resize")
class Resize:
    """Bilinear resize to exact ``(height, width)`` dimensions.

    Parameters
    ----------
    height
        Target height.
    width
        Target width.  Defaults to *height* (square output).
    """

    def __init__(self, height: int, width: int | None = None):
        self.height = height
        self.width = width if width is not None else height

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            video,
            size=(self.height, self.width),
            mode="bilinear",
            align_corners=False,
        )


# ---------------------------------------------------------------------------
# Default presets
# ---------------------------------------------------------------------------

TRAIN_PRESET: list[dict[str, Any]] = [
    {"name": "UniformTemporalSubsample", "num_frames": 16},
    {"name": "ScalePixels"},
    {"name": "Normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    {"name": "RandomShortSideScale", "min_size": 256, "max_size": 320},
    {"name": "RandomCrop", "height": 224},
    {"name": "RandomHorizontalFlip", "p": 0.5},
]
"""Training preset with temporal subsampling and random augmentation."""

VAL_PRESET: list[dict[str, Any]] = [
    {"name": "UniformTemporalSubsample", "num_frames": 16},
    {"name": "ScalePixels"},
    {"name": "Normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    {"name": "Resize", "height": 224},
]
"""Validation preset with temporal subsampling and deterministic resize."""

DECODE_TRAIN_PRESET: list[dict[str, Any]] = [
    {"name": "ScalePixels"},
    {"name": "Normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    {"name": "RandomShortSideScale", "min_size": 256, "max_size": 320},
    {"name": "RandomCrop", "height": 224},
    {"name": "RandomHorizontalFlip", "p": 0.5},
]
"""Training preset without temporal subsampling (decoder handles it)."""

DECODE_VAL_PRESET: list[dict[str, Any]] = [
    {"name": "ScalePixels"},
    {"name": "Normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    {"name": "Resize", "height": 224},
]
"""Validation preset without temporal subsampling (decoder handles it)."""
