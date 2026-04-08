"""Registry-based video transform system.

Transforms are registered by name via the :func:`register` decorator and
instantiated from configuration dicts using :func:`build_transform`.  All
spatial/temporal transforms operate on ``torch.Tensor`` with shape
``(T, C, H, W)``.

Built-in presets:

- ``COMPETITION_PRESET`` — the competition's fixed preprocessing pipeline
  (R2+1D normalisation, 128×171 resize, 112×112 center crop).  Single
  source of truth for preprocess and adapter diffing.
- ``TRAIN_PRESET`` / ``VAL_PRESET`` — default presets matching the LPCVC
  reference solution.

Each trainer saves the val transform config alongside the model checkpoint.
The submission pipeline converts supported validation transforms into an
explicit adapter contract for ONNX export.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose

from lpcv.datasets.info import R2PLUS1D_MEAN, R2PLUS1D_STD

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
# Competition preset (single source of truth)
# ---------------------------------------------------------------------------

COMPETITION_PRESET: list[dict[str, Any]] = [
    {"name": "ScalePixels"},
    {"name": "Resize", "height": 128, "width": 171},
    {"name": "Normalize", "mean": R2PLUS1D_MEAN, "std": R2PLUS1D_STD},
    {"name": "CenterCrop", "height": 112},
]
"""Competition's fixed preprocessing pipeline (R2+1D norm, 128×171, 112×112 crop)."""

# ---------------------------------------------------------------------------
# Default presets (match LPCVC reference solution)
# ---------------------------------------------------------------------------


def make_presets(
    mean: list[float] | None = None,
    std: list[float] | None = None,
    resize_height: int = 128,
    resize_width: int = 171,
    crop_size: int = 112,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build train/val presets with customisable normalisation and spatial size.

    Parameters
    ----------
    mean
        Per-channel mean.  Defaults to R(2+1)D mean (competition default).
    std
        Per-channel std.  Defaults to R(2+1)D std (competition default).
    resize_height
        Height to resize frames to before cropping.
    resize_width
        Width to resize frames to before cropping.
    crop_size
        Spatial crop size (square).

    Returns
    -------
    tuple[list[dict[str, Any]], list[dict[str, Any]]]
        ``(train_preset, val_preset)`` step dicts.
    """
    mean = mean if mean is not None else R2PLUS1D_MEAN
    std = std if std is not None else R2PLUS1D_STD

    train: list[dict[str, Any]] = [
        {"name": "ScalePixels"},
        {"name": "Resize", "height": resize_height, "width": resize_width},
        {"name": "RandomHorizontalFlip", "p": 0.5},
        {"name": "Normalize", "mean": mean, "std": std},
        {"name": "RandomCrop", "height": crop_size},
    ]
    val: list[dict[str, Any]] = [
        {"name": "ScalePixels"},
        {"name": "Resize", "height": resize_height, "width": resize_width},
        {"name": "Normalize", "mean": mean, "std": std},
        {"name": "CenterCrop", "height": crop_size},
    ]
    return train, val


TRAIN_PRESET, VAL_PRESET = make_presets()
"""Default presets — R(2+1)D normalisation, matching the LPCVC reference solution."""


# ---------------------------------------------------------------------------
# Save / load / export-contract utilities
# ---------------------------------------------------------------------------


def save_val_transform_config(config: list[dict[str, Any]], path: str | Path) -> None:
    """Save a validation transform config to a JSON file.

    Parameters
    ----------
    config
        List of transform step dicts (same format as presets).
    path
        File path to write the JSON.
    """
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def load_val_transform_config(path: str | Path) -> list[dict[str, Any]]:
    """Load a validation transform config from a JSON file.

    Parameters
    ----------
    path
        File path to the JSON config.

    Returns
    -------
    list[dict[str, Any]]
        Transform step dicts.
    """
    import json

    with open(Path(path), encoding="utf-8") as f:
        return json.load(f)


def _normalise_spatial_step(step: dict[str, Any]) -> dict[str, int]:
    """Return explicit ``height``/``width`` values for a spatial transform step."""
    height = int(step["height"])
    width = int(step.get("width", height))
    return {"height": height, "width": width}


def _parse_supported_val_config(val_config: list[dict[str, Any]]) -> dict[str, Any]:
    """Parse a supported validation transform config into explicit components.

    Supported validation transforms are the deterministic subset used by the
    current model registry: ``ScalePixels``, ``Resize``, ``Normalize``, and
    ``CenterCrop``.
    """
    parsed: dict[str, Any] = {
        "scale_pixels": None,
        "resize": None,
        "normalize": None,
        "center_crop": None,
    }

    for step in val_config:
        name = step["name"]
        if name == "ScalePixels":
            scale = float(step.get("scale", 255.0))
            if scale != 255.0:
                raise ValueError(f"Unsupported ScalePixels value {scale}. Expected 255.0.")
            parsed["scale_pixels"] = scale
        elif name == "Resize":
            parsed["resize"] = _normalise_spatial_step(step)
        elif name == "Normalize":
            parsed["normalize"] = {
                "mean": [float(x) for x in step["mean"]],
                "std": [float(x) for x in step["std"]],
            }
        elif name == "CenterCrop":
            parsed["center_crop"] = _normalise_spatial_step(step)
        else:
            raise ValueError(
                f"Unsupported validation transform {name!r} for export. "
                "Only ScalePixels, Resize, Normalize, and CenterCrop are supported."
            )

    return parsed


def build_export_config(
    val_config: list[dict[str, Any]],
    *,
    input_layout: str = "BCTHW",
    input_key: str = "pixel_values",
    num_frames: int = 16,
) -> dict[str, Any]:
    """Build an explicit ONNX adapter contract from a validation config.

    The returned config describes only the transforms that must be applied on
    top of the competition input tensor. Unsupported validation transforms
    raise ``ValueError`` instead of being approximated silently.
    """
    baseline = _parse_supported_val_config(COMPETITION_PRESET)
    parsed = _parse_supported_val_config(val_config)
    resize_diff = parsed["resize"] != baseline["resize"]
    crop_diff = parsed["center_crop"] != baseline["center_crop"]

    target_resize = parsed["resize"] if resize_diff else None
    target_crop = parsed["center_crop"] if (crop_diff or resize_diff) else None

    if target_crop is not None:
        crop_h = int(target_crop["height"])
        crop_w = int(target_crop["width"])
        resize_h = int(target_resize["height"]) if target_resize is not None else 112
        resize_w = int(target_resize["width"]) if target_resize is not None else 112
        if crop_h > resize_h or crop_w > resize_w:
            raise ValueError(
                "Unsupported validation crop "
                f"{target_crop} after resize "
                f"{target_resize or {'height': 112, 'width': 112}}."
            )

    return {
        "num_frames": int(num_frames),
        "input_layout": input_layout,
        "input_key": input_key,
        "source_spatial_size": {"height": 112, "width": 112},
        "source_normalization": {
            "mean": [float(x) for x in R2PLUS1D_MEAN],
            "std": [float(x) for x in R2PLUS1D_STD],
        },
        "target_resize": target_resize,
        "target_crop": target_crop,
        "target_normalization": (
            parsed["normalize"] if parsed["normalize"] != baseline["normalize"] else None
        ),
    }


def save_export_config(config: dict[str, Any], path: str | Path) -> None:
    """Save an export adapter config to JSON."""
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def load_export_config(path: str | Path) -> dict[str, Any]:
    """Load an export adapter config from JSON."""
    import json

    with open(Path(path), encoding="utf-8") as f:
        return json.load(f)
