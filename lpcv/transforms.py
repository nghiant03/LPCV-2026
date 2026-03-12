from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose

TransformFactory = Callable[..., Callable[[torch.Tensor], torch.Tensor]]

_REGISTRY: dict[str, TransformFactory] = {}


def register(name: str):
    def _decorator(cls: TransformFactory) -> TransformFactory:
        _REGISTRY[name] = cls
        return cls

    return _decorator


def get(name: str) -> TransformFactory:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown transform '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def build_transform(steps: list[dict[str, Any]]) -> Compose:
    transforms: list[Callable[[torch.Tensor], torch.Tensor]] = []
    for step in steps:
        step = dict(step)
        name = step.pop("name")
        transforms.append(get(name)(**step))
    return Compose(transforms)


# ---------------------------------------------------------------------------
# Format transforms
# ---------------------------------------------------------------------------


@register("FromNumpy")
class FromNumpy:
    def __call__(self, source: Any) -> torch.Tensor:
        return torch.from_numpy(np.array(source)).float()


@register("FromVideo")
class FromVideo:
    def __call__(self, source: Any) -> torch.Tensor:
        frames = list(source)
        return torch.stack(
            [torch.from_numpy(np.array(f)).permute(2, 0, 1) for f in frames]
        ).float()


# ---------------------------------------------------------------------------
# Temporal transforms
# ---------------------------------------------------------------------------


@register("UniformTemporalSubsample")
class UniformTemporalSubsample:
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
    def __init__(self, scale: float = 255.0):
        self.scale = scale

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        return video / self.scale


@register("Normalize")
class Normalize:
    def __init__(self, mean: list[float], std: list[float]):
        self.mean = mean
        self.std = std

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.mean, dtype=video.dtype, device=video.device).view(1, -1, 1, 1)
        std = torch.tensor(self.std, dtype=video.dtype, device=video.device).view(1, -1, 1, 1)
        return (video - mean) / std


@register("RandomShortSideScale")
class RandomShortSideScale:
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
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            return video.flip(-1)
        return video


@register("Resize")
class Resize:
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

VAL_PRESET: list[dict[str, Any]] = [
    {"name": "UniformTemporalSubsample", "num_frames": 16},
    {"name": "ScalePixels"},
    {"name": "Normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    {"name": "Resize", "height": 224},
]
