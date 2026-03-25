"""Model sub-package — trainer registry and model implementations.

Provides a lightweight registry so the CLI and submission pipeline can
look up trainers, loaders, and default presets by name without hardcoding
imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch.nn as nn


@dataclass
class ModelSpec:
    """Everything needed to train, load, and export a model by name.

    Attributes
    ----------
    train_preset
        Default training transform preset (list of step dicts).
    val_preset
        Default validation transform preset (list of step dicts).
    config_cls
        Trainer config dataclass (e.g. ``VideoMAETrainerConfig``).
    trainer_cls
        Trainer class (e.g. ``VideoMAEModelTrainer``).
    loader
        ``(path: str) -> nn.Module`` — loads a checkpoint for inference/export.
    input_layout
        Expected tensor layout: ``"BTCHW"`` or ``"BCTHW"``.
    input_key
        Keyword argument name for the model's forward method.
    output_extractor
        Extracts logits from the model's raw output.
    """

    train_preset: list[dict[str, Any]]
    val_preset: list[dict[str, Any]]
    config_cls: type
    trainer_cls: type
    loader: Callable[[str], nn.Module]
    input_layout: str = "BCTHW"
    input_key: str = "pixel_values"
    output_extractor: Callable[..., Any] = field(default_factory=lambda: lambda out: out.logits)


_REGISTRY: dict[str, ModelSpec] = {}


def register_model(name: str, spec: ModelSpec) -> None:
    """Register a model spec under *name*."""
    _REGISTRY[name] = spec


def get_model_spec(name: str) -> ModelSpec:
    """Look up a registered model spec.

    Raises
    ------
    KeyError
        If *name* is not registered.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown model {name!r}. Available: {available}")
    return _REGISTRY[name]


def list_models() -> list[str]:
    """Return sorted list of registered model names."""
    return sorted(_REGISTRY)


def _register_builtins() -> None:
    """Register built-in model types (deferred imports for fast CLI startup)."""
    from lpcv.transforms import TRAIN_PRESET, VAL_PRESET, make_presets

    def _load_videomae(path: str) -> nn.Module:
        from transformers import VideoMAEForVideoClassification

        return VideoMAEForVideoClassification.from_pretrained(path)

    def _make_videomae_spec() -> ModelSpec:
        from lpcv.models.videomae import VideoMAEModelTrainer, VideoMAETrainerConfig

        return ModelSpec(
            train_preset=TRAIN_PRESET,
            val_preset=VAL_PRESET,
            config_cls=VideoMAETrainerConfig,
            trainer_cls=VideoMAEModelTrainer,
            loader=_load_videomae,
            input_layout="BTCHW",
            input_key="pixel_values",
            output_extractor=lambda out: out.logits,
        )

    def _load_r2plus1d(path: str) -> nn.Module:
        from lpcv.models.r2plus1d import R2Plus1DForClassification

        return R2Plus1DForClassification.load_pretrained(path)

    def _make_r2plus1d_spec() -> ModelSpec:
        from lpcv.models.r2plus1d import R2Plus1DModelTrainer, R2Plus1DTrainerConfig

        return ModelSpec(
            train_preset=TRAIN_PRESET,
            val_preset=VAL_PRESET,
            config_cls=R2Plus1DTrainerConfig,
            trainer_cls=R2Plus1DModelTrainer,
            loader=_load_r2plus1d,
            input_layout="BCTHW",
            input_key="pixel_values",
            output_extractor=lambda out: out.logits,
        )

    def _load_x3d(path: str) -> nn.Module:
        from lpcv.models.x3d import X3DForClassification

        return X3DForClassification.load_pretrained(path)

    def _make_x3d_spec() -> ModelSpec:
        from lpcv.datasets.info import X3D_MEAN, X3D_STD
        from lpcv.models.x3d import X3D_PRESET_DEFAULTS, X3DModelTrainer, X3DTrainerConfig

        default_crop = X3D_PRESET_DEFAULTS["x3d_m"]["crop_size"]
        train_preset, val_preset = make_presets(
            mean=X3D_MEAN,
            std=X3D_STD,
            resize_height=default_crop,
            resize_width=default_crop,
            crop_size=default_crop,
        )
        return ModelSpec(
            train_preset=train_preset,
            val_preset=val_preset,
            config_cls=X3DTrainerConfig,
            trainer_cls=X3DModelTrainer,
            loader=_load_x3d,
            input_layout="BCTHW",
            input_key="pixel_values",
            output_extractor=lambda out: out.logits,
        )

    register_model("videomae", _make_videomae_spec())
    register_model("r2plus1d", _make_r2plus1d_spec())
    register_model("x3d", _make_x3d_spec())


_register_builtins()
