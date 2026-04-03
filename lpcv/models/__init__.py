"""Model sub-package — trainer registry and model implementations.

Provides a lightweight registry so the CLI and submission pipeline can
look up trainers, loaders, and default presets by name without hardcoding
imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from loguru import logger

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
    throwaway_builder
        ``(num_classes: int, **kwargs) -> nn.Module`` — builds a throwaway
        instance with random weights for pipeline validation.  Extra keyword
        arguments are model architecture parameters from a config YAML.
    """

    train_preset: list[dict[str, Any]]
    val_preset: list[dict[str, Any]]
    config_cls: type
    trainer_cls: type
    loader: Callable[[str], nn.Module]
    input_layout: str = "BCTHW"
    input_key: str = "pixel_values"
    output_extractor: Callable[..., Any] = field(default_factory=lambda: lambda out: out.logits)
    throwaway_builder: Callable[..., nn.Module] | None = None


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


def load_model_config(path: str | Path) -> dict[str, Any]:
    """Load a model configuration from a YAML file.

    Parameters
    ----------
    path
        Path to a ``.yaml`` file.

    Returns
    -------
    dict[str, Any]
        Parsed config dict.  Always contains a ``"model"`` key.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the YAML does not contain a ``"model"`` key.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    if "model" not in cfg:
        raise ValueError(f"Config must contain a 'model' key: {path}")

    logger.info(f"Loaded model config from {path}: model={cfg['model']}")
    return cfg


def save_model_config(config: dict[str, Any], output_dir: str | Path) -> Path:
    """Save a model configuration dict as ``model_config.yaml``.

    Parameters
    ----------
    config
        Model config dict (must contain ``"model"`` key).
    output_dir
        Directory to write into.

    Returns
    -------
    Path
        Path to the written file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "model_config.yaml"
    with open(dest, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Saved model config to {dest}")
    return dest


def model_config_from_trainer(model_name: str, config: Any) -> dict[str, Any]:
    """Extract model architecture config from a trainer config dataclass.

    Pulls model-specific fields (those not in :class:`BaseTrainerConfig`)
    into a dict suitable for YAML serialisation.

    Parameters
    ----------
    model_name
        Registered model name.
    config
        Trainer config dataclass instance.

    Returns
    -------
    dict[str, Any]
        Config dict with ``"model"`` key.
    """
    from dataclasses import fields as dc_fields

    from lpcv.models.base import BaseTrainerConfig

    base_names = {f.name for f in dc_fields(BaseTrainerConfig)}
    result: dict[str, Any] = {"model": model_name}
    for f in dc_fields(config):
        if f.name not in base_names:
            result[f.name] = getattr(config, f.name)
    return result


def _register_builtins() -> None:
    """Register built-in model types (deferred imports for fast CLI startup)."""
    from lpcv.transforms import TRAIN_PRESET, VAL_PRESET, make_presets

    def _load_videomae(path: str) -> nn.Module:
        from transformers import VideoMAEForVideoClassification

        return VideoMAEForVideoClassification.from_pretrained(path)

    def _build_videomae_throwaway(num_classes: int, **_kwargs: Any) -> nn.Module:
        from transformers import VideoMAEConfig, VideoMAEForVideoClassification

        config = VideoMAEConfig(num_labels=num_classes)
        return VideoMAEForVideoClassification(config)

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
            throwaway_builder=_build_videomae_throwaway,
        )

    def _load_r2plus1d(path: str) -> nn.Module:
        from lpcv.models.r2plus1d import R2Plus1DForClassification

        return R2Plus1DForClassification.load_pretrained(path)

    def _build_r2plus1d_throwaway(num_classes: int, **_kwargs: Any) -> nn.Module:
        from lpcv.models.r2plus1d import R2Plus1DForClassification

        return R2Plus1DForClassification(num_classes=num_classes, pretrained=False)

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
            throwaway_builder=_build_r2plus1d_throwaway,
        )

    def _load_x3d(path: str) -> nn.Module:
        from lpcv.models.x3d import X3DForClassification

        return X3DForClassification.load_pretrained(path)

    def _build_x3d_throwaway(num_classes: int, **kwargs: Any) -> nn.Module:
        from lpcv.models.x3d import X3DForClassification

        return X3DForClassification(
            num_classes=num_classes,
            preset=kwargs.get("preset", "x3d_m"),
            pretrained=False,
        )

    def _make_x3d_spec() -> ModelSpec:
        from lpcv.datasets.info import X3D_MEAN, X3D_STD
        from lpcv.models.x3d import X3D_PRESET_DEFAULTS, X3DModelTrainer, X3DTrainerConfig

        default_crop = X3D_PRESET_DEFAULTS["x3d_m"]["crop_size"]
        train_preset, val_preset = make_presets(
            mean=X3D_MEAN,
            std=X3D_STD,
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
            throwaway_builder=_build_x3d_throwaway,
        )

    def _load_tsm(path: str) -> nn.Module:
        from lpcv.models.tsm import TSMForClassification

        return TSMForClassification.load_pretrained(path)

    def _build_tsm_throwaway(num_classes: int, **kwargs: Any) -> nn.Module:
        from lpcv.models.tsm import TSMForClassification

        return TSMForClassification(
            num_classes=num_classes,
            backbone_name=kwargs.get("backbone", "resnet50"),
            num_frames=kwargs.get("num_frames", 8),
            shift_div=kwargs.get("shift_div", 8),
            shift_last_n=kwargs.get("shift_last_n", 2),
            pretrained=False,
        )

    def _make_tsm_spec() -> ModelSpec:
        from lpcv.datasets.info import IMAGENET_MEAN, IMAGENET_STD
        from lpcv.models.tsm import TSMModelTrainer, TSMTrainerConfig

        train_preset, val_preset = make_presets(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )
        return ModelSpec(
            train_preset=train_preset,
            val_preset=val_preset,
            config_cls=TSMTrainerConfig,
            trainer_cls=TSMModelTrainer,
            loader=_load_tsm,
            input_layout="BCTHW",
            input_key="pixel_values",
            output_extractor=lambda out: out.logits,
            throwaway_builder=_build_tsm_throwaway,
        )

    def _load_mvitv2(path: str) -> nn.Module:
        from lpcv.models.mvitv2 import MViTv2ForClassification

        return MViTv2ForClassification.load_pretrained(path)

    def _build_mvitv2_throwaway(num_classes: int, **kwargs: Any) -> nn.Module:
        from lpcv.models.mvitv2 import MViTv2ForClassification

        return MViTv2ForClassification(
            num_classes=num_classes,
            crop_size=kwargs.get("crop_size", 112),
            pretrained=False,
        )

    def _make_mvitv2_spec() -> ModelSpec:
        from lpcv.datasets.info import IMAGENET_MEAN, IMAGENET_STD
        from lpcv.models.mvitv2 import MVITV2_CROP_SIZE, MViTv2ModelTrainer, MViTv2TrainerConfig

        train_preset, val_preset = make_presets(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            crop_size=MVITV2_CROP_SIZE,
        )
        return ModelSpec(
            train_preset=train_preset,
            val_preset=val_preset,
            config_cls=MViTv2TrainerConfig,
            trainer_cls=MViTv2ModelTrainer,
            loader=_load_mvitv2,
            input_layout="BCTHW",
            input_key="pixel_values",
            output_extractor=lambda out: out.logits,
            throwaway_builder=_build_mvitv2_throwaway,
        )

    def _load_stam(path: str) -> nn.Module:
        from lpcv.models.stam import STAMForClassification

        return STAMForClassification.load_pretrained(path)

    def _build_stam_throwaway(num_classes: int, **kwargs: Any) -> nn.Module:
        from lpcv.models.stam import STAMForClassification

        return STAMForClassification(
            num_classes=num_classes,
            num_frames=kwargs.get("num_frames", 16),
            crop_size=kwargs.get("crop_size", 112),
            patch_size=kwargs.get("patch_size", 16),
            embed_dim=kwargs.get("embed_dim", 768),
            spatial_depth=kwargs.get("spatial_depth", 12),
            num_heads=kwargs.get("num_heads", 12),
            temporal_layers=kwargs.get("temporal_layers", 6),
        )

    def _make_stam_spec() -> ModelSpec:
        from lpcv.datasets.info import IMAGENET_MEAN, IMAGENET_STD
        from lpcv.models.stam import STAM_CROP_SIZE, STAMModelTrainer, STAMTrainerConfig

        train_preset, val_preset = make_presets(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            crop_size=STAM_CROP_SIZE,
        )
        return ModelSpec(
            train_preset=train_preset,
            val_preset=val_preset,
            config_cls=STAMTrainerConfig,
            trainer_cls=STAMModelTrainer,
            loader=_load_stam,
            input_layout="BCTHW",
            input_key="pixel_values",
            output_extractor=lambda out: out.logits,
            throwaway_builder=_build_stam_throwaway,
        )

    register_model("videomae", _make_videomae_spec())
    register_model("r2plus1d", _make_r2plus1d_spec())
    register_model("x3d", _make_x3d_spec())
    register_model("tsm", _make_tsm_spec())
    register_model("mvitv2", _make_mvitv2_spec())
    register_model("stam", _make_stam_spec())


_register_builtins()
