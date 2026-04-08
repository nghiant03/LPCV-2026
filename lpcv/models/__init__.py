"""Model registry and saved-artifact metadata helpers.

The registry centralises model-specific behaviour so the CLI, evaluation,
and submission flows can resolve architecture defaults, build transforms,
load trained checkpoints, and construct throwaway validation models without
hardcoded branching.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import MISSING, dataclass, field
from dataclasses import fields as dc_fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from loguru import logger

MODEL_CONFIG_FILENAME = "model_config.yaml"
"""Filename used to persist resolved model architecture metadata."""

VAL_TRANSFORM_FILENAME = "val_transform.json"
"""Filename used to persist the resolved validation transform config."""

EXPORT_CONFIG_FILENAME = "export_config.json"
"""Filename used to persist the explicit ONNX adapter contract."""

if TYPE_CHECKING:
    import torch.nn as nn


ConfigResolver = Callable[[dict[str, Any]], dict[str, Any]]
PresetBuilder = Callable[[dict[str, Any]], tuple[list[dict[str, Any]], list[dict[str, Any]]]]
NumFramesResolver = Callable[[dict[str, Any]], int]


@dataclass
class ResolvedModelConfig:
    """Resolved architecture config plus derived training metadata.

    Attributes
    ----------
    model_name
        Registered model name.
    model_config
        Fully resolved architecture config, including defaults.
    train_preset
        Training transform preset for the resolved config.
    val_preset
        Validation transform preset for the resolved config.
    num_frames
        Resolved temporal clip length used for data loading and export.
    """

    model_name: str
    model_config: dict[str, Any]
    train_preset: list[dict[str, Any]]
    val_preset: list[dict[str, Any]]
    num_frames: int


@dataclass
class ModelSpec:
    """Everything needed to train, load, validate, and export a model by name.

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
    config_resolver
        Normalises and validates model-specific config values after dataclass
        defaults are applied.
    preset_builder
        Builds train/val presets from the resolved model config.
    num_frames_resolver
        Extracts the resolved temporal clip length from the model config.
    input_layout
        Expected tensor layout: ``"BTCHW"`` or ``"BCTHW"``.
    input_key
        Keyword argument name for the model's forward method.
    output_extractor
        Extracts logits from the model's raw output.
    throwaway_builder
        ``(num_classes: int, **kwargs) -> nn.Module`` — builds a throwaway
        instance with random weights for pipeline validation.
    """

    train_preset: list[dict[str, Any]]
    val_preset: list[dict[str, Any]]
    config_cls: type
    trainer_cls: type
    loader: Callable[[str], nn.Module]
    config_resolver: ConfigResolver = field(default_factory=lambda: lambda cfg: cfg)
    preset_builder: PresetBuilder | None = None
    num_frames_resolver: NumFramesResolver = field(
        default_factory=lambda: lambda cfg: int(cfg.get("num_frames", 16))
    )
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


def _model_config_path(path: str | Path) -> Path:
    """Resolve a config file path from either a file or artifact directory."""
    path = Path(path)
    return path / MODEL_CONFIG_FILENAME if path.is_dir() else path


def load_model_config(path: str | Path) -> dict[str, Any]:
    """Load a model configuration from a YAML file or artifact directory.

    Parameters
    ----------
    path
        Path to a ``.yaml`` file or a directory containing
        :data:`MODEL_CONFIG_FILENAME`.

    Returns
    -------
    dict[str, Any]
        Parsed config dict. Always contains a ``"model"`` key.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the YAML does not contain a ``"model"`` key.
    """
    path = _model_config_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    if "model" not in cfg:
        raise ValueError(f"Config must contain a 'model' key: {path}")

    logger.info(f"Loaded model config from {path}: model={cfg['model']}")
    return cfg


def save_model_config(config: dict[str, Any], output_dir: str | Path) -> Path:
    """Save a model configuration dict as :data:`MODEL_CONFIG_FILENAME`."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / MODEL_CONFIG_FILENAME
    with open(dest, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Saved model config to {dest}")
    return dest


def _model_specific_defaults(config_cls: type) -> dict[str, Any]:
    """Return model-specific dataclass defaults for a trainer config class."""
    from lpcv.models.base import BaseTrainerConfig

    base_names = {f.name for f in dc_fields(BaseTrainerConfig)}
    defaults: dict[str, Any] = {}
    for dc_field in dc_fields(config_cls):
        if dc_field.name in base_names:
            continue
        if dc_field.default is not MISSING:
            defaults[dc_field.name] = deepcopy(dc_field.default)
        elif dc_field.default_factory is not MISSING:
            defaults[dc_field.name] = dc_field.default_factory()
    return defaults


def resolve_model_config(
    model_name: str, raw_config: dict[str, Any] | None = None
) -> ResolvedModelConfig:
    """Resolve a model config into explicit architecture and transform metadata.

    Parameters
    ----------
    model_name
        Registered model name.
    raw_config
        Partial model config. Values override dataclass defaults.

    Returns
    -------
    ResolvedModelConfig
        Fully resolved architecture config and derived presets.
    """
    spec = get_model_spec(model_name)
    config = _model_specific_defaults(spec.config_cls)
    if raw_config:
        config.update({k: deepcopy(v) for k, v in raw_config.items() if k != "model"})
    config = spec.config_resolver(config)
    train_preset, val_preset = (
        spec.preset_builder(config)
        if spec.preset_builder is not None
        else (deepcopy(spec.train_preset), deepcopy(spec.val_preset))
    )
    num_frames = spec.num_frames_resolver(config)
    return ResolvedModelConfig(
        model_name=model_name,
        model_config={"model": model_name, **config},
        train_preset=train_preset,
        val_preset=val_preset,
        num_frames=num_frames,
    )


def resolve_artifact_model_name(
    model_path: str | Path,
    model_name: str | None = None,
    *,
    force_override: bool = False,
) -> str:
    """Resolve a model name from a saved artifact with optional override.

    Parameters
    ----------
    model_path
        Saved model directory.
    model_name
        Optional explicit model name.
    force_override
        When ``True``, allow *model_name* to differ from the saved artifact.

    Returns
    -------
    str
        Resolved model name.
    """
    artifact_path = Path(model_path)
    saved_name: str | None = None
    config_path = artifact_path / MODEL_CONFIG_FILENAME
    if config_path.is_file():
        saved_name = load_model_config(config_path)["model"]

    if model_name:
        if saved_name is not None and model_name != saved_name and not force_override:
            raise ValueError(
                f"Requested model_type={model_name!r} does not match saved artifact "
                f"model={saved_name!r}. "
                "Pass --force-override to ignore the saved metadata."
            )
        return model_name

    if saved_name is None:
        raise ValueError(
            f"Could not infer model type from {artifact_path}. "
            f"Expected {MODEL_CONFIG_FILENAME} or pass --model-type explicitly."
        )
    return saved_name


def model_config_from_trainer(model_name: str, config: Any) -> dict[str, Any]:
    """Extract model architecture config from a trainer config dataclass."""
    from lpcv.models.base import BaseTrainerConfig

    base_names = {f.name for f in dc_fields(BaseTrainerConfig)}
    result: dict[str, Any] = {"model": model_name}
    for dc_field in dc_fields(config):
        if dc_field.name not in base_names:
            result[dc_field.name] = getattr(config, dc_field.name)
    return result


def _register_builtins() -> None:
    """Register built-in model types (deferred imports for fast CLI startup)."""
    from lpcv.transforms import TRAIN_PRESET, VAL_PRESET, make_presets

    def _copy_default_presets(
        _cfg: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        return deepcopy(TRAIN_PRESET), deepcopy(VAL_PRESET)

    def _resolve_num_frames(cfg: dict[str, Any]) -> int:
        return int(cfg.get("num_frames", 16))

    def _resolve_x3d_config(cfg: dict[str, Any]) -> dict[str, Any]:
        from lpcv.models.x3d import X3D_PRESET_DEFAULTS

        resolved = deepcopy(cfg)
        preset = resolved.get("preset", "x3d_m")
        if preset not in X3D_PRESET_DEFAULTS:
            available = ", ".join(sorted(X3D_PRESET_DEFAULTS))
            raise ValueError(f"Unknown X3D preset {preset!r}. Available: {available}")

        defaults = X3D_PRESET_DEFAULTS[preset]
        num_frames = int(resolved.get("num_frames", 0))
        crop_size = int(resolved.get("crop_size", 0))
        resolved["preset"] = preset
        resolved["num_frames"] = num_frames if num_frames > 0 else defaults["num_frames"]
        resolved["crop_size"] = crop_size if crop_size > 0 else defaults["crop_size"]
        return resolved

    def _build_x3d_presets(
        cfg: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        from lpcv.datasets.info import X3D_MEAN, X3D_STD

        return make_presets(mean=X3D_MEAN, std=X3D_STD, crop_size=int(cfg["crop_size"]))

    def _resolve_mvitv2_config(cfg: dict[str, Any]) -> dict[str, Any]:
        resolved = deepcopy(cfg)
        resolved["num_frames"] = int(resolved.get("num_frames", 16))
        resolved["crop_size"] = int(resolved.get("crop_size", 112))
        return resolved

    def _build_mvit_like_presets(
        cfg: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        from lpcv.datasets.info import IMAGENET_MEAN, IMAGENET_STD

        return make_presets(mean=IMAGENET_MEAN, std=IMAGENET_STD, crop_size=int(cfg["crop_size"]))

    def _resolve_tsm_config(cfg: dict[str, Any]) -> dict[str, Any]:
        from lpcv.models.tsm import TSM_BACKBONES

        resolved = deepcopy(cfg)
        backbone = resolved.get("backbone", "resnet50")
        if backbone not in TSM_BACKBONES:
            available = ", ".join(sorted(TSM_BACKBONES))
            raise ValueError(f"Unknown TSM backbone {backbone!r}. Available: {available}")
        resolved["backbone"] = backbone
        resolved["num_frames"] = int(resolved.get("num_frames", 8))
        resolved["shift_div"] = int(resolved.get("shift_div", 8))
        resolved["shift_last_n"] = int(resolved.get("shift_last_n", 2))
        return resolved

    def _build_tsm_presets(
        _cfg: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        from lpcv.datasets.info import IMAGENET_MEAN, IMAGENET_STD

        return make_presets(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def _resolve_stam_config(cfg: dict[str, Any]) -> dict[str, Any]:
        resolved = deepcopy(cfg)
        resolved["num_frames"] = int(resolved.get("num_frames", 16))
        resolved["crop_size"] = int(resolved.get("crop_size", 112))
        resolved["patch_size"] = int(resolved.get("patch_size", 16))
        resolved["embed_dim"] = int(resolved.get("embed_dim", 768))
        resolved["spatial_depth"] = int(resolved.get("spatial_depth", 12))
        resolved["num_heads"] = int(resolved.get("num_heads", 12))
        resolved["temporal_layers"] = int(resolved.get("temporal_layers", 6))
        return resolved

    def _load_videomae(path: str) -> nn.Module:
        from transformers import VideoMAEForVideoClassification

        return VideoMAEForVideoClassification.from_pretrained(path)

    def _build_videomae_throwaway(num_classes: int, **kwargs: Any) -> nn.Module:
        from transformers import VideoMAEConfig, VideoMAEForVideoClassification

        config = VideoMAEConfig(
            num_labels=num_classes,
            num_frames=int(kwargs.get("num_frames", 16)),
        )
        return VideoMAEForVideoClassification(config)

    def _make_videomae_spec() -> ModelSpec:
        from lpcv.models.videomae import VideoMAEModelTrainer, VideoMAETrainerConfig

        return ModelSpec(
            train_preset=TRAIN_PRESET,
            val_preset=VAL_PRESET,
            config_cls=VideoMAETrainerConfig,
            trainer_cls=VideoMAEModelTrainer,
            loader=_load_videomae,
            config_resolver=lambda cfg: deepcopy(cfg),
            preset_builder=_copy_default_presets,
            num_frames_resolver=_resolve_num_frames,
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
            config_resolver=lambda cfg: deepcopy(cfg),
            preset_builder=_copy_default_presets,
            num_frames_resolver=_resolve_num_frames,
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
        from lpcv.models.x3d import X3DModelTrainer, X3DTrainerConfig

        default_cfg = _resolve_x3d_config(_model_specific_defaults(X3DTrainerConfig))
        train_preset, val_preset = _build_x3d_presets(default_cfg)
        return ModelSpec(
            train_preset=train_preset,
            val_preset=val_preset,
            config_cls=X3DTrainerConfig,
            trainer_cls=X3DModelTrainer,
            loader=_load_x3d,
            config_resolver=_resolve_x3d_config,
            preset_builder=_build_x3d_presets,
            num_frames_resolver=_resolve_num_frames,
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
            num_frames=int(kwargs.get("num_frames", 8)),
            shift_div=int(kwargs.get("shift_div", 8)),
            shift_last_n=int(kwargs.get("shift_last_n", 2)),
            pretrained=False,
        )

    def _make_tsm_spec() -> ModelSpec:
        from lpcv.models.tsm import TSMModelTrainer, TSMTrainerConfig

        train_preset, val_preset = _build_tsm_presets({})
        return ModelSpec(
            train_preset=train_preset,
            val_preset=val_preset,
            config_cls=TSMTrainerConfig,
            trainer_cls=TSMModelTrainer,
            loader=_load_tsm,
            config_resolver=_resolve_tsm_config,
            preset_builder=_build_tsm_presets,
            num_frames_resolver=_resolve_num_frames,
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
            crop_size=int(kwargs.get("crop_size", 112)),
            pretrained=False,
        )

    def _make_mvitv2_spec() -> ModelSpec:
        from lpcv.models.mvitv2 import MViTv2ModelTrainer, MViTv2TrainerConfig

        default_cfg = _resolve_mvitv2_config(_model_specific_defaults(MViTv2TrainerConfig))
        train_preset, val_preset = _build_mvit_like_presets(default_cfg)
        return ModelSpec(
            train_preset=train_preset,
            val_preset=val_preset,
            config_cls=MViTv2TrainerConfig,
            trainer_cls=MViTv2ModelTrainer,
            loader=_load_mvitv2,
            config_resolver=_resolve_mvitv2_config,
            preset_builder=_build_mvit_like_presets,
            num_frames_resolver=_resolve_num_frames,
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
            num_frames=int(kwargs.get("num_frames", 16)),
            crop_size=int(kwargs.get("crop_size", 112)),
            patch_size=int(kwargs.get("patch_size", 16)),
            embed_dim=int(kwargs.get("embed_dim", 768)),
            spatial_depth=int(kwargs.get("spatial_depth", 12)),
            num_heads=int(kwargs.get("num_heads", 12)),
            temporal_layers=int(kwargs.get("temporal_layers", 6)),
        )

    def _make_stam_spec() -> ModelSpec:
        from lpcv.models.stam import STAMModelTrainer, STAMTrainerConfig

        default_cfg = _resolve_stam_config(_model_specific_defaults(STAMTrainerConfig))
        train_preset, val_preset = _build_mvit_like_presets(default_cfg)
        return ModelSpec(
            train_preset=train_preset,
            val_preset=val_preset,
            config_cls=STAMTrainerConfig,
            trainer_cls=STAMModelTrainer,
            loader=_load_stam,
            config_resolver=_resolve_stam_config,
            preset_builder=_build_mvit_like_presets,
            num_frames_resolver=_resolve_num_frames,
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
