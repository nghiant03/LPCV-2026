"""Submission pipeline — preprocess, export, on-device inference via Qualcomm AI Hub.

Provides:

- :func:`preprocess_dataset` — decode videos to ``.npy`` tensors + ``manifest.jsonl``.
- :func:`export_onnx` — wrap model with auto-built adapter and export to ONNX.
- :func:`compile_on_hub` — compile an ONNX model on Qualcomm AI Hub.
- :func:`run_inference_on_hub` — upload tensors and run on-device inference.

The adapter layer is built from an explicit ``export_config.json`` saved
alongside each model checkpoint. Export and compile default to artifact-owned
metadata instead of restated CLI flags.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import onnx
import torch
from loguru import logger
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Callable

DEFAULT_NUM_FRAMES = 16
"""Default number of frames to sample per video clip."""

COMPETITION_SPATIAL_SIZE = 112
"""Spatial resolution of the competition's fixed input pipeline."""

DEFAULT_DEVICE_NAME = "Dragonwing IQ-9075 EVK"
"""Default Qualcomm AI Hub device for compilation and inference."""

CHUNK_SIZE = 538
"""Maximum number of samples per AI Hub inference job (stays under 2 GB flatbuffer limit)."""

FRAME_RATE = 4
"""Frame rate used by the competition's VideoClips-based preprocessing."""

COMPETITION_INPUT_NAME = "video"
"""ONNX input name expected by the competition evaluation pipeline."""


def preprocess_dataset(
    data_dir: str | Path,
    output_dir: str | Path,
    num_frames: int = DEFAULT_NUM_FRAMES,
    decoder_name: str = "pyav",
    target_fps: int = FRAME_RATE,
) -> Path:
    """Decode videos to ``(1, T, 112, 112, 3)`` ``.npy`` tensors and write a manifest.

    Replicates the competition's exact preprocessing pipeline so that locally
    saved tensors match what the organiser's evaluation server produces:

    1. Decode frames, resample to *target_fps* with dynamic adjustment
       for short videos (matching the patched ``VideoClips`` behaviour).
    2. ``ConvertImageDtype(float32)`` → ``Resize(128, 171)``
       → ``Normalize(R2+1D mean/std)`` → ``CenterCrop(112, 112)``.
    3. Add batch dim and arrange to ``(1, T, 112, 112, 3)`` (BTHWC) to
       match the competition's channel-last evaluation layout.

    Parameters
    ----------
    data_dir
        Root of the videofolder dataset (expects ``val/<class>/*.mp4``).
    output_dir
        Directory to write ``.npy`` files and ``manifest.jsonl``.
    num_frames
        Number of frames per clip (must match the competition setting, default 16).
    decoder_name
        Decoder backend name (e.g. ``"pyav"``, ``"torchcodec-cpu"``).
    target_fps
        Target FPS for frame resampling.  Passed to the decoder.

    Returns
    -------
    Path
        Path to the written ``manifest.jsonl``.
    """
    from lpcv.datasets.decoder import get_decoder
    from lpcv.datasets.info import TARGET_LABEL_FILE_NAME, VIDEO_EXTENSIONS
    from lpcv.transforms import COMPETITION_PRESET, build_transform

    decoder = get_decoder(decoder_name, target_fps=target_fps)

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_file = data_dir / TARGET_LABEL_FILE_NAME
    if not label_file.is_file():
        raise FileNotFoundError(f"Label file not found: {label_file}")

    with open(label_file) as f:
        label_names: list[str] = json.load(f)

    val_dir = data_dir / "val"
    if not val_dir.is_dir():
        raise FileNotFoundError(f"Validation split not found: {val_dir}")

    video_entries: list[tuple[Path, str]] = []
    for class_dir in sorted(val_dir.iterdir()):
        if not class_dir.is_dir() or class_dir.name not in label_names:
            continue
        for video_file in sorted(class_dir.iterdir()):
            if video_file.is_file() and video_file.suffix.lower() in VIDEO_EXTENSIONS:
                video_entries.append((video_file, class_dir.name))

    if not video_entries:
        raise FileNotFoundError(f"No videos found in {val_dir}")

    spatial = build_transform(COMPETITION_PRESET)

    manifest_path = output_dir / "manifest.jsonl"
    logger.info(f"Preprocessing {len(video_entries)} videos → {output_dir}")

    with open(manifest_path, "w", encoding="utf-8") as mf:
        for video_path, label in tqdm(video_entries, desc="Preprocessing", unit="video"):
            try:
                clip = decoder.decode(video_path, num_frames)
            except ValueError:
                logger.warning(f"Skipping {video_path}: not enough frames")
                continue

            clip = spatial(clip)
            tensor = clip.unsqueeze(0)

            rel = video_path.relative_to(val_dir)
            npy_rel = rel.with_suffix(".npy")
            npy_path = output_dir / npy_rel
            npy_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(npy_path), tensor.cpu().numpy().astype(np.float32))

            record = {
                "video_path": str(video_path),
                "label": label,
                "tensor_path": str(npy_path),
                "shape": list(tensor.shape),
                "dtype": "float32",
            }
            mf.write(json.dumps(record) + "\n")

    logger.info(f"Manifest written to {manifest_path}")
    return manifest_path


class CompetitionAdapter(torch.nn.Module):
    """Adapter that converts competition input to a model's expected format.

    The competition feeds ``(B, T, H, W, C)`` tensors at 112x112, normalized
    with R(2+1)D mean/std. The adapter permutes to ``(B, C, T, H, W)``
    internally, then applies any re-normalization, resize, or crop needed
    by the wrapped model.
    """

    src_mean: torch.Tensor
    src_std: torch.Tensor
    dst_mean: torch.Tensor
    dst_std: torch.Tensor

    def __init__(
        self,
        model: torch.nn.Module,
        export_config: dict[str, Any],
        output_extractor: Callable[..., torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.input_layout = str(export_config["input_layout"])
        self.input_key = str(export_config["input_key"])
        self.output_extractor = output_extractor or (lambda out: out.logits)

        self.target_resize = export_config.get("target_resize")
        self.target_crop = export_config.get("target_crop")
        self.target_normalization = export_config.get("target_normalization")

        source_normalization = export_config["source_normalization"]
        self._needs_renorm = self.target_normalization is not None
        self._needs_resize = self.target_resize is not None
        self._needs_crop = self.target_crop is not None

        self.register_buffer(
            "src_mean",
            torch.tensor(source_normalization["mean"], dtype=torch.float32).view(1, 3, 1, 1, 1),
        )
        self.register_buffer(
            "src_std",
            torch.tensor(source_normalization["std"], dtype=torch.float32).view(1, 3, 1, 1, 1),
        )

        dst_mean = [0.0, 0.0, 0.0]
        dst_std = [1.0, 1.0, 1.0]
        if self.target_normalization is not None:
            dst_mean = self.target_normalization["mean"]
            dst_std = self.target_normalization["std"]

        self.register_buffer(
            "dst_mean",
            torch.tensor(dst_mean, dtype=torch.float32).view(1, 3, 1, 1, 1),
        )
        self.register_buffer(
            "dst_std",
            torch.tensor(dst_std, dtype=torch.float32).view(1, 3, 1, 1, 1),
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Adapt competition input and forward through the wrapped model.

        Parameters
        ----------
        video
            Tensor of shape ``(B, T, H, W, C)`` — R(2+1)D normalized, 112x112,
            channel-last layout as delivered by the competition evaluation
            pipeline.

        Returns
        -------
        torch.Tensor
            Classification logits.
        """
        x = video.permute(0, 4, 1, 2, 3)

        if self._needs_renorm:
            x = x * self.src_std + self.src_mean

        if self._needs_resize:
            target_resize = self.target_resize
            assert target_resize is not None
            b, c, t, h, w = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            x = torch.nn.functional.interpolate(
                x,
                size=(target_resize["height"], target_resize["width"]),
                mode="bilinear",
                align_corners=False,
            )
            x = x.reshape(b, t, c, target_resize["height"], target_resize["width"])
            x = x.permute(0, 2, 1, 3, 4)

        if self._needs_crop:
            target_crop = self.target_crop
            assert target_crop is not None
            crop_h = int(target_crop["height"])
            crop_w = int(target_crop["width"])
            _, _, _, h, w = x.shape
            top = (h - crop_h) // 2
            left = (w - crop_w) // 2
            x = x[:, :, :, top : top + crop_h, left : left + crop_w]

        if self._needs_renorm:
            x = (x - self.dst_mean) / self.dst_std

        if self.input_layout == "BTCHW":
            x = x.permute(0, 2, 1, 3, 4)

        return self.output_extractor(self.model(**{self.input_key: x}))

    @classmethod
    def from_export_config(
        cls,
        model: torch.nn.Module,
        export_config: dict[str, Any],
        output_extractor: Callable[..., torch.Tensor] | None = None,
    ) -> CompetitionAdapter:
        """Build an adapter from an explicit export config."""
        return cls(
            model=model,
            export_config=export_config,
            output_extractor=output_extractor,
        )


def _resolve_export_num_frames(
    saved_num_frames: int | None,
    requested_num_frames: int | None,
    *,
    force_override: bool,
) -> int:
    """Resolve export/compile frame count against saved artifact metadata."""
    if requested_num_frames is None:
        if saved_num_frames is None:
            raise ValueError("Could not infer num_frames from saved metadata. Pass --num-frames.")
        return saved_num_frames
    if (
        saved_num_frames is not None
        and requested_num_frames != saved_num_frames
        and not force_override
    ):
        raise ValueError(
            f"Requested num_frames={requested_num_frames} does not match saved "
            f"artifact num_frames={saved_num_frames}. "
            "Pass --force-override to ignore the saved metadata."
        )
    return requested_num_frames


def _load_checkpoint_export_config(
    model_path: Path,
    *,
    model_type: str | None = None,
    num_frames: int | None = None,
    force_override: bool = False,
) -> tuple[str, Any, dict[str, Any], int]:
    """Load registry metadata and export config from a saved checkpoint directory."""
    from lpcv.models import (
        EXPORT_CONFIG_FILENAME,
        MODEL_CONFIG_FILENAME,
        VAL_TRANSFORM_FILENAME,
        get_model_spec,
        load_model_config,
        resolve_artifact_model_name,
        resolve_model_config,
    )
    from lpcv.transforms import build_export_config, load_export_config, load_val_transform_config

    resolved_model_type = resolve_artifact_model_name(
        model_path,
        model_name=model_type,
        force_override=force_override,
    )
    spec = get_model_spec(resolved_model_type)

    raw_model_cfg: dict[str, Any] = {}
    if (model_path / MODEL_CONFIG_FILENAME).is_file():
        raw_model_cfg = load_model_config(model_path)
    resolved_model = resolve_model_config(resolved_model_type, raw_model_cfg)

    export_path = model_path / EXPORT_CONFIG_FILENAME
    if export_path.is_file():
        export_config = load_export_config(export_path)
    else:
        val_config_path = model_path / VAL_TRANSFORM_FILENAME
        if val_config_path.is_file():
            val_config = load_val_transform_config(val_config_path)
        else:
            val_config = resolved_model.val_preset
        export_config = build_export_config(
            val_config,
            input_layout=spec.input_layout,
            input_key=spec.input_key,
            num_frames=resolved_model.num_frames,
        )

    resolved_num_frames = _resolve_export_num_frames(
        export_config.get("num_frames"),
        num_frames,
        force_override=force_override,
    )
    export_config["num_frames"] = resolved_num_frames
    return resolved_model_type, spec, export_config, resolved_num_frames


def _infer_num_frames_from_onnx(model_path: Path) -> int | None:
    """Read ``num_frames`` from the ONNX ``video`` input shape (BTHWC)."""
    onnx_file: Path | None = None
    if model_path.is_dir():
        candidates = list(model_path.glob("*.onnx"))
        if candidates:
            onnx_file = candidates[0]
    elif model_path.suffix == ".onnx":
        onnx_file = model_path

    if onnx_file is None or not onnx_file.is_file():
        return None

    model_proto = onnx.load(str(onnx_file), load_external_data=False)
    for inp in model_proto.graph.input:
        if inp.name == COMPETITION_INPUT_NAME:
            dims = inp.type.tensor_type.shape.dim
            if len(dims) >= 2:
                t_dim = dims[1].dim_value
                if t_dim > 0:
                    return int(t_dim)
    return None


def _resolve_compile_num_frames(
    model_path: Path,
    *,
    num_frames: int | None = None,
    force_override: bool = False,
) -> int:
    """Resolve compile-time frame count from exported ONNX metadata."""
    from lpcv.models import EXPORT_CONFIG_FILENAME
    from lpcv.transforms import load_export_config

    saved_num_frames: int | None = None

    saved_num_frames = _infer_num_frames_from_onnx(model_path)

    if saved_num_frames is None:
        export_path = (
            model_path / EXPORT_CONFIG_FILENAME
            if model_path.is_dir()
            else model_path.parent / EXPORT_CONFIG_FILENAME
        )
        if export_path.is_file():
            saved_num_frames = int(load_export_config(export_path)["num_frames"])

    return _resolve_export_num_frames(saved_num_frames, num_frames, force_override=force_override)


def export_onnx(
    model_path: str | Path,
    output_path: str | Path,
    model_type: str | None = None,
    num_frames: int | None = None,
    opset_version: int = 18,
    dynamo: bool = False,
    decompose: bool = True,
    force_override: bool = False,
) -> Path:
    """Wrap a trained checkpoint with the saved adapter contract and export to ONNX."""
    model_path = Path(model_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    resolved_model_type, spec, export_config, resolved_num_frames = _load_checkpoint_export_config(
        model_path,
        model_type=model_type,
        num_frames=num_frames,
        force_override=force_override,
    )

    logger.info(f"Loading {resolved_model_type} model from {model_path}")
    base_model = spec.loader(str(model_path))

    wrapped = CompetitionAdapter.from_export_config(
        model=base_model,
        export_config=export_config,
        output_extractor=spec.output_extractor,
    )
    wrapped.eval()

    dummy_input = torch.randn(
        1,
        resolved_num_frames,
        COMPETITION_SPATIAL_SIZE,
        COMPETITION_SPATIAL_SIZE,
        3,
    )

    logger.info(f"Exporting ONNX (opset {opset_version}, dynamo={dynamo}) → {output_path}")

    if decompose:
        from lpcv.models.base import decompose_depthwise_conv3d

        decompose_depthwise_conv3d(wrapped)

    onnx_dir = output_path.parent / f"{output_path.stem}.onnx"
    if onnx_dir.exists():
        shutil.rmtree(onnx_dir)
    onnx_dir.mkdir(parents=True)

    inner_onnx = onnx_dir / f"{output_path.stem}.onnx"
    data_file = f"{output_path.stem}.data"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_onnx = Path(tmp) / "model.onnx"
        torch.onnx.export(
            wrapped,
            (dummy_input,),
            str(tmp_onnx),
            input_names=[COMPETITION_INPUT_NAME],
            output_names=["logits"],
            dynamic_axes={
                COMPETITION_INPUT_NAME: {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            opset_version=opset_version,
            dynamo=dynamo,
        )
        model_proto = onnx.load(str(tmp_onnx))

    onnx.save(
        model_proto,
        str(inner_onnx),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_file,
    )

    logger.info(f"ONNX model directory saved to {onnx_dir}")
    return onnx_dir


def profile_on_hub(
    model_path: str | Path | None = None,
    device_name: str = DEFAULT_DEVICE_NAME,
    hub_model_id: str | None = None,
    name: str | None = None,
) -> str:
    """Submit a profile job for a compiled model on Qualcomm AI Hub.

    Provide either *model_path* (a local ``.bin`` file) or *hub_model_id*
    (an already-uploaded AI Hub model to reuse).

    Parameters
    ----------
    model_path
        Path to the compiled ``.bin`` model (input shape is already baked in).
    device_name
        AI Hub device name (e.g. ``"Dragonwing IQ-9075 EVK"``).
    hub_model_id
        Reuse an already-uploaded AI Hub model instead of uploading a local file.
    name
        Optional job name on AI Hub.  Auto-generated when ``None``.

    Returns
    -------
    str
        URL of the completed profile job.

    Raises
    ------
    RuntimeError
        If the profile job fails on AI Hub.
    ValueError
        If neither *model_path* nor *hub_model_id* is provided.
    """
    import qai_hub

    device = qai_hub.Device(device_name)

    if hub_model_id is not None:
        logger.info(f"Reusing AI Hub model: {hub_model_id}")
        model: str | qai_hub.Model = qai_hub.get_model(hub_model_id)
        job_name = name or f"profile-{hub_model_id}"
    elif model_path is not None:
        model_path = Path(model_path)
        logger.info(f"Uploading {model_path.name} and submitting profile job on {device_name}")
        model = str(model_path)
        job_name = name or f"profile-{model_path.stem}"
    else:
        raise ValueError("Provide either model_path or hub_model_id")

    profile_job = qai_hub.submit_profile_job(
        model=model,
        device=device,
        name=job_name,
    )

    status = profile_job.wait()
    if not status.success:
        raise RuntimeError(f"Profile job failed: {profile_job.url}")
    logger.info(f"Profile job succeeded: {profile_job.url}")
    return profile_job.url


def compile_on_hub(
    model_path: str | Path,
    device_name: str = DEFAULT_DEVICE_NAME,
    num_frames: int | None = None,
    output_dir: str | Path | None = None,
    download: bool = True,
    hub_model_id: str | None = None,
    name: str | None = None,
    force_override: bool = False,
) -> Path | str:
    """Compile an ONNX model on Qualcomm AI Hub and download the binary.

    Parameters
    ----------
    model_path
        Path to the ONNX model file.  Ignored when *hub_model_id* is provided.
    device_name
        AI Hub device name (e.g. ``"Dragonwing IQ-9075 EVK"``).
    num_frames
        Temporal dimension for input spec. When omitted, infer it from the
        exported ONNX metadata.
    output_dir
        Directory to save the compiled ``.bin``. Defaults to ``./export_assets/``.
    download
        If ``True`` (default), download the compiled binary locally.
        If ``False``, skip the download and return the AI Hub model ID instead.
    hub_model_id
        If provided, reuse an already-uploaded AI Hub model instead of
        uploading *model_path*.
    name
        Optional job name on AI Hub.  Auto-generated when ``None``.

    Returns
    -------
    Path or str
        Path to the downloaded compiled binary when *download* is ``True``,
        or the AI Hub model ID string when *download* is ``False``.
    """
    import qai_hub

    model_path = Path(model_path)

    device = qai_hub.Device(device_name)
    resolved_num_frames = _resolve_compile_num_frames(
        model_path,
        num_frames=num_frames,
        force_override=force_override,
    )
    input_shape = (
        1,
        resolved_num_frames,
        COMPETITION_SPATIAL_SIZE,
        COMPETITION_SPATIAL_SIZE,
        3,
    )

    if hub_model_id is not None:
        logger.info(f"Reusing AI Hub model: {hub_model_id}")
        model: str | qai_hub.Model = qai_hub.get_model(hub_model_id)
    else:
        logger.info(f"Uploading {model_path.name} and submitting compile job on {device_name}")
        model = str(model_path)

    compile_job = qai_hub.submit_compile_job(
        model=model,
        input_specs={COMPETITION_INPUT_NAME: (input_shape, "float32")},
        device=device,
        options="--target_runtime qnn_context_binary",
        name=name or model_path.stem,
    )

    assert compile_job.wait().success, f"Compile job failed: {compile_job.url}"
    logger.info(f"Compile job succeeded: {compile_job.url}")

    target_model = compile_job.get_target_model()
    assert target_model is not None

    model_id: str = target_model.model_id
    logger.info(f"Compiled model ID: {model_id}")

    if not download:
        return model_id

    output_dir = Path(output_dir) if output_dir else Path.cwd() / "export_assets"
    output_dir.mkdir(parents=True, exist_ok=True)

    bin_path = Path(target_model.download(str(output_dir / f"{model_path.stem}.bin")))
    logger.info(f"Compiled binary saved to {bin_path}")
    return bin_path


def validate_on_hub(
    model_type: str = "r2plus1d",
    num_classes: int = 92,
    device_name: str = DEFAULT_DEVICE_NAME,
    opset_version: int = 18,
    dynamo: bool = False,
    decompose: bool = True,
    name: str | None = None,
    model_config: dict[str, Any] | None = None,
) -> str:
    """Instantiate a model, export to ONNX, compile on AI Hub, and submit a profile job.

    Creates a throwaway instance of the specified model architecture with random
    weights, wraps it with the competition adapter, exports to ONNX, compiles
    on AI Hub (without downloading the binary), and finally submits a profile
    job on the compiled model.

    This is a quick end-to-end smoke test to verify that a model architecture
    can pass through the entire submission pipeline on real hardware.

    Parameters
    ----------
    model_type
        Registered model name (e.g. ``"r2plus1d"``, ``"x3d"``, ``"mvitv2"``).
    num_classes
        Number of output classes for the throwaway model.
    device_name
        Qualcomm AI Hub device name.
    opset_version
        ONNX opset version.
    dynamo
        Use the ``torch.export``-based ONNX exporter instead of TorchScript.
        Eliminates dynamic control-flow ops (``Loop``, ``If``, ``SequenceEmpty``)
        that some backends do not support.
    name
        Optional job name prefix on AI Hub.  Auto-generated when ``None``.

    Returns
    -------
    str
        URL of the completed profile job.

    Raises
    ------
    KeyError
        If *model_type* is not registered.
    RuntimeError
        If the compile or profile job fails on AI Hub.
    """
    import qai_hub

    from lpcv.models import (
        EXPORT_CONFIG_FILENAME,
        VAL_TRANSFORM_FILENAME,
        get_model_spec,
        resolve_model_config,
    )
    from lpcv.transforms import build_export_config, save_export_config, save_val_transform_config

    resolved_model = resolve_model_config(model_type, model_config or {"model": model_type})
    spec = get_model_spec(model_type)
    job_prefix = name or f"validate-{model_type}"

    if spec.throwaway_builder is None:
        raise ValueError(f"Model {model_type!r} does not have a throwaway_builder registered")

    logger.info(f"Building throwaway {model_type} model ({num_classes} classes)")
    base_model = spec.throwaway_builder(
        num_classes,
        **{
            k: v
            for k, v in resolved_model.model_config.items()
            if k not in ("model", "num_classes")
        },
    )

    device = qai_hub.Device(device_name)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        base_model.save_pretrained(tmp_path)  # type: ignore[attr-defined]

        save_val_transform_config(resolved_model.val_preset, tmp_path / VAL_TRANSFORM_FILENAME)
        export_config = build_export_config(
            resolved_model.val_preset,
            input_layout=spec.input_layout,
            input_key=spec.input_key,
            num_frames=resolved_model.num_frames,
        )
        save_export_config(export_config, tmp_path / EXPORT_CONFIG_FILENAME)

        wrapped = CompetitionAdapter.from_export_config(
            model=base_model,
            export_config=export_config,
            output_extractor=spec.output_extractor,
        )
        wrapped.eval()

        dummy = torch.randn(
            1,
            resolved_model.num_frames,
            COMPETITION_SPATIAL_SIZE,
            COMPETITION_SPATIAL_SIZE,
            3,
        )
        onnx_dir = tmp_path / f"{model_type}.onnx"
        onnx_dir.mkdir()
        inner_onnx = onnx_dir / f"{model_type}.onnx"
        data_file = f"{model_type}.data"

        tmp_export = tmp_path / "tmp_export"
        tmp_export.mkdir()
        tmp_onnx = tmp_export / "model.onnx"

        logger.info(f"Exporting ONNX (opset {opset_version}, dynamo={dynamo})")

        if decompose:
            from lpcv.models.base import decompose_depthwise_conv3d

            decompose_depthwise_conv3d(wrapped)

        torch.onnx.export(
            wrapped,
            (dummy,),
            str(tmp_onnx),
            input_names=[COMPETITION_INPUT_NAME],
            output_names=["logits"],
            dynamic_axes={
                COMPETITION_INPUT_NAME: {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            opset_version=opset_version,
            dynamo=dynamo,
        )
        model_proto = onnx.load(str(tmp_onnx))
        onnx.save(
            model_proto,
            str(inner_onnx),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_file,
        )

        input_shape = (
            1,
            resolved_model.num_frames,
            COMPETITION_SPATIAL_SIZE,
            COMPETITION_SPATIAL_SIZE,
            3,
        )

        logger.info(f"Submitting compile job on {device_name}")
        compile_job = qai_hub.submit_compile_job(
            model=str(onnx_dir),
            input_specs={COMPETITION_INPUT_NAME: (input_shape, "float32")},
            device=device,
            options="--target_runtime qnn_context_binary",
            name=f"{job_prefix}-compile",
        )

        assert compile_job.wait().success, f"Compile job failed: {compile_job.url}"
        logger.info(f"Compile job succeeded: {compile_job.url}")

        target_model = compile_job.get_target_model()
        assert target_model is not None
        model_id: str = target_model.model_id
        logger.info(f"Compiled model ID: {model_id}")

    logger.info("Submitting profile job")
    profile_job = qai_hub.submit_profile_job(
        model=qai_hub.get_model(model_id),
        device=device,
        name=f"{job_prefix}-profile",
    )

    status = profile_job.wait()
    if not status.success:
        raise RuntimeError(f"Profile job failed: {profile_job.url}")
    logger.info(f"Profile job succeeded: {profile_job.url}")
    return profile_job.url


def inference_on_hub(
    compiled_model_path: str | Path,
    tensor_dir: str | Path,
    manifest_path: str | Path,
    output_h5: str | Path = "dataset-export.h5",
    device_name: str = DEFAULT_DEVICE_NAME,
    channel_last: bool = True,
    hub_model_id: str | None = None,
    name: str | None = None,
) -> Path:
    """Upload preprocessed tensors and run on-device inference via AI Hub.

    Parameters
    ----------
    compiled_model_path
        Path to the compiled ``.bin`` model. Ignored when *hub_model_id* is provided.
    tensor_dir
        Directory containing ``.npy`` tensors (class-subfolder layout matching the manifest).
    manifest_path
        Path to ``manifest.jsonl`` produced by :func:`preprocess_dataset`.
    output_h5
        Output HDF5 file path for collected logits.
    device_name
        AI Hub device name.
    channel_last
        If ``True`` (default), keep tensors in ``NTHWC`` layout for upload.
        Set ``False`` only for legacy ``NCTHW`` tensors.
    hub_model_id
        If provided, reuse an already-uploaded AI Hub model instead of uploading
        *compiled_model_path* again.
    name
        Optional job name prefix on AI Hub.  Auto-generated when ``None``.

    Returns
    -------
    Path
        Path to the written HDF5 logits file.
    """
    import h5py
    import qai_hub

    compiled_model_path = Path(compiled_model_path)
    tensor_dir = Path(tensor_dir)
    output_h5 = Path(output_h5)

    with open(manifest_path, encoding="utf-8") as f:
        manifest_entries = [json.loads(line) for line in f if line.strip()]

    npy_paths: list[Path] = []
    for entry in manifest_entries:
        p = Path(entry["tensor_path"])
        if not p.is_file():
            raise FileNotFoundError(f"Tensor file not found: {p}")
        npy_paths.append(p)

    logger.info(f"Loading {len(npy_paths)} tensors from manifest")
    tensors: list[np.ndarray] = []
    for p in tqdm(npy_paths, desc="Loading tensors", unit="file"):
        x = np.load(str(p)).astype(np.float32)
        if channel_last and x.ndim == 5:
            x = np.transpose(x, (0, 2, 3, 4, 1))
        tensors.append(x)

    device = qai_hub.Device(device_name)

    if hub_model_id is not None:
        logger.info(f"Reusing AI Hub model: {hub_model_id}")
        target_model = qai_hub.get_model(hub_model_id)
    else:
        logger.info(f"Uploading compiled model: {compiled_model_path.name}")
        target_model = qai_hub.upload_model(str(compiled_model_path))

    all_jobs = []
    input_name = COMPETITION_INPUT_NAME

    logger.info(f"Submitting inference in chunks of {CHUNK_SIZE}")
    for i in range(0, len(tensors), CHUNK_SIZE):
        chunk = tensors[i : i + CHUNK_SIZE]
        dataset = qai_hub.upload_dataset(
            {input_name: chunk},
            name=name or f"lpcv_inference_part_{i // CHUNK_SIZE + 1}",
        )
        job = qai_hub.submit_inference_job(
            model=target_model,
            device=device,
            inputs=dataset,
            options="",
        )
        logger.info(f"Chunk {i // CHUNK_SIZE + 1} job: {job.job_id}")
        all_jobs.append(job)

    logger.info("Waiting for inference jobs...")
    combined_logits: list[np.ndarray] = []
    for job in all_jobs:
        job.wait()
        output_data = job.download_output_data()
        key = next(iter(output_data))
        for arr in output_data[key]:
            arr = np.asarray(arr)
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]
            combined_logits.append(arr)

    logger.info(f"Collected {len(combined_logits)} results, writing to {output_h5}")

    with h5py.File(str(output_h5), "w") as f:
        grp = f.create_group("data/0")
        for i, arr in enumerate(combined_logits):
            grp.create_dataset(f"batch_{i}", data=arr)

    logger.info(f"Inference logits saved to {output_h5}")
    return output_h5
