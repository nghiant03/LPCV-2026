"""Submission pipeline — preprocess, export, on-device inference via Qualcomm AI Hub.

Provides:

- :func:`preprocess_dataset` — decode videos to ``.npy`` tensors + ``manifest.jsonl``.
- :func:`export_onnx` — wrap model with competition adapter and export to ONNX.
- :func:`compile_on_hub` — compile an ONNX model on Qualcomm AI Hub.
- :func:`run_inference_on_hub` — upload tensors and run on-device inference.
- :func:`register_model` — register a new model type for the submission pipeline.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import onnx
import torch
import torchvision.transforms as T
from loguru import logger
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Callable

DEFAULT_NUM_FRAMES = 16
"""Default number of frames to sample per video clip."""

COMPETITION_SPATIAL_SIZE = 112
"""Spatial resolution of the competition's fixed input pipeline."""

COMPETITION_RESIZE_HW = (128, 171)
"""Resize target before center crop in the competition pipeline."""

DEFAULT_DEVICE_NAME = "Dragonwing IQ-9075 EVK"
"""Default Qualcomm AI Hub device for compilation and inference."""

CHUNK_SIZE = 538
"""Maximum number of samples per AI Hub inference job (stays under 2 GB flatbuffer limit)."""

FRAME_RATE = 4
"""Frame rate used by the competition's VideoClips-based preprocessing."""


@dataclass
class SubmissionModelConfig:
    """Describes how a model integrates with the competition submission pipeline.

    Parameters
    ----------
    spatial_size
        Spatial resolution expected by the model (e.g. 224 for VideoMAE).
    mean
        Per-channel mean for normalisation (length 3).
    std
        Per-channel standard deviation for normalisation (length 3).
    input_layout
        Expected tensor layout: ``"BTCHW"`` or ``"BCTHW"``.
    input_key
        Keyword argument name for the model's forward method (e.g. ``"pixel_values"``).
    output_extractor
        Callable that extracts logits from the model's raw output.
        Defaults to ``lambda out: out.logits``.
    loader
        Callable ``(path: str) -> torch.nn.Module`` that loads a checkpoint.
    """

    spatial_size: int
    mean: list[float]
    std: list[float]
    input_layout: str = "BTCHW"
    input_key: str = "pixel_values"
    output_extractor: Callable[..., torch.Tensor] = field(
        default_factory=lambda: lambda out: out.logits
    )
    loader: Callable[[str], torch.nn.Module] | None = None


_MODEL_REGISTRY: dict[str, SubmissionModelConfig] = {}
"""Registry of model configs by name."""


def register_model(name: str, config: SubmissionModelConfig) -> None:
    """Register a model configuration for the submission pipeline.

    Parameters
    ----------
    name
        Short name used to select the model (e.g. ``"videomae"``).
    config
        Configuration describing how to load and run the model.
    """
    _MODEL_REGISTRY[name] = config


def get_model_config(name: str) -> SubmissionModelConfig:
    """Retrieve a registered model configuration by name.

    Parameters
    ----------
    name
        Registered model name.

    Returns
    -------
    SubmissionModelConfig
        The configuration for the requested model.

    Raises
    ------
    KeyError
        If the model name is not registered.
    """
    if name not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown model type {name!r}. Available: {available}")
    return _MODEL_REGISTRY[name]


def _register_builtins() -> None:
    """Register built-in model types."""
    from lpcv.datasets.info import IMAGENET_MEAN, IMAGENET_STD

    def _load_videomae(path: str) -> torch.nn.Module:
        from transformers import VideoMAEForVideoClassification

        return VideoMAEForVideoClassification.from_pretrained(path)

    register_model(
        "videomae",
        SubmissionModelConfig(
            spatial_size=224,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            input_layout="BTCHW",
            input_key="pixel_values",
            output_extractor=lambda out: out.logits,
            loader=_load_videomae,
        ),
    )


_register_builtins()


def preprocess_dataset(
    data_dir: str | Path,
    output_dir: str | Path,
    num_frames: int = DEFAULT_NUM_FRAMES,
    decoder_name: str = "pyav",
    target_fps: int = FRAME_RATE,
) -> Path:
    """Decode videos to ``(1, 3, T, 112, 112)`` ``.npy`` tensors and write a manifest.

    Replicates the competition's exact preprocessing pipeline so that locally
    saved tensors match what the organiser's evaluation server produces:

    1. Decode frames, resample to *target_fps* with dynamic adjustment
       for short videos (matching the patched ``VideoClips`` behaviour).
    2. ``ConvertImageDtype(float32)`` → ``Resize(128, 171)``
       → ``Normalize(R2+1D mean/std)`` → ``CenterCrop(112, 112)``.
    3. Permute to ``(C, T, H, W)`` and add batch dim → ``(1, 3, T, 112, 112)``.

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
    from lpcv.datasets.info import (
        R2PLUS1D_MEAN,
        R2PLUS1D_STD,
        TARGET_LABEL_FILE_NAME,
        VIDEO_EXTENSIONS,
    )

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

    spatial = T.Compose(
        [
            T.ConvertImageDtype(torch.float32),
            T.Resize(COMPETITION_RESIZE_HW, antialias=False),
            T.Normalize(mean=R2PLUS1D_MEAN, std=R2PLUS1D_STD),
            T.CenterCrop(COMPETITION_SPATIAL_SIZE),
        ]
    )

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
            tensor = clip.permute(1, 0, 2, 3).unsqueeze(0)

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
    """Adapter that converts competition input to any video model's expected format.

    The competition feeds ``(B, C, T, H, W)`` tensors at 112x112, normalized
    with R(2+1)D mean/std.  Different models expect different spatial sizes,
    normalization stats, and input layouts.

    This wrapper performs (all in-graph for ONNX export):

    1. **Denormalize** — undo the R(2+1)D mean/std.
    2. **Resize** — bilinear interpolate from 112x112 to the model's spatial size.
    3. **Renormalize** — apply the model's expected mean/std.
    4. **Permute** — rearrange to the model's expected input layout.
    5. **Forward** — through the inner model.
    """

    src_mean: torch.Tensor
    src_std: torch.Tensor
    dst_mean: torch.Tensor
    dst_std: torch.Tensor

    def __init__(
        self,
        model: torch.nn.Module,
        src_mean: list[float],
        src_std: list[float],
        dst_mean: list[float],
        dst_std: list[float],
        target_spatial: int = 224,
        input_layout: str = "BTCHW",
        input_key: str = "pixel_values",
        output_extractor: Callable[..., torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.target_spatial = target_spatial
        self.input_layout = input_layout
        self.input_key = input_key
        self.output_extractor = output_extractor or (lambda out: out.logits)
        self.register_buffer("src_mean", torch.tensor(src_mean).view(1, 3, 1, 1, 1))
        self.register_buffer("src_std", torch.tensor(src_std).view(1, 3, 1, 1, 1))
        self.register_buffer("dst_mean", torch.tensor(dst_mean).view(1, 3, 1, 1, 1))
        self.register_buffer("dst_std", torch.tensor(dst_std).view(1, 3, 1, 1, 1))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Adapt competition input and forward through the wrapped model.

        Parameters
        ----------
        pixel_values
            Tensor of shape ``(B, C, T, H, W)`` — R(2+1)D normalized, 112x112.

        Returns
        -------
        torch.Tensor
            Classification logits.
        """
        x = pixel_values * self.src_std + self.src_mean

        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = torch.nn.functional.interpolate(
            x,
            size=(self.target_spatial, self.target_spatial),
            mode="bilinear",
            align_corners=False,
        )
        x = x.reshape(b, t, c, self.target_spatial, self.target_spatial)
        x = x.permute(0, 2, 1, 3, 4)

        x = (x - self.dst_mean) / self.dst_std

        if self.input_layout == "BTCHW":
            x = x.permute(0, 2, 1, 3, 4)

        return self.output_extractor(self.model(**{self.input_key: x}))

    @classmethod
    def from_config(
        cls,
        model: torch.nn.Module,
        config: SubmissionModelConfig,
        src_mean: list[float] | None = None,
        src_std: list[float] | None = None,
    ) -> CompetitionAdapter:
        """Create an adapter from a :class:`SubmissionModelConfig`.

        Parameters
        ----------
        model
            The inner model to wrap.
        config
            Model configuration from the registry.
        src_mean
            Source normalization mean (competition pipeline). If ``None``,
            uses R(2+1)D mean from ``lpcv.datasets.info``.
        src_std
            Source normalization std (competition pipeline). If ``None``,
            uses R(2+1)D std from ``lpcv.datasets.info``.

        Returns
        -------
        CompetitionAdapter
            Configured adapter instance.
        """
        from lpcv.datasets.info import R2PLUS1D_MEAN, R2PLUS1D_STD

        return cls(
            model=model,
            src_mean=src_mean or R2PLUS1D_MEAN,
            src_std=src_std or R2PLUS1D_STD,
            dst_mean=config.mean,
            dst_std=config.std,
            target_spatial=config.spatial_size,
            input_layout=config.input_layout,
            input_key=config.input_key,
            output_extractor=config.output_extractor,
        )


def export_onnx(
    model_path: str | Path,
    output_path: str | Path,
    model_type: str = "videomae",
    num_frames: int = DEFAULT_NUM_FRAMES,
    opset_version: int = 18,
) -> Path:
    """Wrap a trained checkpoint with the competition adapter and export to ONNX.

    The exported model accepts the competition's fixed input shape
    ``(1, 3, T, 112, 112)`` (R(2+1)D-normalised) and internally
    denormalises, resizes to the model's expected spatial resolution,
    re-normalises, permutes, then forwards through the model.

    Parameters
    ----------
    model_path
        Path to a saved model checkpoint directory.
    output_path
        File path for the exported ``.onnx`` model.
    model_type
        Registered model name (e.g. ``"videomae"``).
    num_frames
        Temporal dimension of the input tensor.
    opset_version
        ONNX opset version.

    Returns
    -------
    Path
        Path to the ONNX model directory (AI Hub format).

    Raises
    ------
    KeyError
        If *model_type* is not registered.
    ValueError
        If the model config has no loader.
    """
    model_path = Path(model_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = get_model_config(model_type)
    if config.loader is None:
        raise ValueError(
            f"Model type {model_type!r} has no loader. "
            f"Register one via register_model() with a loader callback."
        )

    logger.info(f"Loading {model_type} model from {model_path}")
    base_model = config.loader(str(model_path))
    wrapped = CompetitionAdapter.from_config(base_model, config)
    wrapped.eval()

    dummy_input = torch.randn(1, 3, num_frames, COMPETITION_SPATIAL_SIZE, COMPETITION_SPATIAL_SIZE)

    logger.info(f"Exporting ONNX (opset {opset_version}) → {output_path}")

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
            input_names=["pixel_values"],
            output_names=["logits"],
            dynamic_axes={
                "pixel_values": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            opset_version=opset_version,
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


def compile_on_hub(
    model_path: str | Path,
    device_name: str = DEFAULT_DEVICE_NAME,
    num_frames: int = DEFAULT_NUM_FRAMES,
    output_dir: str | Path | None = None,
    download: bool = True,
) -> Path | str:
    """Compile an ONNX model on Qualcomm AI Hub and download the binary.

    Parameters
    ----------
    model_path
        Path to the ONNX model file.
    device_name
        AI Hub device name (e.g. ``"Dragonwing IQ-9075 EVK"``).
    num_frames
        Temporal dimension for input spec.
    output_dir
        Directory to save the compiled ``.bin``. Defaults to ``./export_assets/``.
    download
        If ``True`` (default), download the compiled binary locally.
        If ``False``, skip the download and return the AI Hub model ID instead.

    Returns
    -------
    Path or str
        Path to the downloaded compiled binary when *download* is ``True``,
        or the AI Hub model ID string when *download* is ``False``.
    """
    import qai_hub

    model_path = Path(model_path)

    device = qai_hub.Device(device_name)
    input_shape = (1, 3, num_frames, COMPETITION_SPATIAL_SIZE, COMPETITION_SPATIAL_SIZE)

    logger.info(f"Uploading {model_path.name} and submitting compile job on {device_name}")
    compile_job = qai_hub.submit_compile_job(
        model=str(model_path),
        input_specs={"pixel_values": (input_shape, "float32")},
        device=device,
        options="--target_runtime qnn_context_binary",
        name=model_path.stem,
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


def run_inference_on_hub(
    compiled_model_path: str | Path,
    tensor_dir: str | Path,
    manifest_path: str | Path,
    output_h5: str | Path = "dataset-export.h5",
    device_name: str = DEFAULT_DEVICE_NAME,
    channel_last: bool = False,
    hub_model_id: str | None = None,
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
        If ``True``, transpose tensors from ``NCTHW`` to ``NTHWC`` before upload.
    hub_model_id
        If provided, reuse an already-uploaded AI Hub model instead of uploading
        *compiled_model_path* again.

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
    input_name = "pixel_values"

    logger.info(f"Submitting inference in chunks of {CHUNK_SIZE}")
    for i in range(0, len(tensors), CHUNK_SIZE):
        chunk = tensors[i : i + CHUNK_SIZE]
        dataset = qai_hub.upload_dataset(
            {input_name: chunk},
            name=f"lpcv_inference_part_{i // CHUNK_SIZE + 1}",
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
