"""Submission pipeline — preprocess, export, on-device inference via Qualcomm AI Hub.

Provides:

- :func:`preprocess_dataset` — decode videos to ``.npy`` tensors + ``manifest.jsonl``.
- :func:`export_onnx` — wrap model with auto-built adapter and export to ONNX.
- :func:`compile_on_hub` — compile an ONNX model on Qualcomm AI Hub.
- :func:`run_inference_on_hub` — upload tensors and run on-device inference.

The adapter layer is automatically constructed from the ``val_transform.json``
saved alongside each model checkpoint.  Steps that differ from the competition's
fixed pipeline are baked into the ONNX graph.
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
            T.Lambda(lambda x: x / 255.0 if x.max() > 1.0 else x),
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
    with R(2+1)D mean/std.  When the model's validation pipeline differs from
    the competition pipeline, this adapter undoes the competition transforms
    and applies the model's expected transforms — all in-graph for ONNX export.

    When the model's val pipeline matches the competition pipeline exactly
    (i.e. no adapter steps), this module is a thin pass-through that only
    handles the layout permutation and model forward call.
    """

    src_mean: torch.Tensor
    src_std: torch.Tensor
    dst_mean: torch.Tensor
    dst_std: torch.Tensor

    def __init__(
        self,
        model: torch.nn.Module,
        adapter_steps: list[dict[str, Any]],
        input_layout: str = "BCTHW",
        input_key: str = "pixel_values",
        output_extractor: Callable[..., torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.input_layout = input_layout
        self.input_key = input_key
        self.output_extractor = output_extractor or (lambda out: out.logits)

        self._needs_renorm = False
        self._needs_resize = False
        self._target_spatial = COMPETITION_SPATIAL_SIZE

        src_mean = [0.0, 0.0, 0.0]
        src_std = [1.0, 1.0, 1.0]
        dst_mean = [0.0, 0.0, 0.0]
        dst_std = [1.0, 1.0, 1.0]

        for step in adapter_steps:
            name = step["name"]
            if name == "Normalize":
                self._needs_renorm = True
                dst_mean = step["mean"]
                dst_std = step["std"]
            elif name in ("Resize", "CenterCrop", "RandomCrop"):
                self._needs_resize = True
                self._target_spatial = step["height"]

        if self._needs_renorm:
            from lpcv.datasets.info import R2PLUS1D_MEAN, R2PLUS1D_STD

            src_mean = R2PLUS1D_MEAN
            src_std = R2PLUS1D_STD

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
        x = pixel_values

        if self._needs_renorm:
            x = x * self.src_std + self.src_mean

        if self._needs_resize:
            b, c, t, h, w = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            x = torch.nn.functional.interpolate(
                x,
                size=(self._target_spatial, self._target_spatial),
                mode="bilinear",
                align_corners=False,
            )
            x = x.reshape(b, t, c, self._target_spatial, self._target_spatial)
            x = x.permute(0, 2, 1, 3, 4)

        if self._needs_renorm:
            x = (x - self.dst_mean) / self.dst_std

        if self.input_layout == "BTCHW":
            x = x.permute(0, 2, 1, 3, 4)

        return self.output_extractor(self.model(**{self.input_key: x}))

    @classmethod
    def from_saved_config(
        cls,
        model: torch.nn.Module,
        model_path: str | Path,
        input_layout: str = "BCTHW",
        input_key: str = "pixel_values",
        output_extractor: Callable[..., torch.Tensor] | None = None,
    ) -> CompetitionAdapter:
        """Build an adapter from a saved ``val_transform.json``.

        Reads the validation transform config saved alongside the model
        checkpoint, computes which steps differ from the competition
        pipeline, and constructs an adapter that applies only those
        differences in-graph.

        Parameters
        ----------
        model
            The loaded inner model.
        model_path
            Directory containing the checkpoint and ``val_transform.json``.
        input_layout
            Expected tensor layout of the inner model.
        input_key
            Keyword argument name for the model's forward method.
        output_extractor
            Extracts logits from the model's raw output.

        Returns
        -------
        CompetitionAdapter
            Configured adapter instance.
        """
        from lpcv.transforms import extract_adapter_steps, load_val_transform_config

        model_path = Path(model_path)
        config_path = model_path / "val_transform.json"

        if config_path.is_file():
            val_config = load_val_transform_config(config_path)
            adapter_steps = extract_adapter_steps(val_config)
        else:
            logger.warning(
                f"No val_transform.json found in {model_path}; assuming no adapter needed"
            )
            adapter_steps = []

        return cls(
            model=model,
            adapter_steps=adapter_steps,
            input_layout=input_layout,
            input_key=input_key,
            output_extractor=output_extractor,
        )


def export_onnx(
    model_path: str | Path,
    output_path: str | Path,
    model_type: str = "r2plus1d",
    num_frames: int = DEFAULT_NUM_FRAMES,
    opset_version: int = 18,
) -> Path:
    """Wrap a trained checkpoint with the auto-built adapter and export to ONNX.

    The adapter is constructed automatically from the ``val_transform.json``
    saved alongside the model checkpoint.  Only transforms that differ from
    the competition's fixed pipeline are baked into the ONNX graph.

    Parameters
    ----------
    model_path
        Path to a saved model checkpoint directory.
    output_path
        File path for the exported ``.onnx`` model.
    model_type
        Registered model name (e.g. ``"r2plus1d"``, ``"videomae"``).
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
    """
    from lpcv.models import get_model_spec

    model_path = Path(model_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    spec = get_model_spec(model_type)

    logger.info(f"Loading {model_type} model from {model_path}")
    base_model = spec.loader(str(model_path))

    wrapped = CompetitionAdapter.from_saved_config(
        model=base_model,
        model_path=model_path,
        input_layout=spec.input_layout,
        input_key=spec.input_key,
        output_extractor=spec.output_extractor,
    )
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
