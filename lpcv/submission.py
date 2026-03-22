"""Submission pipeline — preprocess, export, on-device inference via Qualcomm AI Hub.

Provides:

- :func:`preprocess_dataset` — decode videos to ``.npy`` tensors + ``manifest.jsonl``.
- :func:`export_onnx` — trace a trained VideoMAE checkpoint to ONNX.
- :func:`compile_on_hub` — compile an ONNX model on Qualcomm AI Hub.
- :func:`run_inference_on_hub` — upload tensors and run on-device inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from lpcv.datasets.info import IMAGENET_MEAN, IMAGENET_STD

if TYPE_CHECKING:
    from lpcv.datasets.decoder import VideoDecoder

DEFAULT_NUM_FRAMES = 16
"""Default number of frames to sample per video clip."""

DEFAULT_SPATIAL_SIZE = 224
"""Default spatial resolution for VideoMAE input."""

DEFAULT_DEVICE_NAME = "Dragonwing IQ-9075 EVK"
"""Default Qualcomm AI Hub device for compilation and inference."""

CHUNK_SIZE = 538
"""Maximum number of samples per AI Hub inference job (stays under 2 GB flatbuffer limit)."""


def preprocess_dataset(
    data_dir: str | Path,
    output_dir: str | Path,
    decoder: VideoDecoder,
    num_frames: int = DEFAULT_NUM_FRAMES,
    spatial_size: int = DEFAULT_SPATIAL_SIZE,
) -> Path:
    """Decode videos to ``(1, C, T, H, W)`` ``.npy`` tensors and write a manifest.

    Parameters
    ----------
    data_dir
        Root of the videofolder dataset (expects ``val/<class>/*.mp4``).
    output_dir
        Directory to write ``.npy`` files and ``manifest.jsonl``.
    decoder
        Video decoder instance used to extract frames.
    num_frames
        Number of frames to uniformly sample per clip.
    spatial_size
        Target spatial size (height and width) after resizing.

    Returns
    -------
    Path
        Path to the written ``manifest.jsonl``.
    """
    from lpcv.datasets.info import TARGET_LABEL_FILE_NAME, VIDEO_EXTENSIONS

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

    manifest_path = output_dir / "manifest.jsonl"
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

    logger.info(f"Preprocessing {len(video_entries)} videos → {output_dir}")

    with open(manifest_path, "w", encoding="utf-8") as mf:
        for video_path, label in tqdm(video_entries, desc="Preprocessing", unit="video"):
            frames = decoder.decode(video_path, num_frames)

            frames = frames.float() / 255.0
            frames = torch.nn.functional.interpolate(
                frames, size=(spatial_size, spatial_size), mode="bilinear", align_corners=False
            )
            frames = (frames - mean) / std
            tensor = frames.unsqueeze(0)

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


def export_onnx(
    model_path: str | Path,
    output_path: str | Path,
    num_frames: int = DEFAULT_NUM_FRAMES,
    spatial_size: int = DEFAULT_SPATIAL_SIZE,
    opset_version: int = 17,
) -> Path:
    """Trace a trained VideoMAE checkpoint and export to ONNX.

    Parameters
    ----------
    model_path
        Path to a HuggingFace-saved VideoMAE checkpoint directory.
    output_path
        File path for the exported ``.onnx`` model.
    num_frames
        Temporal dimension of the input tensor.
    spatial_size
        Spatial height/width of the input tensor.
    opset_version
        ONNX opset version.

    Returns
    -------
    Path
        Path to the written ONNX file.
    """
    from transformers import VideoMAEForVideoClassification

    model_path = Path(model_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model from {model_path}")
    model = VideoMAEForVideoClassification.from_pretrained(str(model_path))
    model.eval()

    dummy_input = torch.randn(1, num_frames, 3, spatial_size, spatial_size)

    logger.info(f"Exporting ONNX (opset {opset_version}) → {output_path}")
    torch.onnx.export(
        model,
        (dummy_input,),
        str(output_path),
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=opset_version,
    )

    logger.info(f"ONNX model saved to {output_path}")
    return output_path


def compile_on_hub(
    model_path: str | Path,
    device_name: str = DEFAULT_DEVICE_NAME,
    num_frames: int = DEFAULT_NUM_FRAMES,
    spatial_size: int = DEFAULT_SPATIAL_SIZE,
    output_dir: str | Path | None = None,
) -> Path | None:
    """Compile an ONNX model on Qualcomm AI Hub and download the binary.

    Parameters
    ----------
    model_path
        Path to the ONNX model file.
    device_name
        AI Hub device name (e.g. ``"Dragonwing IQ-9075 EVK"``).
    num_frames
        Temporal dimension for input spec.
    spatial_size
        Spatial height/width for input spec.
    output_dir
        Directory to save the compiled ``.bin``. Defaults to ``./export_assets/``.

    Returns
    -------
    Path or None
        Path to the downloaded compiled binary, or ``None`` if download was skipped.
    """
    import qai_hub  # type: ignore[import-untyped]

    model_path = Path(model_path)
    output_dir = Path(output_dir) if output_dir else Path.cwd() / "export_assets"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = qai_hub.Device(device_name)

    logger.info(f"Uploading {model_path.name} and submitting compile job on {device_name}")
    compile_job = qai_hub.submit_compile_job(
        model=str(model_path),
        input_specs={"pixel_values": ((1, num_frames, 3, spatial_size, spatial_size), "float32")},
        device=device,
        options="--target_runtime qnn_context_binary",
        name=model_path.stem,
    )

    assert compile_job.wait().success, f"Compile job failed: {compile_job.url}"
    logger.info(f"Compile job succeeded: {compile_job.url}")

    target_model = compile_job.get_target_model()
    assert target_model is not None

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
) -> Path:
    """Upload preprocessed tensors and run on-device inference via AI Hub.

    Parameters
    ----------
    compiled_model_path
        Path to the compiled ``.bin`` model.
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

    Returns
    -------
    Path
        Path to the written HDF5 logits file.
    """
    import h5py
    import qai_hub  # type: ignore[import-untyped]

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
