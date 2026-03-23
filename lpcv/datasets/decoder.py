"""Video decoder protocol and concrete implementations.

Provides a ``VideoDecoder`` protocol and three backends:

- ``PyAVDecoder`` — CPU decoding via PyAV (most compatible).
- ``TorchCodecCPUDecoder`` — CPU decoding via TorchCodec (seek-based).
- ``TorchCodecNVDECDecoder`` — GPU decoding via TorchCodec + NVDEC.

All decoders support two frame-sampling strategies controlled by the
*target_fps* constructor parameter:

- ``None`` (default) — uniform index spacing across all frames.
- A positive integer — FPS-based resampling matching the LPCVC
  competition's patched ``VideoClips`` behaviour.

Use ``get_decoder`` to instantiate a decoder by name.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    from pathlib import Path


@runtime_checkable
class VideoDecoder(Protocol):
    """Protocol for video decoders that extract sampled frames.

    All implementations return a float tensor of shape ``(T, C, H, W)`` with
    pixel values in ``[0, 255]``.
    """

    def decode(self, path: Path, num_frames: int) -> torch.Tensor:
        """Decode *num_frames* frames from a video file.

        Parameters
        ----------
        path
            Path to the video file.
        num_frames
            Number of frames to sample.

        Returns
        -------
        torch.Tensor
            Float tensor of shape ``(num_frames, 3, H, W)`` in ``[0, 255]``.
        """
        ...


def _fps_resample_indices(
    total: int, original_fps: float, target_fps: int, num_frames: int
) -> list[int]:
    """Compute frame indices using FPS-based resampling.

    Replicates the LPCVC competition's patched ``VideoClips`` logic:
    resample at *target_fps*, increasing the effective rate for short
    videos so that at least *num_frames* can be produced.

    Parameters
    ----------
    total
        Total decoded frame count.
    original_fps
        Native frame rate of the video.
    target_fps
        Desired resampling rate.
    num_frames
        Minimum number of frames required.

    Returns
    -------
    list[int]
        Frame indices into the original decoded frame list.

    Raises
    ------
    ValueError
        If *num_frames* cannot be satisfied.
    """
    effective_fps: float = target_fps
    total_resampled = total * effective_fps / original_fps

    if total_resampled < num_frames:
        video_duration = total / original_fps
        effective_fps = math.ceil(num_frames / video_duration)
        total_resampled = total * effective_fps / original_fps

    step = original_fps / effective_fps
    indices = [min(int(i * step), total - 1) for i in range(int(math.floor(total_resampled)))]

    if len(indices) < num_frames:
        raise ValueError(
            f"Cannot form {num_frames}-frame clip ({total} frames at {original_fps:.1f} fps)"
        )

    return indices[:num_frames]


def _select_indices(
    total: int, num_frames: int, original_fps: float, target_fps: int | None
) -> list[int]:
    """Choose frame indices using either uniform spacing or FPS-based resampling.

    Parameters
    ----------
    total
        Total number of available frames.
    num_frames
        Number of indices to return.
    original_fps
        Native frame rate of the video.
    target_fps
        When ``None``, use uniform index spacing.  Otherwise, use
        FPS-based resampling at this rate.

    Returns
    -------
    list[int]
        Selected frame indices.
    """
    if target_fps is None:
        from lpcv.datasets.utils import uniform_temporal_indices

        return uniform_temporal_indices(total, num_frames)
    return _fps_resample_indices(total, original_fps, target_fps, num_frames)


class PyAVDecoder:
    """Baseline CPU decoder using PyAV.

    Decodes **all** frames into memory, then subsamples.  Suitable for any
    video format that libav supports but relatively slow for long videos.

    Parameters
    ----------
    target_fps
        When ``None`` (default), frames are selected via uniform index
        spacing.  When set to a positive integer, FPS-based resampling is
        used instead, replicating the LPCVC competition's patched
        ``VideoClips`` behaviour.
    """

    def __init__(self, target_fps: int | None = None) -> None:
        self.target_fps = target_fps

    def decode(self, path: Path, num_frames: int) -> torch.Tensor:
        """Decode frames using PyAV.

        Parameters
        ----------
        path
            Path to the video file.
        num_frames
            Number of frames to sample.

        Returns
        -------
        torch.Tensor
            Float tensor ``(num_frames, 3, H, W)`` in ``[0, 255]``.
        """
        import av
        import numpy as np

        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            original_fps = float(stream.average_rate or stream.guessed_rate or 30)
            frames = [f for f in container.decode(stream)]

        total = len(frames)
        if total == 0:
            raise ValueError(f"No frames decoded from {path}")

        indices = _select_indices(total, num_frames, original_fps, self.target_fps)
        sampled = [frames[i] for i in indices]
        t = torch.stack([torch.from_numpy(np.array(f.to_image().convert("RGB"))) for f in sampled])
        return t.float().permute(0, 3, 1, 2)


class TorchCodecCPUDecoder:
    """CPU decoder using TorchCodec.

    Only decodes the requested frame indices (seek-based), avoiding full
    video decode.  Requires the ``torchcodec`` package.

    Parameters
    ----------
    target_fps
        When ``None`` (default), frames are selected via uniform index
        spacing.  When set to a positive integer, FPS-based resampling is
        used instead.
    """

    def __init__(self, target_fps: int | None = None) -> None:
        self.target_fps = target_fps

    def decode(self, path: Path, num_frames: int) -> torch.Tensor:
        """Decode frames using TorchCodec on CPU.

        Parameters
        ----------
        path
            Path to the video file.
        num_frames
            Number of frames to sample.

        Returns
        -------
        torch.Tensor
            Float tensor ``(num_frames, C, H, W)`` in ``[0, 255]``.
        """
        from torchcodec.decoders import VideoDecoder as TVideoDecoder

        decoder = TVideoDecoder(str(path), device="cpu", dimension_order="NCHW")
        total = decoder.metadata.num_frames or 1
        original_fps = float(decoder.metadata.average_fps or 30)
        indices = _select_indices(total, num_frames, original_fps, self.target_fps)
        return decoder.get_frames_at(indices).data.float()


class TorchCodecNVDECDecoder:
    """GPU decoder using TorchCodec + NVDEC beta backend.

    Frames are decoded directly on the GPU as CUDA tensors, eliminating
    the CPU-to-GPU transfer.  Requires ``torchcodec`` with CUDA support.

    When *num_gpus* is set, the decoder distributes work across multiple
    GPUs by assigning each ``DataLoader`` worker to a GPU based on its
    worker ID (``worker_id % num_gpus``).  Outside a worker the *device*
    parameter is used as-is.

    Parameters
    ----------
    device
        CUDA device string, e.g. ``"cuda"`` or ``"cuda:0"``.  Ignored
        when *num_gpus* is set and decoding runs inside a DataLoader worker.
    num_gpus
        Number of GPUs to distribute across.  When *None*, all decoding
        happens on *device*.
    target_fps
        When ``None`` (default), frames are selected via uniform index
        spacing.  When set to a positive integer, FPS-based resampling is
        used instead.
    """

    def __init__(
        self,
        device: str = "cuda",
        num_gpus: int | None = None,
        target_fps: int | None = None,
    ) -> None:
        self.device = device
        self.num_gpus = num_gpus
        self.target_fps = target_fps

    def _resolve_device(self) -> str:
        """Return the CUDA device string for the current context.

        In distributed training the ``LOCAL_RANK`` environment variable takes
        precedence so that each process decodes on its own GPU.  When
        *num_gpus* is set without distributed training, DataLoader worker IDs
        are used to round-robin across GPUs.
        """
        import os

        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank is not None:
            return f"cuda:{local_rank}"
        if self.num_gpus is None:
            return self.device
        info = torch.utils.data.get_worker_info()
        gpu_id = (info.id if info else 0) % self.num_gpus
        return f"cuda:{gpu_id}"

    def decode(self, path: Path, num_frames: int) -> torch.Tensor:
        """Decode frames using NVDEC on GPU.

        Parameters
        ----------
        path
            Path to the video file.
        num_frames
            Number of frames to sample.

        Returns
        -------
        torch.Tensor
            Float CUDA tensor ``(num_frames, C, H, W)`` in ``[0, 255]``.
        """
        from torchcodec.decoders import VideoDecoder as TVideoDecoder
        from torchcodec.decoders import set_cuda_backend

        device = self._resolve_device()
        with set_cuda_backend("beta"):
            decoder = TVideoDecoder(str(path), device=device, dimension_order="NCHW")
            total = decoder.metadata.num_frames or 1
            original_fps = float(decoder.metadata.average_fps or 30)
            indices = _select_indices(total, num_frames, original_fps, self.target_fps)
            return decoder.get_frames_at(indices).data.float()


DECODERS: dict[str, type[PyAVDecoder | TorchCodecCPUDecoder | TorchCodecNVDECDecoder]] = {
    "pyav": PyAVDecoder,
    "torchcodec-cpu": TorchCodecCPUDecoder,
    "torchcodec-nvdec": TorchCodecNVDECDecoder,
}
"""Mapping from decoder name to class."""


def get_decoder(
    name: str, **kwargs: str | int | None
) -> PyAVDecoder | TorchCodecCPUDecoder | TorchCodecNVDECDecoder:
    """Instantiate a decoder by name.

    Parameters
    ----------
    name
        One of ``"pyav"``, ``"torchcodec-cpu"``, ``"torchcodec-nvdec"``.
    **kwargs
        Extra keyword arguments forwarded to the decoder constructor.
        All decoders accept ``target_fps`` to switch from uniform index
        sampling to FPS-based resampling.

    Returns
    -------
    PyAVDecoder | TorchCodecCPUDecoder | TorchCodecNVDECDecoder
        An instance of the requested decoder.

    Raises
    ------
    ValueError
        If *name* is not a recognised decoder.
    """
    if name not in DECODERS:
        raise ValueError(f"Unknown decoder '{name}'. Available: {sorted(DECODERS)}")
    return DECODERS[name](**kwargs)  # type: ignore[arg-type]
