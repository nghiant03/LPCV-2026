"""Video decoder protocol and concrete implementations.

Provides a ``VideoDecoder`` protocol and three backends:

- ``PyAVDecoder`` — CPU decoding via PyAV (most compatible).
- ``TorchCodecCPUDecoder`` — CPU decoding via TorchCodec (seek-based).
- ``TorchCodecNVDECDecoder`` — GPU decoding via TorchCodec + NVDEC.

Use ``get_decoder`` to instantiate a decoder by name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    from pathlib import Path


@runtime_checkable
class VideoDecoder(Protocol):
    """Protocol for video decoders that extract uniformly-sampled frames.

    All implementations return a float tensor of shape ``(T, C, H, W)`` with
    pixel values in ``[0, 255]``.
    """

    def decode(self, path: Path, num_frames: int) -> torch.Tensor:
        """Decode *num_frames* uniformly-spaced frames from a video file.

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


class PyAVDecoder:
    """Baseline CPU decoder using PyAV.

    Decodes **all** frames into memory, then uniformly subsamples.  Suitable
    for any video format that libav supports but relatively slow for long
    videos.
    """

    def decode(self, path: Path, num_frames: int) -> torch.Tensor:
        """Decode frames using PyAV.

        Parameters
        ----------
        path
            Path to the video file.
        num_frames
            Number of frames to sample uniformly.

        Returns
        -------
        torch.Tensor
            Float tensor ``(num_frames, 3, H, W)`` in ``[0, 255]``.
        """
        import av
        import numpy as np

        from lpcv.datasets.utils import uniform_temporal_indices

        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            frames = [f for f in container.decode(stream)]

        indices = uniform_temporal_indices(len(frames), num_frames)
        sampled = [frames[i] for i in indices]
        t = torch.stack([torch.from_numpy(np.array(f.to_image().convert("RGB"))) for f in sampled])
        return t.float().permute(0, 3, 1, 2)


class TorchCodecCPUDecoder:
    """CPU decoder using TorchCodec.

    Only decodes the requested frame indices (seek-based), avoiding full
    video decode.  Requires the ``torchcodec`` package.
    """

    def decode(self, path: Path, num_frames: int) -> torch.Tensor:
        """Decode frames using TorchCodec on CPU.

        Parameters
        ----------
        path
            Path to the video file.
        num_frames
            Number of frames to sample uniformly.

        Returns
        -------
        torch.Tensor
            Float tensor ``(num_frames, C, H, W)`` in ``[0, 255]``.
        """
        from torchcodec.decoders import VideoDecoder as TVideoDecoder

        from lpcv.datasets.utils import uniform_temporal_indices

        decoder = TVideoDecoder(str(path), device="cpu", dimension_order="NCHW")
        total = decoder.metadata.num_frames or 1
        indices = uniform_temporal_indices(total, num_frames)
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
    """

    def __init__(self, device: str = "cuda", num_gpus: int | None = None) -> None:
        self.device = device
        self.num_gpus = num_gpus

    def _resolve_device(self) -> str:
        """Return the CUDA device string for the current context."""
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
            Number of frames to sample uniformly.

        Returns
        -------
        torch.Tensor
            Float CUDA tensor ``(num_frames, C, H, W)`` in ``[0, 255]``.
        """
        from torchcodec.decoders import VideoDecoder as TVideoDecoder
        from torchcodec.decoders import set_cuda_backend

        from lpcv.datasets.utils import uniform_temporal_indices

        with set_cuda_backend("beta"):
            decoder = TVideoDecoder(
                str(path), device=self._resolve_device(), dimension_order="NCHW"
            )
        total = decoder.metadata.num_frames or 1
        indices = uniform_temporal_indices(total, num_frames)
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
