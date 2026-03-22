"""Video utility functions for probing, remuxing, validation and subsampling.

Provides helpers used by the dataset conversion and loading pipelines:

- Video integrity checks (probe, remux on failure).
- Dimension validation (minimum size, aspect ratio).
- Frame index computation for temporal subsampling.
"""

import shutil
from pathlib import Path

import av
from av.video.stream import VideoStream
from loguru import logger

from lpcv.datasets.info import SPLIT_DIRS, VIDEO_EXTENSIONS


def is_compatible_with_dataset(data_dir: Path):
    """Check whether *data_dir* looks like a valid videofolder dataset.

    A valid layout has at least one split directory (``train`` / ``val``)
    containing class sub-directories with video files.

    Parameters
    ----------
    data_dir
        Root directory to check.

    Returns
    -------
    bool
        ``True`` if the directory matches the expected videofolder structure.
    """
    if not data_dir.is_dir():
        return False

    split_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name in SPLIT_DIRS]
    if not split_dirs:
        return False

    for split_dir in split_dirs:
        class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        if not class_dirs:
            return False
        if not any(
            f.suffix.lower() in VIDEO_EXTENSIONS
            for class_dir in class_dirs
            for f in class_dir.iterdir()
            if f.is_file()
        ):
            return False

    return True


def probe_video(path: Path) -> bool:
    """Try to open and decode one keyframe from *path*.

    Parameters
    ----------
    path
        Path to the video file.

    Returns
    -------
    bool
        ``True`` if at least one frame can be decoded.
    """
    try:
        with av.open(str(path)) as container:
            video_streams = container.streams.video
            if not video_streams:
                return False
            stream = video_streams[0]
            stream.codec_context.skip_frame = "NONKEY"
            for _ in container.decode(stream):
                break
        return True
    except Exception:
        logger.debug(f"Failed to probe {path} with pyav.")
        return False


def remux_video(src: Path) -> Path | None:
    """Re-mux *src* in-place to fix container-level corruption.

    The video stream is copied without re-encoding.  If remuxing fails the
    temporary file is cleaned up and ``None`` is returned.

    Parameters
    ----------
    src
        Path to the source video file.

    Returns
    -------
    Path | None
        *src* on success, ``None`` on failure.
    """
    remuxed_path = src.with_suffix(".remux" + src.suffix)
    try:
        with av.open(str(src)) as input_container:
            input_stream = input_container.streams.video[0]
            with av.open(str(remuxed_path), mode="w") as output_container:
                output_stream = output_container.add_stream(
                    codec_name=input_stream.codec_context.name,
                    rate=input_stream.base_rate,
                )
                assert isinstance(output_stream, VideoStream)
                output_stream.width = input_stream.codec_context.width
                output_stream.height = input_stream.codec_context.height

                for packet in input_container.demux(input_stream):
                    if packet.dts is None:
                        continue
                    packet.stream = output_stream
                    output_container.mux(packet)

        shutil.move(remuxed_path, src)
        return src
    except Exception:
        if remuxed_path.is_file():
            remuxed_path.unlink()
        logger.debug(f"Failed to remux {src}.")
        return None


def check_video_integrity(src: Path) -> bool:
    """Verify that *src* can be decoded, attempting a remux if the first probe fails.

    Parameters
    ----------
    src
        Path to the video file.

    Returns
    -------
    bool
        ``True`` if the video is decodable (possibly after remuxing).
    """
    if probe_video(src):
        return True

    remuxed = remux_video(src)
    return bool(remuxed is not None and probe_video(remuxed))


def check_video_dimensions(
    src: Path,
    *,
    min_dim: int = 16,
    max_aspect_ratio: float = 10.0,
) -> bool:
    """Validate that *src* has reasonable spatial dimensions.

    Parameters
    ----------
    src
        Path to the video file.
    min_dim
        Minimum acceptable width **or** height in pixels.
    max_aspect_ratio
        Maximum acceptable ratio of the longer side to the shorter side.

    Returns
    -------
    bool
        ``True`` if dimensions pass all checks.
    """
    try:
        with av.open(str(src)) as container:
            stream = container.streams.video[0]
            w = stream.codec_context.width
            h = stream.codec_context.height
    except Exception:
        logger.debug(f"Failed to read dimensions for {src}")
        return False

    if w < min_dim or h < min_dim:
        logger.debug(f"Dimensions too small ({w}x{h}): {src}")
        return False

    aspect = max(w, h) / max(min(w, h), 1)
    if aspect > max_aspect_ratio:
        logger.debug(f"Extreme aspect ratio {aspect:.1f} ({w}x{h}): {src}")
        return False

    return True


def uniform_temporal_indices(total: int, num_frames: int) -> list[int]:
    """Return *num_frames* uniformly-spaced indices in ``[0, total)``.

    Parameters
    ----------
    total
        Total number of available frames.
    num_frames
        Number of indices to return.

    Returns
    -------
    list[int]
        Sorted list of frame indices.
    """
    import torch

    return torch.linspace(0, total - 1, num_frames).long().tolist()
