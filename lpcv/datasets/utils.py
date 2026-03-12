import shutil
from pathlib import Path

import av
import numpy as np
from av.video.stream import VideoStream
from loguru import logger

from lpcv.datasets.info import SPLIT_DIRS, VIDEO_EXTENSIONS

MIN_DIMENSION = 16
MAX_ASPECT_RATIO = 10.0

def is_compatible_with_dataset(data_dir: Path):
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
    if probe_video(src):
        return True

    remuxed = remux_video(src)
    return bool(remuxed is not None and probe_video(remuxed))




def check_video_dimensions(
    src: Path,
    *,
    min_dim: int = MIN_DIMENSION,
    max_aspect_ratio: float = MAX_ASPECT_RATIO,
) -> bool:
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


def subsample(
    frames: list,
    num_frames: int = 16,
    *,
    mode: str = "dense",
    stride: int = 4,
) -> list:
    total = len(frames)
    if total == 0:
        return []

    if mode == "dense":
        window_size = num_frames * stride
        if total >= window_size:
            start = int(np.random.randint(0, total - window_size + 1))
            indices = list(range(start, start + window_size, stride))
            return [frames[i] for i in indices]
        # fallback to uniform when video is too short
        indices = np.linspace(0, total - 1, num_frames).astype(int).tolist()
        return [frames[i] for i in indices]

    # uniform mode
    if total >= num_frames:
        indices = np.linspace(0, total - 1, num_frames).astype(int).tolist()
        return [frames[i] for i in indices]

    return frames + [frames[-1]] * (num_frames - total)
