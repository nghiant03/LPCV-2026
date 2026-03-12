from __future__ import annotations

import json
import multiprocessing as mp
import shutil
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import torch
from datasets import Video
from loguru import logger
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

from lpcv.transforms import TRAIN_PRESET, VAL_PRESET, build_transform

if TYPE_CHECKING:
    from datasets import DatasetDict
    from torchvision.transforms import Compose

DEFAULT_SHORT_SIDE = 320
DEFAULT_MAX_FRAMES = 64
JPEG_QUALITY = 85

_CHECKPOINT_DIR_NAME = "_precompute_ckpt"
_METADATA_FILE = "metadata.json"
_FAIL_SENTINEL = b""


# ---------------------------------------------------------------------------
# Module-level worker — must be a plain function to be picklable by mp.Pool
# ---------------------------------------------------------------------------


def _worker(
    args: tuple[int, dict],
    short_side: int,
    max_frames: int | None,
    ckpt_dir: str,
) -> int | None:
    """Decode one video, JPEG-encode frames, save to checkpoint dir."""
    idx, row = args

    ckpt_path = f"{ckpt_dir}/{idx}.npz"
    if _is_done(ckpt_path):
        return idx

    video_info = row.get("video")
    video_path = video_info.get("path") if isinstance(video_info, dict) else None
    if not video_path:
        _mark_failed(ckpt_path)
        return None

    try:
        from decord import VideoReader, cpu

        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        total = len(vr)
        if total == 0:
            _mark_failed(ckpt_path)
            return None

        n = min(max_frames, total) if max_frames else total
        indices = np.linspace(0, total - 1, n).astype(int)

        frames = vr.get_batch(indices).asnumpy()  # (T, H, W, C) uint8
    except Exception:
        logger.warning(f"[{idx}] Failed to decode: {video_path}")
        _mark_failed(ckpt_path)
        return None

    frames = _resize(frames, short_side)
    if frames is None:
        logger.warning(f"[{idx}] Skipped oversized video: {video_path}")
        _mark_failed(ckpt_path)
        return None

    jpeg_frames = _encode_jpeg_batch(frames)

    tmp_path = f"{ckpt_dir}/{idx}.tmp.npz"
    frame_data: dict[str, np.ndarray] = {f"f{i}": buf for i, buf in enumerate(jpeg_frames)}
    frame_data["label"] = np.array(int(row["label"]))
    np.savez(tmp_path, **frame_data)  # type: ignore[arg-type]
    shutil.move(tmp_path, ckpt_path)
    return idx


def _is_done(ckpt_path: str) -> bool:
    p = Path(ckpt_path)
    return p.exists() and p.stat().st_size > 0


def _mark_failed(ckpt_path: str) -> None:
    """Write a zero-byte sentinel so we don't retry known-bad videos."""
    Path(ckpt_path).write_bytes(_FAIL_SENTINEL)


def _encode_jpeg_batch(frames: np.ndarray) -> list[np.ndarray]:
    """JPEG-encode each frame (T, H, W, C) → list of 1-D uint8 arrays."""
    result = []
    for i in range(frames.shape[0]):
        bgr = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            raise RuntimeError(f"JPEG encode failed for frame {i}")
        result.append(buf.ravel())
    return result


def _decode_jpeg_batch(npz_data: dict[str, Any]) -> np.ndarray:
    """Decode JPEG buffers from npz back to (T, H, W, C) uint8 numpy array."""
    frame_keys = sorted(
        (k for k in npz_data if k.startswith("f")),
        key=lambda k: int(k[1:]),
    )
    frames = []
    for k in frame_keys:
        buf = npz_data[k]
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"JPEG decode failed for key {k}")
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return np.stack(frames)


def _resize(frames: np.ndarray, short_side: int) -> np.ndarray | None:
    t, h, w, _ = frames.shape
    ss = min(h, w)
    if ss == short_side:
        return frames

    scale = short_side / ss
    new_h, new_w = int(round(h * scale)), int(round(w * scale))

    if t * 3 * new_h * new_w > 500_000_000:
        return None

    return np.stack(
        [cv2.resize(frames[i], (new_w, new_h), interpolation=cv2.INTER_LINEAR) for i in range(t)]
    )


# ---------------------------------------------------------------------------
# Lazy dataset backed by per-video .npz checkpoint files
# ---------------------------------------------------------------------------


class _LazyNpzDataset(TorchDataset):
    """Torch Dataset that lazily reads JPEG-encoded npz files and applies transforms."""

    def __init__(
        self,
        npz_paths: list[Path],
        labels: list[int],
        transform: Compose | None = None,
    ):
        self.npz_paths = npz_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.npz_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        data = np.load(self.npz_paths[index])
        frames = _decode_jpeg_batch(data)  # (T, H, W, C) uint8

        video = torch.from_numpy(frames).permute(0, 3, 1, 2).float()  # (T, C, H, W)
        if self.transform is not None:
            video = self.transform(video)

        return {"pixel_values": video, "labels": self.labels[index]}


class _FakeClassLabel:
    """Minimal stand-in for datasets.ClassLabel so VideoMAEModelTrainer can read label names."""

    def __init__(self, names: list[str]):
        self.names = names
        self.num_classes = len(names)


class _FakeFeatures(dict):  # type: ignore[type-arg]
    """Minimal dict that satisfies trainer.features.get('label')."""

    def __init__(self, label_names: list[str]):
        super().__init__({"label": _FakeClassLabel(label_names)})


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class PrecomputedDataset:
    def __init__(
        self,
        dataset: DatasetDict,
        num_workers: int | None = None,
        short_side: int = DEFAULT_SHORT_SIDE,
        max_frames: int | None = DEFAULT_MAX_FRAMES,
        chunksize: int = 16,
    ):
        self.dataset = dataset
        self.num_workers = num_workers or mp.cpu_count()
        self.short_side = short_side
        self.max_frames = max_frames
        self.chunksize = chunksize

    # ------------------------------------------------------------------
    # Precompute
    # ------------------------------------------------------------------

    def precompute(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        ckpt_root = output_dir / _CHECKPOINT_DIR_NAME

        metadata: dict[str, Any] = {"splits": {}}

        for split_name, split_ds in self.dataset.items():
            split_name = str(split_name)
            split_ds = split_ds.cast_column("video", Video(decode=False))

            ckpt_dir = ckpt_root / split_name
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            all_args = list(enumerate(split_ds))
            done_ids = {
                int(p.stem)
                for p in ckpt_dir.iterdir()
                if p.suffix == ".npz" and p.stem.isdigit() and p.stat().st_size > 0
            }
            pending: list[tuple[int, dict]] = [
                (i, dict(row)) for i, row in all_args if i not in done_ids
            ]

            logger.info(
                f"Precomputing {split_name}: {len(split_ds)} total, "
                f"{len(done_ids)} already done, {len(pending)} pending "
                f"(workers={self.num_workers}, short_side={self.short_side}, "
                f"max_frames={self.max_frames})"
            )

            if pending:
                fn = partial(
                    _worker,
                    short_side=self.short_side,
                    max_frames=self.max_frames,
                    ckpt_dir=str(ckpt_dir),
                )
                with mp.Pool(self.num_workers) as pool:
                    for _ in tqdm(
                        pool.imap_unordered(fn, pending, chunksize=self.chunksize),
                        total=len(pending),
                        desc=split_name,
                    ):
                        pass

            label_feature = split_ds.features.get("label")
            label_names = list(label_feature.names) if label_feature else []

            kept = sum(
                1
                for p in ckpt_dir.iterdir()
                if p.suffix == ".npz" and p.stem.isdigit() and p.stat().st_size > 0
            )
            metadata["splits"][split_name] = {
                "total": len(all_args),
                "kept": kept,
                "label_names": label_names,
            }
            logger.info(f"  {kept}/{len(all_args)} videos kept → {split_name}")

        meta_path = output_dir / _METADATA_FILE
        meta_path.write_text(json.dumps(metadata, indent=2))
        logger.info(f"Precomputed checkpoints saved to {output_dir}")

    # ------------------------------------------------------------------
    # Load — returns torch Datasets compatible with HF Trainer
    # ------------------------------------------------------------------

    @staticmethod
    def load(
        cache_dir: Path,
        train_transform: Compose | None = None,
        val_transform: Compose | None = None,
    ) -> tuple[_LazyNpzDataset, _LazyNpzDataset]:
        cache_dir = Path(cache_dir)
        logger.info(f"Loading precomputed dataset from {cache_dir}")

        meta_path = cache_dir / _METADATA_FILE
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {meta_path}. Run 'precompute' first or check the path."
            )

        metadata = json.loads(meta_path.read_text())
        splits = metadata["splits"]

        ckpt_root = cache_dir / _CHECKPOINT_DIR_NAME

        if train_transform is None:
            train_transform = build_transform(TRAIN_PRESET)
        if val_transform is None:
            val_transform = build_transform(VAL_PRESET)

        def _build_split(split_name: str, transform: Compose) -> _LazyNpzDataset:
            ckpt_dir = ckpt_root / split_name
            npz_paths: list[Path] = []
            labels: list[int] = []

            for p in sorted(
                ckpt_dir.iterdir(), key=lambda x: int(x.stem) if x.stem.isdigit() else -1
            ):
                if p.suffix != ".npz" or not p.stem.isdigit():
                    continue
                if p.stat().st_size == 0:
                    continue
                data = np.load(p, mmap_mode="r")
                npz_paths.append(p)
                labels.append(int(data["label"]))

            ds = _LazyNpzDataset(npz_paths, labels, transform)
            label_names = splits[split_name].get("label_names", [])
            ds.features = _FakeFeatures(label_names)  # type: ignore[attr-defined]
            logger.info(f"  Loaded {split_name}: {len(npz_paths)} samples")
            return ds

        train_ds = _build_split("train", train_transform)

        val_key = next((k for k in ("val", "validation", "test") if k in splits), None)
        if val_key is None:
            raise KeyError(f"No validation split found in {list(splits.keys())}")
        val_ds = _build_split(val_key, val_transform)

        return train_ds, val_ds
