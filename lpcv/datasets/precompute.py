from __future__ import annotations

import multiprocessing as mp
import shutil
from functools import partial
from typing import TYPE_CHECKING

import cv2
import numpy as np
from datasets import Dataset, DatasetDict, Video, load_from_disk
from decord import VideoReader, cpu
from loguru import logger
from tqdm import tqdm

from lpcv.transforms import TRAIN_PRESET, VAL_PRESET, build_transform

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from torchvision.transforms import Compose

DEFAULT_SHORT_SIDE = 320
DEFAULT_MAX_FRAMES = 64

_CHECKPOINT_DIR_NAME = "_precompute_ckpt"
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
    """Decode one video, save to checkpoint dir, return idx on success."""
    idx, row = args

    ckpt_path = f"{ckpt_dir}/{idx}.npz"

    video_info = row.get("video")
    video_path = video_info.get("path") if isinstance(video_info, dict) else None
    if not video_path:
        _mark_failed(ckpt_path)
        return None

    try:
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

    tmp_path = f"{ckpt_path}.tmp"
    np.savez_compressed(tmp_path, frames=frames, label=np.array(int(row["label"])))
    shutil.move(tmp_path, ckpt_path)
    return idx


def _mark_failed(ckpt_path: str) -> None:
    """Write a zero-byte sentinel so we don't retry known-bad videos."""
    from pathlib import Path as _P

    _P(ckpt_path).write_bytes(_FAIL_SENTINEL)


def _resize(frames: np.ndarray, short_side: int) -> np.ndarray | None:
    t, h, w, _ = frames.shape
    ss = min(h, w)
    if ss == short_side:
        return frames

    scale = short_side / ss
    new_h, new_w = int(round(h * scale)), int(round(w * scale))

    if t * 3 * new_h * new_w > 500_000_000:
        return None

    return np.stack([
        cv2.resize(frames[i], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        for i in range(t)
    ])


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
        self.dataset     = dataset
        self.num_workers = num_workers or mp.cpu_count()
        self.short_side  = short_side
        self.max_frames  = max_frames
        self.chunksize   = chunksize

    # ------------------------------------------------------------------
    # Precompute
    # ------------------------------------------------------------------

    def precompute(self, output_dir: Path) -> DatasetDict:
        output_dir.mkdir(parents=True, exist_ok=True)
        ckpt_root = output_dir / _CHECKPOINT_DIR_NAME
        result_splits: dict[str, Dataset] = {}

        for split_name, split_ds in self.dataset.items():
            split_name = str(split_name)
            split_ds   = split_ds.cast_column("video", Video(decode=False))

            ckpt_dir = ckpt_root / split_name
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            all_args = list(enumerate(split_ds))
            done_ids = {
                int(p.stem)
                for p in ckpt_dir.iterdir()
                if p.suffix == ".npz"
            }
            pending = [a for a in all_args if a[0] not in done_ids]

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

            records: list[dict] = []
            for p in sorted(ckpt_dir.iterdir()):
                if p.suffix != ".npz" or p.stat().st_size == 0:
                    continue
                data = np.load(p)
                records.append({
                    "frames": data["frames"],
                    "label": int(data["label"]),
                })

            result_splits[split_name] = Dataset.from_list(records)
            logger.info(f"  {len(records)}/{len(all_args)} videos kept → {split_name}")

        result = DatasetDict(result_splits)
        result.save_to_disk(str(output_dir), num_proc=self.num_workers)

        shutil.rmtree(ckpt_root, ignore_errors=True)
        logger.info(f"Precomputed DatasetDict saved to {output_dir}")

        return result

    # ------------------------------------------------------------------
    # Load — returns HF Datasets compatible with Trainer
    # ------------------------------------------------------------------

    @staticmethod
    def load(
        cache_dir: Path,
        train_transform: Compose | None = None,
        val_transform:   Compose | None = None,
    ) -> tuple[Dataset, Dataset]:
        logger.info(f"Loading precomputed dataset from {cache_dir}")

        ds = load_from_disk(str(cache_dir))
        if not isinstance(ds, DatasetDict):
            raise TypeError(f"Expected DatasetDict, got {type(ds).__name__}")

        fmt_step = [{"name": "FromNumpy"}]
        if train_transform is None:
            train_transform = build_transform(fmt_step + TRAIN_PRESET)
        if val_transform is None:
            val_transform   = build_transform(fmt_step + VAL_PRESET)

        def _make_transform_fn(transform: Compose) -> Callable:
            def _apply(examples: dict) -> dict:
                examples["pixel_values"] = [
                    transform(frames) for frames in examples["frames"]
                ]
                examples["labels"] = examples["label"]
                return examples
            return _apply

        train_ds = ds["train"]
        train_ds.set_transform(_make_transform_fn(train_transform))

        val_key = next(
            (k for k in ("val", "validation", "test") if k in ds), None
        )
        if val_key is None:
            raise KeyError(f"No validation split found in {list(ds.keys())}")
        val_ds = ds[val_key]
        val_ds.set_transform(_make_transform_fn(val_transform))

        return train_ds, val_ds
