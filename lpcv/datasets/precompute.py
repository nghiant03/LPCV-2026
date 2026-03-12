from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from datasets import Dataset, DatasetDict, load_from_disk
from loguru import logger

from lpcv.transforms import TRAIN_PRESET, VAL_PRESET, ShortSideScale, build_transform

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from torchvision.transforms import Compose

DEFAULT_SHORT_SIDE = 320
DEFAULT_MAX_FRAMES = 64


class PrecomputedDataset:
    def __init__(
        self,
        dataset: DatasetDict,
        num_workers: int | None = None,
        short_side: int = DEFAULT_SHORT_SIDE,
        max_frames: int | None = DEFAULT_MAX_FRAMES,
    ):
        self.dataset = dataset
        self.num_workers = num_workers
        self.short_side = short_side
        self.max_frames = max_frames
        self._resize = ShortSideScale(size=short_side)

    def _preprocess_example(self, example: dict) -> dict:
        frames = list(example["video"])
        if not frames:
            return {"frames": None, "label": example["label"]}

        if self.max_frames is not None and len(frames) > self.max_frames:
            indices = np.linspace(0, len(frames) - 1, self.max_frames).astype(int)
            frames = [frames[i] for i in indices]

        import torch

        video = torch.stack(
            [torch.from_numpy(np.array(f)) for f in frames]
        ).float()

        t, c, h, w = video.shape
        short_side = min(h, w)
        scale = self.short_side / short_side
        new_h, new_w = int(h * scale), int(w * scale)
        alloc_bytes = t * c * new_h * new_w * 4
        logger.debug(
            f"Resize {h}x{w} -> {new_h}x{new_w} "
            f"(frames={t}, scale={scale:.2f}, alloc={alloc_bytes / 1e9:.2f}GB)"
        )
        if alloc_bytes > 2e9:
            logger.warning(
                f"Skipping video with extreme resize: {h}x{w} -> {new_h}x{new_w} "
                f"(alloc={alloc_bytes / 1e9:.2f}GB)"
            )
            return {"frames": None, "label": example["label"]}

        video = self._resize(video)
        frames_chw = video.to(torch.uint8).numpy()
        return {"frames": frames_chw, "label": example["label"]}

    def precompute(self, output_dir: Path) -> DatasetDict:
        result_splits: dict[str, Dataset] = {}

        for split_name, split_ds in self.dataset.items():
            split_name = str(split_name)
            logger.info(
                f"Precomputing {split_name}: {len(split_ds)} videos "
                f"(num_workers={self.num_workers}, "
                f"short_side={self.short_side}, max_frames={self.max_frames})"
            )

            processed = split_ds.map(
                self._preprocess_example,
                remove_columns=split_ds.column_names,
                num_proc=self.num_workers,
                writer_batch_size=100,
                desc=f"Precomputing {split_name}",
            )

            processed = processed.filter(
                lambda x: x["frames"] is not None,
                num_proc=self.num_workers,
            )

            result_splits[split_name] = processed

        result = DatasetDict(result_splits)  # type: ignore[arg-type]

        output_dir.mkdir(parents=True, exist_ok=True)
        result.save_to_disk(str(output_dir), num_proc=self.num_workers)
        logger.info(f"Precomputed dataset saved to {output_dir}")

        return result

    @staticmethod
    def load(
        cache_dir: Path,
        train_transform: Compose | None = None,
        val_transform: Compose | None = None,
    ) -> tuple[Dataset, Dataset]:
        logger.info(f"Loading precomputed dataset from {cache_dir}")
        ds = load_from_disk(str(cache_dir))
        if not isinstance(ds, DatasetDict):
            raise TypeError(f"Expected DatasetDict, got {type(ds).__name__}")

        fmt_step = [{"name": "FromNumpy"}]
        if train_transform is None:
            train_transform = build_transform(fmt_step + TRAIN_PRESET)
        if val_transform is None:
            val_transform = build_transform(fmt_step + VAL_PRESET)

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

        val_key = next((k for k in ("val", "validation", "test") if k in ds), None)
        if val_key is None:
            raise KeyError(f"No validation split found in {list(ds.keys())}")
        val_ds = ds[val_key]
        val_ds.set_transform(_make_transform_fn(val_transform))

        return train_ds, val_ds
