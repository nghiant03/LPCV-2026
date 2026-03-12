from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from datasets import Dataset, DatasetDict, load_from_disk
from loguru import logger

if TYPE_CHECKING:
    from pathlib import Path


class PrecomputedDataset:
    def __init__(
        self,
        dataset: DatasetDict,
        num_workers: int | None = None,
    ):
        self.dataset = dataset
        self.num_workers = num_workers

    def _preprocess_example(self, example: dict) -> dict:
        frames = list(example["video"])
        if not frames:
            return {"frames": None, "label": example["label"]}

        frames_array = np.stack([np.array(f) for f in frames])
        return {"frames": frames_array, "label": example["label"]}

    def precompute(self, output_dir: Path) -> DatasetDict:
        result_splits: dict[str, Dataset] = {}

        for split_name, split_ds in self.dataset.items():
            split_name = str(split_name)
            logger.info(
                f"Precomputing {split_name}: {len(split_ds)} videos "
                f"(num_workers={self.num_workers})"
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
    def load(cache_dir: Path) -> DatasetDict:
        logger.info(f"Loading precomputed dataset from {cache_dir}")
        ds = load_from_disk(str(cache_dir))
        if not isinstance(ds, DatasetDict):
            raise TypeError(f"Expected DatasetDict, got {type(ds).__name__}")
        return ds
