from __future__ import annotations

from typing import TYPE_CHECKING

from datasets import Array3D, Dataset, DatasetDict, Features, load_from_disk
from loguru import logger
from transformers import VideoMAEImageProcessor

from lpcv.datasets.utils import subsample

if TYPE_CHECKING:
    from pathlib import Path


class PrecomputedDataset:
    def __init__(
        self,
        dataset: DatasetDict,
        model_name: str,
        num_frames: int = 16,
        image_size: int = 224,
        num_workers: int | None = None,
    ):
        self.dataset = dataset
        self.model_name = model_name
        self.num_frames = num_frames
        self.image_size = image_size
        self.num_workers = num_workers

        self.processor = VideoMAEImageProcessor.from_pretrained(
            model_name,
            do_resize=True,
            size={"shortest_edge": image_size},
            do_center_crop=True,
            crop_size={"height": image_size, "width": image_size},
        )

        first_split = next(iter(dataset))
        self.label_feature = dataset[first_split].features["label"]

    def _preprocess_example(self, example: dict) -> dict:
        frames = list(example["video"])
        sampled = subsample(frames, self.num_frames)
        if not sampled:
            return {"pixel_values": None, "label": example["label"]}

        pixel_values = self.processor(sampled, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.squeeze(0).numpy()
        return {"pixel_values": pixel_values, "label": example["label"]}

    def precompute(self, output_dir: Path) -> DatasetDict:
        features = Features(
            {
                "pixel_values": Array3D(
                    dtype="float32",
                    shape=(self.num_frames, self.image_size, self.image_size),
                ),
                "label": self.label_feature,
            }
        )

        result_splits: dict[str, Dataset] = {}

        for split_name, split_ds in self.dataset.items():
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
                lambda x: x["pixel_values"] is not None,
                num_proc=self.num_workers,
            )

            result_splits[split_name] = processed.cast(features)

        result = DatasetDict(result_splits)

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
