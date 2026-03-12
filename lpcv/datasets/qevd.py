import json
import re
import shutil
from functools import cached_property
from pathlib import Path

from datasets import DatasetDict, load_dataset, load_from_disk
from loguru import logger
from pydantic import BaseModel
from tqdm.contrib.concurrent import process_map

from lpcv.datasets.info import SPLIT_DIRS, TARGET_LABEL_FILE_NAME
from lpcv.datasets.utils import (
    check_video_integrity,
    is_compatible_with_dataset,
)

FOLDER_PATTERN = "QEVD-FIT-300k-Part-"
SOURCE_LABEL_FILE_NAME = "fine_grained_labels.json"
QUARANTINE_DIR_NAME = "quarantine"
SPLIT_MAP = {
    "test": "val"
}
CHUNK_SIZE = 10000


class QEVDLabel(BaseModel):
    video_path: str
    labels: list[str]
    labels_descriptive: list[str]
    split: str


class QEVDAdapter:
    def __init__(
        self,
        data_dir: Path | str,
        target_label: Path | list[str] | None = None,
        source_label_path: Path | None = None,
        num_workers: int | None = None,
    ):
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)

        self.data_dir = data_dir

        if target_label is None:
            probe_path = self.data_dir / TARGET_LABEL_FILE_NAME
            if not probe_path.is_file():
                raise FileNotFoundError(
                    f"No target labels provided."
                    f" No label file exists at {probe_path}"
                )
            target_label = probe_path

        target_list: list[str]
        if isinstance(target_label, Path):
            with open(target_label) as target_label_file:
                target_label_raw = json.load(target_label_file)
                if not all(isinstance(obj, str) for obj in target_label_raw):
                    raise ValueError("Invalid value in target label.")

                target_list = target_label_raw
        else:
            target_list = target_label


        self.target_label = target_list

        self.source_label_path = source_label_path
        self.num_workers = num_workers

    @cached_property
    def available_parts(self) -> list[Path]:
        return self._discover_parts()

    @property
    def compatible(self) -> bool:
        return is_compatible_with_dataset(self.data_dir)

    def _discover_parts(self) -> list[Path]:
        logger.debug("Finding available parts from QEVD dataset.")
        part_paths = []
        for i in range(1, 5):
            part_dir = self.data_dir / (FOLDER_PATTERN + str(i))
            if part_dir.exists() and part_dir.is_dir():
                part_paths.append(part_dir)
            else:
                logger.warning(f"Missing part {i} of QEVD dataset.")

        return part_paths

    @staticmethod
    def _process_entry(
        entry: QEVDLabel,
        target_label_set: set[str],
        part_paths: list[Path],
        data_dir: Path,
        quarantine_dir: Path,
    ) -> None:
        target_class = QEVDAdapter._match_label(entry.labels, target_label_set)
        split_name = SPLIT_MAP.get(entry.split, entry.split)
        src_filename = Path(entry.video_path).name

        src_path = None
        for part_path in part_paths:
            candidate = part_path / src_filename
            if candidate.is_file():
                src_path = candidate
                break

        if src_path is None:
            src_path = data_dir / src_filename

        if not src_path.is_file():
            logger.warning(f"Video file not found: {src_filename}")
            return

        if not check_video_integrity(src_path):
            logger.warning(f"Quarantining corrupt video: {src_filename}")
            shutil.move(src_path, quarantine_dir / src_filename)
            return

        dst_path = data_dir / split_name / target_class / src_filename
        shutil.move(src_path, dst_path)

    def convert(self):
        if self.source_label_path is None:
            for part_path in self.available_parts:
                candidate_path = part_path / SOURCE_LABEL_FILE_NAME
                if candidate_path.is_file():
                    self.source_label_path = candidate_path
                    break

        if self.source_label_path is None:
            raise FileNotFoundError(
                "No source label file provided."
                "Automatic searching inside available parts failed"
            )

        with open(self.source_label_path) as source_label_file:
            source_label_raw = json.load(source_label_file)
            source_label_list = [QEVDLabel(**obj) for obj in source_label_raw]

        target_label_set = set(self.target_label)
        quarantine_dir = self.data_dir / QUARANTINE_DIR_NAME
        quarantine_dir.mkdir(parents=True, exist_ok=True)

        for split_name in SPLIT_DIRS:
            for label_name in self.target_label:
                (self.data_dir / split_name / label_name).mkdir(parents=True, exist_ok=True)

        part_paths = self.available_parts
        data_dir = self.data_dir
        n = len(source_label_list)

        process_map(
            QEVDAdapter._process_entry,
            source_label_list,
            [target_label_set] * n,
            [part_paths] * n,
            [data_dir] * n,
            [quarantine_dir] * n,
            max_workers=self.num_workers,
            chunksize=CHUNK_SIZE,
            desc="Converting QEVD",
        )

        shutil.move(self.source_label_path, self.data_dir)

    def load(self, cache_dir: Path | None = None) -> DatasetDict:
        if cache_dir is not None and cache_dir.is_dir():
            logger.info(f"Loading cached dataset from {cache_dir}")
            ds = load_from_disk(str(cache_dir))
            if not isinstance(ds, DatasetDict):
                raise TypeError(f"Expected DatasetDict, got {type(ds).__name__}")
            return ds

        ds = load_dataset("videofolder", data_dir=str(self.data_dir))
        if isinstance(ds, DatasetDict) and QUARANTINE_DIR_NAME in ds:
            ds.pop(QUARANTINE_DIR_NAME)

        if cache_dir is not None:
            logger.info(f"Saving dataset cache to {cache_dir}")
            ds.save_to_disk(str(cache_dir), num_proc=self.num_workers)

        return ds

    @staticmethod
    def _match_label(source_labels: list[str], target_label_set: set[str]) -> str:
        for source_label in source_labels:
            exercise = source_label.split(" - ", 1)[0]
            if exercise in target_label_set:
                return exercise
            normalized = re.sub(r"\s*\([^)]*\)\s*$", "", exercise).strip()
            if normalized in target_label_set:
                return normalized
        return "background"
