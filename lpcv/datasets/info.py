"""Constants shared across the dataset pipeline."""

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
"""Recognised video file extensions (lower-case, with leading dot)."""

SPLIT_DIRS = {"train", "val"}
"""Expected split directory names inside a videofolder dataset."""

TARGET_LABEL_FILE_NAME = "class_labels.json"
"""Default filename for the JSON list of target class labels."""

IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
"""ImageNet per-channel mean used for normalisation."""

IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]
"""ImageNet per-channel standard deviation used for normalisation."""
