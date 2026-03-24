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

R2PLUS1D_MEAN: list[float] = [0.43216, 0.394666, 0.37645]
"""R(2+1)D per-channel mean used by the LPCVC competition preprocessing."""

R2PLUS1D_STD: list[float] = [0.22803, 0.22145, 0.216989]
"""R(2+1)D per-channel standard deviation used by the LPCVC competition preprocessing."""

X3D_MEAN: list[float] = [0.45, 0.45, 0.45]
"""X3D per-channel mean (Kinetics-400 convention)."""

X3D_STD: list[float] = [0.225, 0.225, 0.225]
"""X3D per-channel standard deviation (Kinetics-400 convention)."""
