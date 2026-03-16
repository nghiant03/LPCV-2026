"""Constants shared across the dataset pipeline."""

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
"""Recognised video file extensions (lower-case, with leading dot)."""

SPLIT_DIRS = {"train", "val"}
"""Expected split directory names inside a videofolder dataset."""

TARGET_LABEL_FILE_NAME = "class_labels.json"
"""Default filename for the JSON list of target class labels."""
