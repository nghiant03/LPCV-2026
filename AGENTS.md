# AGENTS.md

## Environment Manager

This project uses **uv** for dependency and environment management.

- Python version: `3.12` (pinned in `.python-version`)
- Virtual environment: `.venv` (managed by uv)
- Install dependencies: `uv sync`
- Add a dependency: `uv add <package>`
- Add a dev dependency: `uv add --group dev <package>`
- Run a command in the environment: `uv run <command>`
- Project CLI entrypoint: `uv run lpcv`

## Code Quality

### Ruff (linter + formatter)

- Lint: `uv run ruff check .`
- Lint with autofix: `uv run ruff check --fix .`
- Format: `uv run ruff format .`
- Format check: `uv run ruff format --check .`
- Config: `pyproject.toml` under `[tool.ruff]`
- Line length: 100
- Selected rules: E, F, W, I, N, UP, B, SIM, TCH

### Pyright (type checker)

- Type check: `uv run pyright`
- Mode: `standard`
- Config: `pyproject.toml` under `[tool.pyright]`

## Mandatory: After Every Code Change

After **every** code change you must run all three checks and fix any issues before finishing:

```sh
uv run ruff format .
uv run ruff check --fix .
uv run pyright
```

Do not consider a task complete until all three pass cleanly. Update this file with any relevant information after a code change.

## Project Overview

LPCVC 2026 Track 2: Video Classification with Dynamic Frame Selection. Built around fine-tuning **VideoMAE** models on the QEVD (exercise/fitness video) dataset using HuggingFace Transformers + Trainer API.

## Project Structure

```
lpcv/
├── __init__.py
├── evaluation.py          # Top-k accuracy, H5 logit evaluation, full model evaluation
├── transforms.py          # Video transform registry (temporal, spatial, normalization)
├── cli/
│   ├── __init__.py
│   ├── main.py            # Typer app root, mounts sub-commands
│   ├── data.py            # CLI: convert, precompute, cache
│   ├── evaluate.py        # CLI: evaluate model or H5 logits
│   └── train.py           # CLI: train videomae
├── datasets/
│   ├── __init__.py
│   ├── info.py            # Constants: video extensions, split dirs, label file name
│   ├── precompute.py      # PrecomputedDataset: decode + resize + save frames to Arrow
│   ├── qevd.py            # QEVDAdapter: convert raw QEVD to videofolder, load/cache
│   └── utils.py           # Video probing, remuxing, dimension checks, frame subsampling
└── models/
    ├── __init__.py
    └── videomae.py         # VideoMAETrainerConfig + VideoMAEModelTrainer (HF Trainer wrapper)
```

## Architecture & Patterns

### CLI Layer (`lpcv/cli/`)

- Uses **Typer** with sub-app pattern: `main.py` mounts `data`, `train`, `evaluate` sub-commands.
- Heavy imports (torch, transformers, datasets) are deferred inside command functions to keep CLI startup fast.
- All CLI parameters use `Annotated[T, typer.Option/Argument]` style.

### Dataset Pipeline (`lpcv/datasets/`)

Two loading paths:

1. **Raw video** → `QEVDAdapter.load()` → applies `FromVideo` + transforms on-the-fly via `set_transform`.
2. **Precomputed** → `PrecomputedDataset.precompute()` saves decoded/resized frames as Arrow → `PrecomputedDataset.load()` applies `FromNumpy` + transforms via `set_transform`.

`QEVDAdapter.convert()` reorganizes raw QEVD parts into HuggingFace `videofolder` layout (`train/<class>/*.mp4`, `val/<class>/*.mp4`), quarantining corrupt or bad-dimension videos.

### Transform System (`lpcv/transforms.py`)

- Registry pattern: `@register("Name")` adds callable classes to `_REGISTRY`.
- `build_transform(steps)` constructs a `torchvision.transforms.Compose` from a list of `{"name": ..., **kwargs}` dicts.
- All transforms operate on `torch.Tensor` with shape `(T, C, H, W)`.
- Two presets: `TRAIN_PRESET` (with random augmentation) and `VAL_PRESET` (deterministic resize).
- Format transforms (`FromNumpy`, `FromVideo`) convert raw data into the expected tensor shape.

### Model Training (`lpcv/models/videomae.py`)

- `VideoMAETrainerConfig`: dataclass holding all training hyperparameters.
- `VideoMAEModelTrainer`: wraps HuggingFace `Trainer`. Handles model loading, freeze strategies (`none`/`backbone`/`partial`), custom collation, and top-1/top-5 metric computation.
- Label metadata is read from the dataset's `label` feature (ClassLabel).
- Multi-GPU via `torch.distributed.launcher.api.elastic_launch` (configured in CLI).

### Evaluation (`lpcv/evaluation.py`)

- `topk_accuracy()`: generic top-k accuracy on tensors.
- `evaluate_h5()`: loads logits from HDF5 (`data/0/sample_N` keys), compares against a JSONL manifest + class map JSON.
- `evaluate_model()`: end-to-end inference on a dataset using a saved model checkpoint.

## Coding Conventions

- Use `from __future__ import annotations` for deferred type evaluation where present.
- Use `TYPE_CHECKING` guard for import-heavy type hints (torch, datasets, etc.).
- Logging via **loguru** (`from loguru import logger`), not stdlib logging.
- Path handling: accept `str | Path`, convert internally with `Path()`.
- Use **pydantic** `BaseModel` for structured data validation (e.g., `QEVDLabel`).
- Constants live in `lpcv/datasets/info.py`.
- No relative imports; all imports are absolute from `lpcv.*`.

## Key Dependencies

| Package       | Purpose                              |
|---------------|--------------------------------------|
| torch         | Core tensor ops, model training      |
| transformers  | VideoMAE model, Trainer, processors  |
| datasets      | HuggingFace dataset loading/caching  |
| av (PyAV)     | Video decoding, probing, remuxing    |
| torchvision   | Compose, image transforms            |
| typer         | CLI framework                        |
| loguru        | Structured logging                   |
| pydantic      | Data validation                      |
| h5py          | HDF5 logit file I/O                  |
| accelerate    | Distributed training support         |
