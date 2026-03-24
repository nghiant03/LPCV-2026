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

## Documentation Style

All source files use **NumPy-style docstrings** (compatible with `mkdocstrings-python`). Every public module, class, function, and constant has a docstring. Private methods with significant logic also have docstrings. When adding or modifying code, maintain this convention.

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
│   ├── data.py            # CLI: convert raw QEVD to videofolder
│   ├── evaluate.py        # CLI: evaluate model or H5 logits
│   ├── train.py           # CLI: train videomae / r2plus1d (generic flow)
│   └── submit.py          # CLI: preprocess, export, compile, infer on Qualcomm AI Hub
├── datasets/
│   ├── __init__.py
│   ├── info.py            # Constants: video extensions, split dirs, label file name
│   ├── base.py            # VideoDataset (PyTorch Dataset) + load_video_dataset()
│   ├── decoder.py         # VideoDecoder protocol + PyAV/TorchCodec implementations
│   ├── qevd.py            # QEVDAdapter: convert raw QEVD to videofolder layout
│   └── utils.py           # Video probing, remuxing, dimension checks, frame subsampling
└── models/
    ├── __init__.py         # ModelSpec registry + get_model_spec() / register_model()
    ├── videomae.py         # VideoMAETrainerConfig + VideoMAEModelTrainer (HF Trainer wrapper)
    └── r2plus1d.py         # R2Plus1DTrainerConfig + R2Plus1DModelTrainer + R2Plus1DForClassification
```

## Architecture & Patterns

### CLI Layer (`lpcv/cli/`)

- Uses **Typer** with sub-app pattern: `main.py` mounts `data`, `train`, `evaluate`, `submit` sub-commands.
- Heavy imports (torch, transformers, datasets) are deferred inside command functions to keep CLI startup fast.
- All CLI parameters use `Annotated[T, typer.Option/Argument]` style.

### Dataset Pipeline (`lpcv/datasets/`)

Two-stage pipeline:

1. **Convert** — `QEVDAdapter.convert()` discovers raw QEVD parts, validates video integrity/dimensions, matches labels via Pydantic-validated `QEVDLabel`, and reorganizes files into HuggingFace `videofolder` layout (`train/<class>/*.mp4`, `val/<class>/*.mp4`). Corrupt or bad-dimension videos are quarantined.

2. **Load** — `load_video_dataset()` in `base.py` builds train/val `VideoDataset` instances from the videofolder layout. Each `VideoDataset` is a standard PyTorch `Dataset` that delegates frame decoding to an injected `VideoDecoder` and applies transforms. Returns `{"pixel_values": Tensor, "labels": int}` dicts.

### Decoder System (`lpcv/datasets/decoder.py`)

- `VideoDecoder` protocol: `decode(path, num_frames) -> Tensor (T, C, H, W)`.
- Three implementations: `PyAVDecoder` (full decode + subsample), `TorchCodecCPUDecoder` (seek-based CPU), `TorchCodecNVDECDecoder` (GPU/NVDEC).
- `DECODERS` registry dict maps string names (`"pyav"`, `"torchcodec-cpu"`, `"torchcodec-nvdec"`) to classes.
- `get_decoder(name, **kwargs)` factory function instantiates by name.
- All decoders use `uniform_temporal_indices()` from `utils.py` for frame subsampling.
- Note: `torchcodec` is an optional dependency not listed in `pyproject.toml`.

### Transform System (`lpcv/transforms.py`)

- Registry pattern: `@register("Name")` adds callable classes to `_REGISTRY`.
- `build_transform(steps)` constructs a `torchvision.transforms.Compose` from a list of `{"name": ..., **kwargs}` dicts.
- `get(name)` retrieves a registered transform class by name.
- All transforms operate on `torch.Tensor` with shape `(T, C, H, W)`.
- `VideoTransformCallable` wraps a `Compose` pipeline for use with HF `set_transform`.
- Default presets match the LPCVC reference solution (R2+1D norm, 128×171 resize, 112×112 crop):
  - `COMPETITION_PRESET` — ScalePixels → Resize(128,171) → Normalize(R2+1D) → CenterCrop(112). Single source of truth for the competition's fixed pipeline; used by `preprocess_dataset()` and `extract_adapter_steps()`.
  - `TRAIN_PRESET` — ScalePixels → Resize(128,171) → RandomHorizontalFlip → Normalize(R2+1D) → RandomCrop(112).
  - `VAL_PRESET` — alias for `COMPETITION_PRESET`.
- `save_val_transform_config()` / `load_val_transform_config()` — persist val transform config as JSON alongside model checkpoints.
- `extract_adapter_steps()` — diffs a saved val config against the competition pipeline; returns only the extra steps needed for the ONNX adapter.
- Registered transforms: `FromVideo`, `UniformTemporalSubsample`, `ScalePixels`, `Normalize`, `RandomShortSideScale`, `ShortSideScale`, `RandomCrop`, `CenterCrop`, `RandomHorizontalFlip`, `Resize`.

### Model Registry (`lpcv/models/__init__.py`)

- `ModelSpec` dataclass: bundles train/val presets, config class, trainer class, loader, input layout, input key, and output extractor for each model.
- `register_model(name, spec)` / `get_model_spec(name)` / `list_models()` — lightweight registry.
- Built-in registrations: `"videomae"`, `"r2plus1d"`.
- CLI `train.py` uses the registry via `_run_training()` — a single generic flow that loads the spec, builds datasets with the model's presets, and delegates to the spec's trainer class.

### Model Training (`lpcv/models/videomae.py`)

- `VideoMAETrainerConfig`: dataclass holding all training hyperparameters (~30 fields).
- `VideoMAEModelTrainer`: wraps HuggingFace `Trainer`. Handles model loading, freeze strategies (`none`/`backbone`/`partial`), custom collation via `_collate_fn()`, and top-1/top-5 metric computation via `_compute_metrics()`.
- Accepts `val_transform_config` and saves it as `val_transform.json` alongside the model.
- Label metadata is read from the dataset's `label` feature (ClassLabel).
- Multi-GPU via `torch.distributed.launcher.api.elastic_launch` (configured in CLI).

### Model Training (`lpcv/models/r2plus1d.py`)

- `R2Plus1DTrainerConfig`: dataclass with defaults optimised for few-epoch fine-tuning (2 epochs, cosine LR, label smoothing 0.1, partial freeze).
- `R2Plus1DForClassification`: wraps torchvision `r2plus1d_18` with HF Trainer-compatible interface (`pixel_values`/`labels` kwargs, `.logits` output).
- `R2Plus1DModelTrainer`: same pattern as VideoMAE — HF `Trainer` wrapper with freeze strategies (`none`/`backbone`/`partial`), label smoothing, and `val_transform.json` saving.
- `save_pretrained()` / `load_pretrained()` — custom checkpoint format (`model.pt` with state dict + num_classes).

### Submission Pipeline (`lpcv/submission.py`)

- `preprocess_dataset()`: decodes videos to `(1,3,T,112,112)` `.npy` tensors using competition's fixed R(2+1)D normalization pipeline. Writes a `manifest.jsonl` mapping tensor paths to labels.
- `CompetitionAdapter`: `nn.Module` that adapts competition input (BCTHW, 112×112, R2+1D norm) to any model's expected format. Handles re-normalization, resize, and layout permutation in-graph. Auto-built from `val_transform.json` via `CompetitionAdapter.from_saved_config()` — no hardcoded registry needed.
- `export_onnx()`: loads model via `ModelSpec.loader`, wraps with `CompetitionAdapter`, exports ONNX with external data.
- `compile_on_hub()`: uploads ONNX to Qualcomm AI Hub, submits compile job targeting `qnn_context_binary`, downloads `.bin`.
- `run_inference_on_hub()`: loads tensors from manifest, uploads in chunks of 538 (≤2GB flatbuffer limit), submits inference jobs, collects logits into HDF5.
- Key constants: `DEFAULT_DEVICE_NAME = "Dragonwing IQ-9075 EVK"`, `CHUNK_SIZE = 538`, `FRAME_RATE = 4`, `COMPETITION_SPATIAL_SIZE = 112`.

### Submission CLI (`lpcv/cli/submit.py`)

Four subcommands under `uv run lpcv submit`:

| Command | Usage | Description |
|---|---|---|
| `preprocess` | `<data_dir> <output_dir>` | Decode videos → `.npy` tensors + `manifest.jsonl` |
| `export` | `<model_path> -o model.onnx` | Export checkpoint to ONNX with competition adapter |
| `compile` | `<onnx_path> -o export_assets/` | Compile ONNX on Qualcomm AI Hub → `.bin` |
| `infer` | `<tensor_dir> -c <compiled.bin>` | Upload tensors, run on-device inference on AI Hub |

Typical end-to-end submission workflow:

```sh
# 1. Preprocess validation videos into tensors
uv run lpcv submit preprocess ./data/qevd ./tensors

# 2. Export trained model to ONNX (auto-wraps with CompetitionAdapter)
uv run lpcv submit export ./checkpoints/best -o model.onnx

# 3. Compile ONNX for Qualcomm hardware
uv run lpcv submit compile model.onnx -o export_assets/

# 4. Run on-device inference and collect logits
uv run lpcv submit infer ./tensors -c export_assets/compiled.bin

# 5. Evaluate logits against ground truth
uv run lpcv evaluate h5 ./inference_results.h5 ./tensors/manifest.jsonl ./class_map.json
```

### Evaluation (`lpcv/evaluation.py`)

- `topk_accuracy()`: generic top-k accuracy on tensors.
- `load_logits_h5()`: loads logits from HDF5 files.
- `load_labels_from_manifest()`: reads ground-truth labels from JSONL manifest + class map JSON.
- `evaluate_h5()`: end-to-end evaluation from H5 logit files.
- `evaluate_model()`: end-to-end inference on a dataset using a saved model checkpoint.

## Coding Conventions

- Use `from __future__ import annotations` for deferred type evaluation where present.
- Use `TYPE_CHECKING` guard for import-heavy type hints (torch, datasets, etc.).
- Logging via **loguru** (`from loguru import logger`), not stdlib logging.
- Path handling: accept `str | Path`, convert internally with `Path()`.
- Use **pydantic** `BaseModel` for structured data validation (e.g., `QEVDLabel`).
- Constants live in `lpcv/datasets/info.py`.
- No relative imports; all imports are absolute from `lpcv.*`.
- NumPy-style docstrings on all public APIs.

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
| tqdm          | Progress bars                        |
| pillow        | Image processing support             |
| evaluate      | HuggingFace evaluation metrics       |

### Optional

| Package     | Purpose                                      |
|-------------|----------------------------------------------|
| torchcodec  | Alternative video decoding (CPU / NVDEC GPU) |
