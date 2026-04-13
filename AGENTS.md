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

### Model Architecture Requirements

When developing or evaluating new model architectures, they **must** meet the following on-device constraint:

- **Inference latency < 34 ms** on the target device (Dragonwing IQ-9075 EVK) as reported by `uv run lpcv submit validate`.

Architectures that exceed this budget should not be merged. Use the validate subcommand to verify before committing.

### Checking Qualcomm AI Hub Jobs

Always check QAI Hub job status and results using the **Python SDK**, not the web UI. Example:

```python
import qai_hub

job = qai_hub.get_job("<job_id>")
print(job.get_status())

# For profile jobs:
profile = job.download_profile()
print(profile["execution_summary"]["estimated_inference_time"])  # microseconds

# For compile jobs:
job.download_target_model("output.bin")

# For inference jobs:
output = job.download_output_data()
```

Use `qai_hub.get_job()` to inspect any job by ID. Profile times are in **microseconds**.

## Project Structure

```
lpcv/
├── __init__.py
├── evaluation.py          # Top-k accuracy, H5 logit evaluation, full model evaluation
├── submission.py          # Competition adapter, ONNX export, Qualcomm AI Hub compile/infer/profile
├── transforms.py          # Video transform registry (temporal, spatial, normalization)
├── cli/
│   ├── __init__.py
│   ├── main.py            # Typer app root, mounts sub-commands
│   ├── data.py            # CLI: convert raw QEVD to videofolder
│   ├── evaluate.py        # CLI: evaluate model or H5 logits
│   ├── train.py           # CLI: single `train` command, reads model config from YAML
│   └── submit.py          # CLI: preprocess, export, compile, profile, infer, validate on Qualcomm AI Hub
├── datasets/
│   ├── __init__.py
│   ├── info.py            # Constants: video extensions, split dirs, label file name
│   ├── base.py            # VideoDataset (PyTorch Dataset) + load_video_dataset()
│   ├── decoder.py         # VideoDecoder protocol + PyAV/TorchCodec implementations
│   ├── qevd.py            # QEVDAdapter: convert raw QEVD to videofolder layout
│   └── utils.py           # Video probing, remuxing, dimension checks, frame subsampling
└── models/
    ├── __init__.py        # ModelSpec registry + get_model_spec() / register_model() + YAML config load/save
    ├── base.py            # ModelOutput, BaseForClassification, BaseModelTrainer (shared)
    ├── videomae.py        # VideoMAETrainerConfig + VideoMAEModelTrainer (HF Trainer wrapper)
    ├── r2plus1d.py        # R2Plus1DTrainerConfig + R2Plus1DModelTrainer + R2Plus1DForClassification
    ├── x3d.py             # X3DTrainerConfig + X3DModelTrainer + X3DForClassification (torch hub)
    ├── tsm.py             # TSMTrainerConfig + TSMModelTrainer + TSMForClassification (temporal shift)
    ├── mvitv2.py          # MViTv2TrainerConfig + MViTv2ModelTrainer + MViTv2ForClassification (torchvision)
    └── stam.py            # STAMTrainerConfig + STAMModelTrainer + STAMForClassification (space-time attention)
```

## Architecture & Patterns

### CLI Layer (`lpcv/cli/`)

- Uses **Typer** with sub-app pattern: `main.py` mounts `train` and `convert` as direct commands, `evaluate` and `submit` as sub-groups (nested Typer apps).
- Heavy imports (torch, transformers, datasets) are deferred inside command functions to keep CLI startup fast.
- All CLI parameters use `Annotated[T, typer.Option/Argument]` style.
- The `train` CLI exposes `--gradient-checkpointing/--no-gradient-checkpointing` for every trainer via the shared `BaseTrainerConfig`.
- `train --decoder torchcodec-nvdec` validates that `--num-workers` matches `--num-gpus` so each decode worker maps cleanly to one GPU, and forces DataLoader workers to use the `spawn` start method because CUDA/NVDEC initialization is unsafe in forked workers.
- **Model config via YAML**: Both `train` and `submit validate` accept a YAML config file as the first positional argument. The YAML defines model architecture params (e.g. `model`, `num_frames`, `crop_size`). Training hyperparams (epochs, lr, batch size, etc.) remain CLI options. Default configs live in `configs/<model>.yaml`.
- **WARNING**: Functions passed to `elastic_launch` (multi-GPU training) **must** be defined at module level, not as closures or nested functions. Nested functions cannot be pickled by `torch.distributed`'s spawn-based multiprocessing and will raise `AttributeError: Can't pickle local object`. The `_launch()` function in `train.py` is intentionally at module level for this reason.

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
- `torchcodec` is a required dependency (pinned `<0.11` in `pyproject.toml`).

### Transform System (`lpcv/transforms.py`)

- Normalisation statistics live here as module-level constants (`R2PLUS1D_MEAN/STD`, `IMAGENET_MEAN/STD`, `X3D_MEAN/STD`), co-located with the transforms that consume them.
- Registry pattern: `@register("Name")` adds callable classes to `_REGISTRY`.
- `build_transform(steps)` constructs a `torchvision.transforms.Compose` from a list of `{"name": ..., **kwargs}` dicts.
- `get(name)` retrieves a registered transform class by name.
- All transforms operate on `torch.Tensor` with shape `(T, C, H, W)`.
- `VideoTransformCallable` wraps a `Compose` pipeline for use with HF `set_transform`.
- Default presets match the LPCVC reference solution (R2+1D norm, 128×171 resize, 112×112 crop):
  - `COMPETITION_PRESET` — ScalePixels → Resize(128,171) → Normalize(R2+1D) → CenterCrop(112). Single source of truth for the competition's fixed pipeline; used by `preprocess_dataset()` and export-contract generation.
  - `TRAIN_PRESET` — ScalePixels → Resize(128,171) → RandomHorizontalFlip → Normalize(R2+1D) → RandomCrop(112).
  - `VAL_PRESET` — alias for `COMPETITION_PRESET`.
- `make_presets(mean, std, resize_height, resize_width, crop_size)` — generic preset builder with customizable normalization and spatial sizes; module-level `TRAIN_PRESET` and `VAL_PRESET` are produced by calling `make_presets()` with R2+1D defaults.
- `save_val_transform_config()` / `load_val_transform_config()` — persist val transform config as JSON alongside model checkpoints.
- `build_export_config()` / `save_export_config()` / `load_export_config()` — convert a supported val config into an explicit ONNX adapter contract (`export_config.json`) and persist it alongside checkpoints/exports.
- Registered transforms: `FromVideo`, `UniformTemporalSubsample`, `ScalePixels`, `Normalize`, `RandomShortSideScale`, `ShortSideScale`, `RandomCrop`, `CenterCrop`, `RandomHorizontalFlip`, `Resize`.

### Model Registry (`lpcv/models/__init__.py`)

- `ModelSpec` dataclass: bundles default presets, config class, trainer class, loader, input layout, input key, output extractor, throwaway builder, plus model-specific config resolution / preset-building / `num_frames` resolution hooks for each registered model.
- `register_model(name, spec)` / `get_model_spec(name)` / `list_models()` — lightweight registry.
- `resolve_model_config(model_name, raw_config)` — resolves defaults and validation through the registry into an explicit model config plus derived train/val presets and `num_frames`.
- `load_model_config(path)` / `save_model_config(config, output_dir)` — YAML load/save for model architecture config (`model_config.yaml`); passing a saved model directory loads `model_config.yaml` from the artifact root.
- Saved-artifact metadata filenames are centralized here: `model_config.yaml`, `val_transform.json`, and `export_config.json`.
- `model_config_from_trainer(model_name, config)` — extracts model-specific fields from a trainer config dataclass.
- Built-in registrations: `"videomae"`, `"r2plus1d"`, `"x3d"`, `"tsm"`, `"mvitv2"`, `"stam"`.
- `resolve_artifact_model_name(model_path, model_name)` — infers model name from a saved artifact directory, with optional override.
- CLI `train.py` uses the registry via `_run_training()` — a single generic flow that resolves the model config through the registry, builds datasets with the resolved presets, and delegates to the spec's trainer class.

### Model Training (`lpcv/models/videomae.py`)

- `VideoMAETrainerConfig(BaseTrainerConfig)`: adds `model_name`, `num_frames`; overrides defaults (15 epochs, 5e-5 LR, linear scheduler). Gradient checkpointing now comes from shared `BaseTrainerConfig`.
- `VideoMAEModelTrainer(BaseModelTrainer)`: wraps HuggingFace `Trainer`. Handles model loading via `from_pretrained`, freeze strategies (`none`/`backbone`/`partial`), BTCHW collation (no permute), and `trainer.save_model()` for checkpoints.
- Multi-GPU via `torch.distributed.launcher.api.elastic_launch` (configured in CLI).

### Shared Model Base (`lpcv/models/base.py`)

- `ModelOutput`: lightweight output wrapper with `.loss` and `.logits` — supports attribute, dict, and index access for HF Trainer compatibility.
- `BaseTrainerConfig`: shared training hyperparameters (~29 fields) inherited by all model configs. Subclasses add model-specific fields and override defaults.
- `compute_metrics()`: shared top-1/top-5 accuracy computation.
- `collate_for_video()`: shared collation with optional `(T,C,H,W)` → `(C,T,H,W)` permutation.
- `log_freeze_stats()`: shared parameter-count logging after freezing.
- `decompose_depthwise_conv3d()`: replaces all depthwise `Conv3d` modules with `DecomposedDepthwiseConv3d` (spatial 2D + temporal 1D decomposition for Qualcomm AI Hub compatibility).
- `BaseForClassification(nn.Module)`: base for custom classification wrappers — provides shared `forward()`, HuggingFace-compatible `gradient_checkpointing_enable()` / `gradient_checkpointing_disable()`, `save_pretrained()`, and `_extra_save_meta()` hook. Checkpointing is applied generically around the wrapped backbone during training.
- `BaseModelTrainer`: shared HF Trainer wrapper — reads `train_dataset.label_names`, handles TF32/cuDNN setup, `train()` loop, val-transform saving, model config saving (`model_config.yaml`), export-contract saving (`export_config.json`), `ddp_find_unused_parameters=False`, and custom DataLoader multiprocessing context overrides when required by a decoder/runtime. Subclasses override `_init_model()`, `_apply_freeze_strategy()`, and optionally `_collate_fn()`, `_extra_training_args()`, `_save_model()`.

### Model Training (`lpcv/models/r2plus1d.py`)

- `R2Plus1DTrainerConfig(BaseTrainerConfig)`: adds `num_classes`, `num_frames`, `label_smoothing`; overrides defaults (2 epochs, cosine LR, 1e-2 LR, partial freeze).
- `R2Plus1DForClassification(BaseForClassification)`: wraps torchvision `r2plus1d_18` with HF Trainer-compatible interface.
- `R2Plus1DModelTrainer(BaseModelTrainer)`: overrides `_init_model()` and `_apply_freeze_strategy()` for R2+1D-specific freezing (`layer4` + `fc`).

### Model Training (`lpcv/models/x3d.py`)

- `X3D_PRESET_DEFAULTS`: maps preset names (`x3d_xs`/`x3d_s`/`x3d_m`/`x3d_l`) to default `num_frames` and `crop_size` (from pytorchvideo hub: 160/160/224/312).
- `X3DTrainerConfig(BaseTrainerConfig)`: adds `preset`, `crop_size`, `num_frames`, `label_smoothing`; `resolved_num_frames()`/`resolved_crop_size()` fall back to preset defaults when 0.
- `X3DForClassification(BaseForClassification)`: loads X3D from `facebookresearch/pytorchvideo` via `torch.hub`, replaces the head projection with a custom linear layer, swaps fixed `AvgPool3d` with `AdaptiveAvgPool3d((1,1,1))` for resolution-agnostic pooling.
- `X3DModelTrainer(BaseModelTrainer)`: overrides `_init_model()` and `_apply_freeze_strategy()` for X3D-specific freezing (`blocks.4` + `blocks.5`).

### Model Training (`lpcv/models/tsm.py`)

- **Temporal Shift Module (TSM)** — efficient video understanding using only 2D convolutions, no Conv3d.
- `TSMTrainerConfig(BaseTrainerConfig)`: adds `backbone` (`resnet18`/`resnet50`), `num_classes`, `num_frames` (default 8), `shift_div`, `shift_last_n`, `label_smoothing`; overrides defaults (10 epochs, cosine LR, 1e-2 LR, partial freeze).
- `TSMForClassification(BaseForClassification)`: wraps torchvision ResNet with `TemporalShiftWrapper` that injects `torch.roll`-based channel shifting into selected residual blocks. Input `(B,C,T,H,W)` → permute+reshape to `(B*T,C,H,W)` → 2D ResNet → temporal mean pooling.
- `TSMModelTrainer(BaseModelTrainer)`: overrides `_init_model()` and `_apply_freeze_strategy()` — `"partial"` freezes all except `layer4` + `fc`.

### Model Training (`lpcv/models/stam.py`)

- **STAM = Space-Time Attention Model** — two-stage ViT: per-frame spatial ViT (ImageNet-21k pretrained via `timm`) → temporal TransformerEncoder → linear head. No 3D convolutions.
- `STAMTrainerConfig(BaseTrainerConfig)`: adds `num_classes`, `num_frames` (default 16), `crop_size` (default 112), `patch_size`, `embed_dim`, `spatial_depth`, `num_heads`, `temporal_layers`, `label_smoothing`; defaults to 1e-4 LR and `"partial"` freeze.
- `STAMForClassification(BaseForClassification)`: `SpatialViT` (ViT-B/16 with learnable CLS + positional embeddings, 12 transformer blocks) produces per-frame CLS tokens → `_TemporalAggregate` (temporal CLS + positional embeddings, 6-layer TransformerEncoder) → video-level embedding.
- `STAMModelTrainer(BaseModelTrainer)`: overrides `_init_model()` and `_apply_freeze_strategy()` — `"partial"` freezes patch embed + blocks 0–7, trains blocks 8–11 + norm + temporal + head.

### Model Training (`lpcv/models/mvitv2.py`)

- `MVITV2_NUM_FRAMES = 16`, `MVITV2_CROP_SIZE = 112`: defaults matching the competition pipeline.
- `_build_mvitv2()`: constructs MViTv2-S via `torchvision.models.video.mvit.MViT` directly (bypasses the factory to allow custom `spatial_size`). Loads Kinetics-400 pretrained weights with bicubic interpolation of `rel_pos_h`/`rel_pos_w` tables when spatial size ≠ 224.
- `_interpolate_rel_pos()`: resizes relative positional embedding tables via bilinear interpolation to match the target spatial resolution.
- `MViTv2TrainerConfig(BaseTrainerConfig)`: adds `num_classes`, `num_frames`, `crop_size`, `label_smoothing`; defaults to `1e-4` LR and `"partial"` freeze.
- `MViTv2ForClassification(BaseForClassification)`: wraps torchvision MViTv2-S with resolution-aware weight loading. Replaces the classification head for the target number of classes.
- `MViTv2ModelTrainer(BaseModelTrainer)`: overrides `_init_model()` and `_apply_freeze_strategy()` — `"partial"` freezes all but the last 4 transformer blocks (12–15), norm, and head.

### Submission Pipeline (`lpcv/submission.py`)

- `preprocess_dataset()`: decodes videos to `(1,3,T,112,112)` `.npy` tensors using competition's fixed R(2+1)D normalization pipeline. Writes a `manifest.jsonl` mapping tensor paths to labels.
- `CompetitionAdapter`: `nn.Module` that adapts competition input (BCTHW, 112×112, R2+1D norm) to any model's expected format. Handles only the supported deterministic export contract: optional re-normalization, resize, center crop, and layout permutation. It is built from `export_config.json`, not by diffing transform step lists.
- `export_onnx()`: loads model via `ModelSpec.loader`, infers model type / `num_frames` from the saved artifact by default, wraps with `CompetitionAdapter`, exports ONNX with external data, and writes `export_config.json` into the ONNX output directory.
- `compile_on_hub()`: uploads ONNX to Qualcomm AI Hub, infers `num_frames` from the exported ONNX metadata by default, submits compile job targeting `qnn_context_binary`, downloads `.bin`.
- `profile_on_hub()`: submits a profile job for a compiled `.bin` or existing AI Hub model; returns job URL.
- `validate_on_hub()`: end-to-end smoke test — builds throwaway model via registry → export ONNX → compile → profile on AI Hub.
- `run_inference_on_hub()`: loads tensors from manifest, uploads in chunks of 538 (≤2GB flatbuffer limit), submits inference jobs, collects logits into HDF5.
- Key constants: `DEFAULT_NUM_FRAMES = 16`, `DEFAULT_DEVICE_NAME = "Dragonwing IQ-9075 EVK"`, `CHUNK_SIZE = 538`, `FRAME_RATE = 4`, `COMPETITION_SPATIAL_SIZE = 112`.

### Submission CLI (`lpcv/cli/submit.py`)

Six subcommands under `uv run lpcv submit`:

| Command | Usage | Description |
|---|---|---|
| `preprocess` | `<data_dir> <output_dir>` | Decode videos → `.npy` tensors + `manifest.jsonl` |
| `export` | `<model_path> -o model.onnx` | Export checkpoint to ONNX with competition adapter; model type and `num_frames` default to saved artifact metadata |
| `compile` | `<onnx_path> -o export_assets/` | Compile ONNX on Qualcomm AI Hub → `.bin`; `num_frames` defaults to exported ONNX metadata |
| `profile` | `<compiled.bin>` | Submit a profile job for a compiled model on AI Hub |
| `infer` | `<tensor_dir> -c <compiled.bin>` | Upload tensors, run on-device inference on AI Hub |
| `validate` | `<config.yaml>` | Resolve model config through the registry, build a throwaway model, export, compile, and profile on AI Hub |

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
- `evaluate_model()`: end-to-end inference on a dataset using any registered saved model checkpoint; batching/collation follows the model registry's input layout and key.

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
| torchvision   | Compose, video models, transforms    |
| torchcodec    | Video decoding (CPU / NVDEC GPU)     |
| typer         | CLI framework                        |
| loguru        | Structured logging                   |
| pydantic      | Data validation                      |
| h5py          | HDF5 logit file I/O                  |
| accelerate    | Distributed training support         |
| tqdm          | Progress bars                        |
| pillow        | Image processing support             |
| evaluate      | HuggingFace evaluation metrics       |
| timm          | Pretrained ViT weights (STAM)        |
| fvcore        | PyTorchVideo / X3D model support     |
| onnxscript    | ONNX export support                  |
| pyyaml        | YAML config loading/saving           |
| qai-hub       | Qualcomm AI Hub SDK                  |
