# LPCV-2026

LPCVC 2026 Track 2: video classification with dynamic frame selection.

This repository trains and evaluates video classifiers on the QEVD exercise dataset and
includes an end-to-end Qualcomm AI Hub submission pipeline for export, compilation,
profiling, and on-device inference.

## What Is Included

- QEVD conversion into a `videofolder`-style dataset layout
- Training via a single `lpcv train` CLI with YAML model configs
- Evaluation from either saved checkpoints or HDF5 logits
- ONNX export with a competition adapter
- Qualcomm AI Hub compile, profile, validate, and inference commands

Supported model families in the current registry:

- `videomae`
- `r2plus1d`
- `x3d`
- `tsm`
- `mvitv2`
- `stam`

## Environment

The project uses `uv` and targets Python `3.12`.

```bash
uv sync
```

Common commands:

```bash
uv run lpcv --help
uv run ruff format .
uv run ruff check --fix .
uv run pyright
```

## Dataset Layout

Training and evaluation operate on a converted videofolder layout:

```text
data/qevd/
  class_labels.json
  train/
    class_a/
      clip_001.mp4
  val/
    class_a/
      clip_002.mp4
```

`class_labels.json` must be a JSON list of class names. The dataset loader reads videos from
`train/<class>/` and `val/<class>/`.

## Convert Raw QEVD Data

If you start from the raw QEVD release, convert it first:

```bash
uv run lpcv convert ./data/qevd_raw
```

Useful options:

- `--target-label` to point at a custom target label list
- `--source-label` to point at `fine_grained_labels.json`
- `--num-workers` to control parallel conversion

The converter looks for QEVD parts named `QEVD-FIT-300k-Part-1` through `QEVD-FIT-300k-Part-4`,
validates video integrity and dimensions, maps the QEVD `test` split to `val`, and moves invalid
videos into `quarantine/`.

## Model Configs

Model architecture settings live in YAML files under [`configs/`](./configs):

- [`configs/videomae.yaml`](./configs/videomae.yaml)
- [`configs/r2plus1d.yaml`](./configs/r2plus1d.yaml)
- [`configs/x3d.yaml`](./configs/x3d.yaml)
- [`configs/tsm.yaml`](./configs/tsm.yaml)
- [`configs/mvitv2.yaml`](./configs/mvitv2.yaml)
- [`configs/stam.yaml`](./configs/stam.yaml)

`lpcv train` accepts either `--config`, `--model`, or both. If both are supplied, `--model`
overrides the `model` key in the YAML.

## Training

Train from a config:

```bash
uv run lpcv train ./data/qevd \
  --config configs/x3d.yaml \
  --output-dir model/x3d \
  --epochs 20 \
  --batch-size 8 \
  --num-gpus 1
```

Train by naming the architecture directly:

```bash
uv run lpcv train ./data/qevd \
  --model r2plus1d \
  --output-dir model/r2plus1d
```

Important options:

- `--decoder {pyav,torchcodec-cpu,torchcodec-nvdec}`
- `--num-gpus` for elastic multi-GPU training
- `--data-percent` for stratified subsampling
- `--grad-checkpoint`, `--freeze`, `--lr-scheduler`, `--compile`, `--tf32`

Notes:

- The default decoder is `torchcodec-nvdec`.
- When using `torchcodec-nvdec`, the training CLI expects `--num-workers == --num-gpus` so each
  decode worker maps cleanly to one GPU.
- Saved artifacts include `model_config.yaml`, `val_transform.json`, and `export_config.json`
  alongside the trained model.

## Evaluation

Evaluate a saved checkpoint on the validation split:

```bash
uv run lpcv evaluate model ./data/qevd ./model/x3d
```

Evaluate HDF5 logits produced by the submission pipeline:

```bash
uv run lpcv evaluate h5 ./result/dataset-export.h5 ./tensors/manifest.jsonl \
  --class-map ./class_map.json
```

## Submission Pipeline

The submission flow mirrors the competition preprocessing and wraps exported models with a
competition adapter before ONNX export.

### 1. Preprocess Validation Videos

```bash
uv run lpcv submit preprocess ./data/qevd ./tensors
```

This writes `.npy` tensors and `manifest.jsonl` using the fixed competition preprocessing
pipeline.

### 2. Export To ONNX

```bash
uv run lpcv submit export ./model/x3d -o ./export/model.onnx
```

### 3. Compile On Qualcomm AI Hub

```bash
uv run lpcv submit compile ./export/model.onnx -o ./export_assets
```

### 4. Profile The Compiled Model

```bash
uv run lpcv submit profile ./export_assets/compiled.bin
```

### 5. Run On-Device Inference

```bash
uv run lpcv submit infer ./tensors -c ./export_assets/compiled.bin -o ./result/logits.h5
```

### 6. Validate A Model Config Against The Device Budget

```bash
uv run lpcv submit validate ./configs/x3d.yaml
```

Target-device requirement:

- Inference latency must stay below `34 ms` on the Dragonwing IQ-9075 EVK.

Submission commands require working Qualcomm AI Hub credentials in your environment.

## Typical End-To-End Workflow

```bash
# 1. Convert raw QEVD into videofolder format
uv run lpcv convert ./data/QEVD

# 2. Train a model
uv run lpcv train ./data/QEVD --config configs/x3d.yaml --output-dir model/x3d

# 3. Evaluate the trained checkpoint locally
uv run lpcv evaluate model ./data/sQEVD ./model/x3d

# 4. Export, compile, and profile for the target device
uv run lpcv submit export ./model/x3d -o ./export/model.onnx
uv run lpcv submit compile ./export/model.onnx -o ./export_assets
uv run lpcv submit profile ./export_assets/compiled.bin
```

## Development Notes

- Lint and format with Ruff.
- Type-check with Pyright.
- Public APIs use NumPy-style docstrings.
- Prefer `uv run ...` for all project commands so they run inside the managed environment.
