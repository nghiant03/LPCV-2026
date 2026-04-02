"""CLI sub-commands for model training.

Each model type has its own sub-command (``videomae``, ``r2plus1d``, ``x3d``,
``tsm``) that accepts model-specific options.  All commands share the same
data-loading and training flow via :func:`_run_training`.
"""

from pathlib import Path
from typing import Annotated, Any

import typer
from loguru import logger

app = typer.Typer(help="Model training operations.")


def _launch(model_name: str, cfg, train, val, val_cfg) -> None:
    """Top-level training entry point for multi-GPU elastic launch.

    Must be defined at module level so it can be pickled by
    ``torch.distributed``'s spawn-based multiprocessing.
    """
    from lpcv.models import get_model_spec

    spec = get_model_spec(model_name)
    trainer = spec.trainer_cls(
        config=cfg,
        train_dataset=train,
        eval_dataset=val,
        val_transform_config=val_cfg,
    )
    trainer.train()


def _run_training(
    model_name: str,
    data_dir: Path,
    config_overrides: dict[str, Any],
    decoder: str,
    num_frames: int,
    num_gpus: int,
    train_preset_override: list[dict[str, Any]] | None = None,
    val_preset_override: list[dict[str, Any]] | None = None,
    data_percent: float = 100.0,
) -> None:
    """Shared training flow for any registered model.

    Loads the model spec from the registry, builds datasets with the
    model's default presets (or overrides), constructs the trainer config,
    and runs training.  Saves ``val_transform.json`` alongside the model for
    adapter auto-construction at export time.
    """
    from lpcv.datasets.base import load_video_dataset
    from lpcv.datasets.decoder import get_decoder
    from lpcv.models import get_model_spec
    from lpcv.transforms import build_transform

    spec = get_model_spec(model_name)

    train_preset = train_preset_override or spec.train_preset
    val_preset = val_preset_override or spec.val_preset

    pin_memory = True
    decoder_kwargs: dict[str, str | int | None] = {}
    if decoder == "torchcodec-nvdec":
        pin_memory = False
        if num_gpus > 1:
            decoder_kwargs["num_gpus"] = num_gpus
            logger.info(f"NVDEC multi-GPU enabled: distributing decoding across {num_gpus} GPUs")

    video_decoder = get_decoder(decoder, **decoder_kwargs)
    train_transform = build_transform(train_preset)
    val_transform = build_transform(val_preset)
    train_ds, eval_ds = load_video_dataset(
        data_dir=data_dir,
        decoder=video_decoder,
        train_transform=train_transform,
        val_transform=val_transform,
        num_frames=num_frames,
        data_percent=data_percent,
    )

    config_overrides["dataloader_pin_memory"] = pin_memory
    config = spec.config_cls(**config_overrides)

    if num_gpus > 1:
        from torch.distributed.launcher.api import LaunchConfig, elastic_launch

        launch_config = LaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=num_gpus,
            rdzv_backend="c10d",
            rdzv_endpoint="localhost:0",
        )
        elastic_launch(launch_config, _launch)(model_name, config, train_ds, eval_ds, val_preset)
    else:
        _launch(model_name, config, train_ds, eval_ds, val_preset)


@app.command()
def videomae(
    data_dir: Annotated[Path, typer.Argument(help="Path to QEVD dataset or cached DatasetDict.")],
    output_dir: Annotated[
        str,
        typer.Option("--output-dir", "-o", help="Directory to save model and checkpoints."),
    ] = "model",
    model_name: Annotated[
        str,
        typer.Option("--model-name", "-m", help="Pretrained VideoMAE model name or path."),
    ] = "MCG-NJU/videomae-base",
    epochs: Annotated[
        int,
        typer.Option("--epochs", "-e", help="Number of training epochs."),
    ] = 15,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", "-b", help="Per-device training batch size."),
    ] = 8,
    learning_rate: Annotated[
        float,
        typer.Option("--learning-rate", "--lr", help="Learning rate."),
    ] = 5e-5,
    num_frames: Annotated[
        int,
        typer.Option("--num-frames", help="Number of frames to sample per video."),
    ] = 16,
    num_workers: Annotated[
        int,
        typer.Option("--num-workers", "-w", help="Number of dataloader workers."),
    ] = 4,
    fp16: Annotated[
        bool,
        typer.Option("--fp16", help="Use FP16 mixed precision."),
    ] = False,
    bf16: Annotated[
        bool,
        typer.Option("--bf16", help="Use BF16 mixed precision."),
    ] = False,
    gradient_accumulation_steps: Annotated[
        int,
        typer.Option(
            "--grad-accum", help="Gradient accumulation steps for larger effective batch size."
        ),
    ] = 1,
    gradient_checkpointing: Annotated[
        bool,
        typer.Option("--grad-checkpoint", help="Enable gradient checkpointing to save memory."),
    ] = False,
    freeze_strategy: Annotated[
        str,
        typer.Option("--freeze", help="Freeze strategy: 'none', 'backbone', or 'partial'."),
    ] = "none",
    lr_scheduler_type: Annotated[
        str,
        typer.Option(
            "--lr-scheduler",
            help="LR scheduler: 'linear', 'cosine', 'cosine_with_restarts', 'polynomial',"
            " 'constant', 'constant_with_warmup', or 'inverse_sqrt'.",
        ),
    ] = "linear",
    torch_compile: Annotated[
        bool,
        typer.Option("--compile", help="Use torch.compile for faster training."),
    ] = False,
    tf32: Annotated[
        bool,
        typer.Option("--tf32", help="Enable TF32 for faster matmuls on Ampere+ GPUs."),
    ] = False,
    max_steps: Annotated[
        int,
        typer.Option(
            "--max-steps", help="Stop after N optimizer steps. Overrides --epochs when > 0."
        ),
    ] = -1,
    resume: Annotated[
        str | None,
        typer.Option("--resume", help="Path to checkpoint to resume training from."),
    ] = None,
    decoder: Annotated[
        str,
        typer.Option(
            "--decoder",
            help="Video decoder backend: 'pyav', 'torchcodec-cpu', or 'torchcodec-nvdec'.",
        ),
    ] = "torchcodec-nvdec",
    num_gpus: Annotated[
        int,
        typer.Option("--num-gpus", "-g", help="Number of GPUs for distributed training."),
    ] = 1,
    data_percent: Annotated[
        float,
        typer.Option(
            "--data-percent",
            help="Percentage of data to use (0–100]. Stratified sampling preserves class ratio.",
        ),
    ] = 100.0,
) -> None:
    """Train a VideoMAE model on the QEVD dataset."""
    _run_training(
        model_name="videomae",
        data_dir=data_dir,
        config_overrides={
            "model_name": model_name,
            "num_frames": num_frames,
            "output_dir": output_dir,
            "num_train_epochs": epochs,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "learning_rate": learning_rate,
            "dataloader_num_workers": num_workers,
            "fp16": fp16,
            "bf16": bf16,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "gradient_checkpointing": gradient_checkpointing,
            "freeze_strategy": freeze_strategy,
            "lr_scheduler_type": lr_scheduler_type,
            "torch_compile": torch_compile,
            "tf32": tf32,
            "max_steps": max_steps,
            "resume_from_checkpoint": resume,
        },
        decoder=decoder,
        num_frames=num_frames,
        num_gpus=num_gpus,
        data_percent=data_percent,
    )


@app.command()
def r2plus1d(
    data_dir: Annotated[Path, typer.Argument(help="Path to QEVD dataset or cached DatasetDict.")],
    output_dir: Annotated[
        str,
        typer.Option("--output-dir", "-o", help="Directory to save model and checkpoints."),
    ] = "model",
    epochs: Annotated[
        int,
        typer.Option("--epochs", "-e", help="Number of training epochs."),
    ] = 2,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", "-b", help="Per-device training batch size."),
    ] = 24,
    learning_rate: Annotated[
        float,
        typer.Option("--learning-rate", "--lr", help="Learning rate."),
    ] = 1e-2,
    num_frames: Annotated[
        int,
        typer.Option("--num-frames", help="Number of frames to sample per video."),
    ] = 16,
    num_workers: Annotated[
        int,
        typer.Option("--num-workers", "-w", help="Number of dataloader workers."),
    ] = 4,
    fp16: Annotated[
        bool,
        typer.Option("--fp16", help="Use FP16 mixed precision."),
    ] = False,
    bf16: Annotated[
        bool,
        typer.Option("--bf16", help="Use BF16 mixed precision."),
    ] = False,
    gradient_accumulation_steps: Annotated[
        int,
        typer.Option(
            "--grad-accum", help="Gradient accumulation steps for larger effective batch size."
        ),
    ] = 1,
    freeze_strategy: Annotated[
        str,
        typer.Option("--freeze", help="Freeze strategy: 'none', 'backbone', or 'partial'."),
    ] = "partial",
    label_smoothing: Annotated[
        float,
        typer.Option("--label-smoothing", help="Label smoothing factor for cross-entropy."),
    ] = 0.1,
    lr_scheduler_type: Annotated[
        str,
        typer.Option(
            "--lr-scheduler",
            help="LR scheduler: 'linear', 'cosine', 'cosine_with_restarts', 'polynomial',"
            " 'constant', 'constant_with_warmup', or 'inverse_sqrt'.",
        ),
    ] = "cosine",
    torch_compile: Annotated[
        bool,
        typer.Option("--compile", help="Use torch.compile for faster training."),
    ] = False,
    tf32: Annotated[
        bool,
        typer.Option("--tf32", help="Enable TF32 for faster matmuls on Ampere+ GPUs."),
    ] = False,
    max_steps: Annotated[
        int,
        typer.Option(
            "--max-steps", help="Stop after N optimizer steps. Overrides --epochs when > 0."
        ),
    ] = -1,
    resume: Annotated[
        str | None,
        typer.Option("--resume", help="Path to checkpoint to resume training from."),
    ] = None,
    decoder: Annotated[
        str,
        typer.Option(
            "--decoder",
            help="Video decoder backend: 'pyav', 'torchcodec-cpu', or 'torchcodec-nvdec'.",
        ),
    ] = "torchcodec-nvdec",
    num_gpus: Annotated[
        int,
        typer.Option("--num-gpus", "-g", help="Number of GPUs for distributed training."),
    ] = 1,
    data_percent: Annotated[
        float,
        typer.Option(
            "--data-percent",
            help="Percentage of data to use (0–100]. Stratified sampling preserves class ratio.",
        ),
    ] = 100.0,
) -> None:
    """Train an R(2+1)D-18 model on the QEVD dataset."""
    _run_training(
        model_name="r2plus1d",
        data_dir=data_dir,
        config_overrides={
            "num_frames": num_frames,
            "output_dir": output_dir,
            "num_train_epochs": epochs,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "learning_rate": learning_rate,
            "label_smoothing": label_smoothing,
            "dataloader_num_workers": num_workers,
            "fp16": fp16,
            "bf16": bf16,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "freeze_strategy": freeze_strategy,
            "lr_scheduler_type": lr_scheduler_type,
            "torch_compile": torch_compile,
            "tf32": tf32,
            "max_steps": max_steps,
            "resume_from_checkpoint": resume,
        },
        decoder=decoder,
        num_frames=num_frames,
        num_gpus=num_gpus,
        data_percent=data_percent,
    )


@app.command()
def x3d(
    data_dir: Annotated[Path, typer.Argument(help="Path to QEVD dataset or cached DatasetDict.")],
    output_dir: Annotated[
        str,
        typer.Option("--output-dir", "-o", help="Directory to save model and checkpoints."),
    ] = "model",
    preset: Annotated[
        str,
        typer.Option(
            "--preset",
            "-p",
            help="X3D variant: 'x3d_xs', 'x3d_s', 'x3d_m', or 'x3d_l'.",
        ),
    ] = "x3d_m",
    epochs: Annotated[
        int,
        typer.Option("--epochs", "-e", help="Number of training epochs."),
    ] = 10,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", "-b", help="Per-device training batch size."),
    ] = 8,
    learning_rate: Annotated[
        float,
        typer.Option("--learning-rate", "--lr", help="Learning rate."),
    ] = 5e-3,
    num_frames: Annotated[
        int,
        typer.Option(
            "--num-frames",
            help="Number of frames to sample per video. 0 = use preset default.",
        ),
    ] = 0,
    crop_size: Annotated[
        int,
        typer.Option(
            "--crop-size",
            help="Spatial crop size. 0 = use preset default.",
        ),
    ] = 0,
    num_workers: Annotated[
        int,
        typer.Option("--num-workers", "-w", help="Number of dataloader workers."),
    ] = 4,
    fp16: Annotated[
        bool,
        typer.Option("--fp16", help="Use FP16 mixed precision."),
    ] = False,
    bf16: Annotated[
        bool,
        typer.Option("--bf16", help="Use BF16 mixed precision."),
    ] = False,
    gradient_accumulation_steps: Annotated[
        int,
        typer.Option(
            "--grad-accum", help="Gradient accumulation steps for larger effective batch size."
        ),
    ] = 1,
    freeze_strategy: Annotated[
        str,
        typer.Option("--freeze", help="Freeze strategy: 'none', 'backbone', or 'partial'."),
    ] = "partial",
    label_smoothing: Annotated[
        float,
        typer.Option("--label-smoothing", help="Label smoothing factor for cross-entropy."),
    ] = 0.1,
    lr_scheduler_type: Annotated[
        str,
        typer.Option(
            "--lr-scheduler",
            help="LR scheduler: 'linear', 'cosine', 'cosine_with_restarts', 'polynomial',"
            " 'constant', 'constant_with_warmup', or 'inverse_sqrt'.",
        ),
    ] = "cosine",
    torch_compile: Annotated[
        bool,
        typer.Option("--compile", help="Use torch.compile for faster training."),
    ] = False,
    tf32: Annotated[
        bool,
        typer.Option("--tf32", help="Enable TF32 for faster matmuls on Ampere+ GPUs."),
    ] = False,
    max_steps: Annotated[
        int,
        typer.Option(
            "--max-steps", help="Stop after N optimizer steps. Overrides --epochs when > 0."
        ),
    ] = -1,
    resume: Annotated[
        str | None,
        typer.Option("--resume", help="Path to checkpoint to resume training from."),
    ] = None,
    decoder: Annotated[
        str,
        typer.Option(
            "--decoder",
            help="Video decoder backend: 'pyav', 'torchcodec-cpu', or 'torchcodec-nvdec'.",
        ),
    ] = "torchcodec-nvdec",
    num_gpus: Annotated[
        int,
        typer.Option("--num-gpus", "-g", help="Number of GPUs for distributed training."),
    ] = 1,
    data_percent: Annotated[
        float,
        typer.Option(
            "--data-percent",
            help="Percentage of data to use (0–100]. Stratified sampling preserves class ratio.",
        ),
    ] = 100.0,
) -> None:
    """Train an X3D model on the QEVD dataset."""
    from lpcv.datasets.info import X3D_MEAN, X3D_STD
    from lpcv.models.x3d import X3D_PRESET_DEFAULTS
    from lpcv.transforms import make_presets

    if preset not in X3D_PRESET_DEFAULTS:
        available = ", ".join(sorted(X3D_PRESET_DEFAULTS))
        raise typer.BadParameter(f"Unknown preset {preset!r}. Available: {available}")

    defaults = X3D_PRESET_DEFAULTS[preset]
    resolved_frames = num_frames if num_frames > 0 else defaults["num_frames"]
    resolved_crop = crop_size if crop_size > 0 else defaults["crop_size"]

    train_preset, val_preset = make_presets(
        mean=X3D_MEAN,
        std=X3D_STD,
        crop_size=resolved_crop,
    )

    _run_training(
        model_name="x3d",
        data_dir=data_dir,
        config_overrides={
            "preset": preset,
            "num_frames": resolved_frames,
            "crop_size": resolved_crop,
            "output_dir": output_dir,
            "num_train_epochs": epochs,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "learning_rate": learning_rate,
            "label_smoothing": label_smoothing,
            "dataloader_num_workers": num_workers,
            "fp16": fp16,
            "bf16": bf16,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "freeze_strategy": freeze_strategy,
            "lr_scheduler_type": lr_scheduler_type,
            "torch_compile": torch_compile,
            "tf32": tf32,
            "max_steps": max_steps,
            "resume_from_checkpoint": resume,
        },
        decoder=decoder,
        num_frames=resolved_frames,
        num_gpus=num_gpus,
        train_preset_override=train_preset,
        val_preset_override=val_preset,
        data_percent=data_percent,
    )


@app.command()
def tsm(
    data_dir: Annotated[Path, typer.Argument(help="Path to QEVD dataset or cached DatasetDict.")],
    output_dir: Annotated[
        str,
        typer.Option("--output-dir", "-o", help="Directory to save model and checkpoints."),
    ] = "model",
    backbone: Annotated[
        str,
        typer.Option(
            "--backbone",
            help="TSM backbone: 'resnet18' or 'resnet50'.",
        ),
    ] = "resnet50",
    epochs: Annotated[
        int,
        typer.Option("--epochs", "-e", help="Number of training epochs."),
    ] = 10,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", "-b", help="Per-device training batch size."),
    ] = 16,
    learning_rate: Annotated[
        float,
        typer.Option("--learning-rate", "--lr", help="Learning rate."),
    ] = 1e-2,
    num_frames: Annotated[
        int,
        typer.Option("--num-frames", help="Number of temporal segments per video."),
    ] = 8,
    shift_div: Annotated[
        int,
        typer.Option("--shift-div", help="Fraction of channels to shift (default 8)."),
    ] = 8,
    num_workers: Annotated[
        int,
        typer.Option("--num-workers", "-w", help="Number of dataloader workers."),
    ] = 4,
    fp16: Annotated[
        bool,
        typer.Option("--fp16", help="Use FP16 mixed precision."),
    ] = False,
    bf16: Annotated[
        bool,
        typer.Option("--bf16", help="Use BF16 mixed precision."),
    ] = False,
    gradient_accumulation_steps: Annotated[
        int,
        typer.Option(
            "--grad-accum", help="Gradient accumulation steps for larger effective batch size."
        ),
    ] = 1,
    freeze_strategy: Annotated[
        str,
        typer.Option("--freeze", help="Freeze strategy: 'none', 'backbone', or 'partial'."),
    ] = "partial",
    label_smoothing: Annotated[
        float,
        typer.Option("--label-smoothing", help="Label smoothing factor for cross-entropy."),
    ] = 0.1,
    lr_scheduler_type: Annotated[
        str,
        typer.Option(
            "--lr-scheduler",
            help="LR scheduler: 'linear', 'cosine', 'cosine_with_restarts', 'polynomial',"
            " 'constant', 'constant_with_warmup', or 'inverse_sqrt'.",
        ),
    ] = "cosine",
    torch_compile: Annotated[
        bool,
        typer.Option("--compile", help="Use torch.compile for faster training."),
    ] = False,
    tf32: Annotated[
        bool,
        typer.Option("--tf32", help="Enable TF32 for faster matmuls on Ampere+ GPUs."),
    ] = False,
    max_steps: Annotated[
        int,
        typer.Option(
            "--max-steps", help="Stop after N optimizer steps. Overrides --epochs when > 0."
        ),
    ] = -1,
    resume: Annotated[
        str | None,
        typer.Option("--resume", help="Path to checkpoint to resume training from."),
    ] = None,
    decoder: Annotated[
        str,
        typer.Option(
            "--decoder",
            help="Video decoder backend: 'pyav', 'torchcodec-cpu', or 'torchcodec-nvdec'.",
        ),
    ] = "torchcodec-nvdec",
    num_gpus: Annotated[
        int,
        typer.Option("--num-gpus", "-g", help="Number of GPUs for distributed training."),
    ] = 1,
    data_percent: Annotated[
        float,
        typer.Option(
            "--data-percent",
            help="Percentage of data to use (0–100]. Stratified sampling preserves class ratio.",
        ),
    ] = 100.0,
) -> None:
    """Train a TSM (Temporal Shift Module) model on the QEVD dataset."""
    from lpcv.models.tsm import TSM_BACKBONES

    if backbone not in TSM_BACKBONES:
        available = ", ".join(sorted(TSM_BACKBONES))
        raise typer.BadParameter(f"Unknown backbone {backbone!r}. Available: {available}")

    _run_training(
        model_name="tsm",
        data_dir=data_dir,
        config_overrides={
            "backbone": backbone,
            "num_frames": num_frames,
            "shift_div": shift_div,
            "output_dir": output_dir,
            "num_train_epochs": epochs,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "learning_rate": learning_rate,
            "label_smoothing": label_smoothing,
            "dataloader_num_workers": num_workers,
            "fp16": fp16,
            "bf16": bf16,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "freeze_strategy": freeze_strategy,
            "lr_scheduler_type": lr_scheduler_type,
            "torch_compile": torch_compile,
            "tf32": tf32,
            "max_steps": max_steps,
            "resume_from_checkpoint": resume,
        },
        decoder=decoder,
        num_frames=num_frames,
        num_gpus=num_gpus,
        data_percent=data_percent,
    )
