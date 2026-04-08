"""CLI sub-command for model training.

A single ``train`` command that accepts ``--model`` and/or ``--config``
to select the architecture.  ``--model`` overrides the ``model`` key in
the YAML when both are provided.
"""

from pathlib import Path
from typing import Annotated, Any

import typer
from loguru import logger


def _launch(
    spec: Any,
    cfg: Any,
    train: Any,
    val: Any,
    val_cfg: list[dict[str, Any]] | None,
    model_cfg: dict[str, Any] | None,
) -> None:
    """Top-level training entry point for multi-GPU elastic launch.

    Must be defined at module level so it can be pickled by
    ``torch.distributed``'s spawn-based multiprocessing.
    """
    trainer = spec.trainer_cls(
        config=cfg,
        train_dataset=train,
        eval_dataset=val,
        val_transform_config=val_cfg,
        model_config=model_cfg,
    )
    trainer.input_layout = spec.input_layout
    trainer.input_key = spec.input_key
    trainer.train()


def _run_training(
    data_dir: Path,
    spec: Any,
    resolved_model: Any,
    config_overrides: dict[str, Any],
    decoder: str,
    num_gpus: int,
    data_percent: float = 100.0,
) -> None:
    """Shared training flow for any registered model.

    Loads the model spec from the registry, builds datasets with the
    model's default presets (or overrides), constructs the trainer config,
    and runs training.  Saves ``val_transform.json`` and ``model_config.yaml``
    alongside the model for adapter auto-construction at export time.
    """
    from lpcv.datasets.base import load_video_dataset
    from lpcv.datasets.decoder import get_decoder
    from lpcv.transforms import build_transform

    pin_memory = True
    decoder_kwargs: dict[str, str | int | None] = {}
    if decoder == "torchcodec-nvdec":
        pin_memory = False
        if num_gpus > 1:
            decoder_kwargs["num_gpus"] = num_gpus
            logger.info(f"NVDEC multi-GPU enabled: distributing decoding across {num_gpus} GPUs")

    video_decoder = get_decoder(decoder, **decoder_kwargs)
    train_transform = build_transform(resolved_model.train_preset)
    val_transform = build_transform(resolved_model.val_preset)
    train_ds, eval_ds = load_video_dataset(
        data_dir=data_dir,
        decoder=video_decoder,
        train_transform=train_transform,
        val_transform=val_transform,
        num_frames=resolved_model.num_frames,
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
        elastic_launch(launch_config, _launch)(
            spec,
            config,
            train_ds,
            eval_ds,
            resolved_model.val_preset,
            resolved_model.model_config,
        )
    else:
        _launch(
            spec,
            config,
            train_ds,
            eval_ds,
            resolved_model.val_preset,
            resolved_model.model_config,
        )


def train(
    data_dir: Annotated[
        Path,
        typer.Argument(help="Path to QEVD dataset or cached DatasetDict."),
    ],
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Model name."),
    ] = "",
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to a model config YAML file."),
    ] = None,
    output_dir: Annotated[
        str,
        typer.Option("--output-dir", "-o", help="Directory to save model and checkpoints."),
    ] = "model",
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
    ] = 0.0,
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
        typer.Option(
            "--grad-checkpoint",
            help="Override model gradient checkpointing when supported by the selected trainer.",
        ),
    ] = False,
    freeze_strategy: Annotated[
        str,
        typer.Option("--freeze", help="Freeze strategy: 'none', 'backbone', or 'partial'."),
    ] = "",
    lr_scheduler_type: Annotated[
        str,
        typer.Option(
            "--lr-scheduler",
            help="LR scheduler: 'linear', 'cosine', 'cosine_with_restarts', 'polynomial',"
            " 'constant', 'constant_with_warmup', or 'inverse_sqrt'.",
        ),
    ] = "",
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
            help="Percentage of data to use (0-100]. Stratified sampling preserves class ratio.",
        ),
    ] = 100.0,
) -> None:
    """Train a model.

    Provide ``--model`` to select the architecture directly, ``--config``
    to load from a YAML file, or both (``--model`` wins).
    """
    from lpcv.models import get_model_spec, list_models, load_model_config, resolve_model_config

    if decoder == "torchcodec-nvdec" and num_workers != num_gpus:
        logger.warning(
            "torchcodec-nvdec requires --num-workers to match --num-gpus so each "
            f"decode worker maps cleanly onto one GPU. Settings num_workers to {num_gpus}"
        )
        num_workers = num_gpus

    model_cfg: dict[str, Any] | None = None

    if config is not None:
        model_cfg = load_model_config(config)

    model_name = model or (model_cfg["model"] if model_cfg else "")

    if not model_name:
        available = ", ".join(list_models())
        raise typer.BadParameter(
            f"Specify --model or --config (with a 'model' key). Available models: {available}"
        )

    if model_name and model_cfg and model_name != model_cfg.get("model"):
        logger.info(f"--model={model_name!r} overrides config model={model_cfg['model']!r}")
        model_cfg["model"] = model_name

    if model_cfg is None:
        model_cfg = {"model": model_name}

    resolved_model = resolve_model_config(model_name, model_cfg)
    spec = get_model_spec(model_name)

    config_overrides: dict[str, Any] = {
        **{k: v for k, v in resolved_model.model_config.items() if k != "model"},
        "output_dir": output_dir,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "dataloader_num_workers": num_workers,
        "fp16": fp16,
        "bf16": bf16,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "torch_compile": torch_compile,
        "tf32": tf32,
        "max_steps": max_steps,
        "resume_from_checkpoint": resume,
    }
    if gradient_checkpointing is not None:
        config_overrides["gradient_checkpointing"] = gradient_checkpointing
    if learning_rate > 0.0:
        config_overrides["learning_rate"] = learning_rate
    if freeze_strategy:
        config_overrides["freeze_strategy"] = freeze_strategy
    if lr_scheduler_type:
        config_overrides["lr_scheduler_type"] = lr_scheduler_type

    _run_training(
        data_dir=data_dir,
        spec=spec,
        resolved_model=resolved_model,
        config_overrides=config_overrides,
        decoder=decoder,
        num_gpus=num_gpus,
        data_percent=data_percent,
    )
