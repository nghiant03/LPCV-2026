"""CLI sub-commands for model training."""

from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

app = typer.Typer(help="Model training operations.")


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
) -> None:
    """Train a VideoMAE model on the QEVD dataset."""
    from lpcv.datasets.base import load_video_dataset
    from lpcv.datasets.decoder import get_decoder
    from lpcv.models.videomae import VideoMAETrainerConfig
    from lpcv.transforms import TRAIN_PRESET, VAL_PRESET, build_transform

    pin_memory = True
    decoder_kwargs: dict[str, str | int | None] = {}
    if decoder == "torchcodec-nvdec":
        pin_memory = False
        if num_gpus > 1:
            decoder_kwargs["num_gpus"] = num_gpus
            logger.info(f"NVDEC multi-GPU enabled: distributing decoding across {num_gpus} GPUs")

    video_decoder = get_decoder(decoder, **decoder_kwargs)
    train_transform = build_transform(TRAIN_PRESET)
    val_transform = build_transform(VAL_PRESET)
    train_ds, eval_ds = load_video_dataset(
        data_dir=data_dir,
        decoder=video_decoder,
        train_transform=train_transform,
        val_transform=val_transform,
        num_frames=num_frames,
    )

    config = VideoMAETrainerConfig(
        model_name=model_name,
        num_frames=num_frames,
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=pin_memory,
        fp16=fp16,
        bf16=bf16,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        freeze_strategy=freeze_strategy,
        torch_compile=torch_compile,
        tf32=tf32,
        max_steps=max_steps,
        resume_from_checkpoint=resume,
    )

    if num_gpus > 1:
        from torch.distributed.launcher.api import LaunchConfig, elastic_launch

        launch_config = LaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=num_gpus,
            rdzv_backend="c10d",
            rdzv_endpoint="localhost:0",
        )
        elastic_launch(launch_config, run_trainer)(config, train_ds, eval_ds)
    else:
        run_trainer(config, train_ds, eval_ds)


def run_trainer(config, train_ds, eval_ds) -> None:
    """Instantiate and run the VideoMAE trainer."""
    from lpcv.models.videomae import VideoMAEModelTrainer

    trainer = VideoMAEModelTrainer(config=config, train_dataset=train_ds, eval_dataset=eval_ds)
    trainer.train()
