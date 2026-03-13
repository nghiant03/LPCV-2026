from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help="Model training operations.")

DEFAULT_NPROC = 1


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
    num_gpus: Annotated[
        int,
        typer.Option("--num-gpus", "-g", help="Number of GPUs for distributed training."),
    ] = DEFAULT_NPROC,
) -> None:
    """Train a VideoMAE model on the QEVD dataset."""
    from lpcv.datasets.qevd import QEVDAdapter
    from lpcv.models.videomae import VideoMAETrainerConfig
    from lpcv.transforms import TRAIN_PRESET, VAL_PRESET, build_transform

    fmt_step = [{"name": "FromVideo"}]
    train_transform = build_transform(fmt_step + TRAIN_PRESET)
    val_transform = build_transform(fmt_step + VAL_PRESET)

    adapter = QEVDAdapter(data_dir=data_dir)
    train_ds, eval_ds = adapter.load(train_transform=train_transform, val_transform=val_transform)

    config = VideoMAETrainerConfig(
        model_name=model_name,
        num_frames=num_frames,
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        dataloader_num_workers=num_workers,
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
    from lpcv.models.videomae import VideoMAEModelTrainer

    trainer = VideoMAEModelTrainer(config=config, train_dataset=train_ds, eval_dataset=eval_ds)
    trainer.train()
