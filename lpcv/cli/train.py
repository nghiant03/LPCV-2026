from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help="Model training operations.")

DEFAULT_NPROC = 1


@app.command()
def videomae(
    data_dir: Annotated[Path, typer.Argument(help="Path to QEVD dataset or cached/precomputed DatasetDict.")],
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
        typer.Option("--grad-accum", help="Gradient accumulation steps for larger effective batch size."),
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
    is_cache: Annotated[
        bool,
        typer.Option("--is-cache", help="Treat data_dir as a saved DatasetDict (cached or precomputed)."),
    ] = False,
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
    from datasets import DatasetDict, load_from_disk

    from lpcv.datasets.qevd import QEVDAdapter
    from lpcv.models.videomae import VideoMAEModelTrainer, VideoMAETrainerConfig

    if is_cache:
        dataset = load_from_disk(str(data_dir))
        if not isinstance(dataset, DatasetDict):
            raise typer.BadParameter(f"Expected DatasetDict at {data_dir}, got {type(dataset).__name__}")
    else:
        adapter = QEVDAdapter(data_dir=data_dir)
        dataset = adapter.load()

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
        resume_from_checkpoint=resume,
    )

    if num_gpus > 1:
        from torch.distributed.launcher.api import LaunchConfig, elastic_launch

        launch_config = LaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=num_gpus,
            rdzv_backend="static",
            rdzv_endpoint="localhost:0",
        )

        def _worker() -> None:
            trainer = VideoMAEModelTrainer(config=config, dataset=dataset)
            trainer.train()

        elastic_launch(launch_config, _worker)()
    else:
        trainer = VideoMAEModelTrainer(config=config, dataset=dataset)
        trainer.train()
