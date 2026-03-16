"""CLI sub-commands for model evaluation."""

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help="Model evaluation operations.")


@app.command("model")
def model(
    model_path: Annotated[Path, typer.Argument(help="Path to the trained model directory.")],
    data_dir: Annotated[Path, typer.Argument(help="Root directory of the QEVD dataset.")],
    num_frames: Annotated[
        int,
        typer.Option("--num-frames", help="Number of frames to sample per video."),
    ] = 16,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", "-b", help="Batch size for inference."),
    ] = 8,
    num_workers: Annotated[
        int,
        typer.Option("--num-workers", "-w", help="Number of dataloader workers."),
    ] = 4,
) -> None:
    """Evaluate a trained VideoMAE model on the QEVD validation set."""
    from lpcv.evaluation import evaluate_model

    results = evaluate_model(
        model_path=model_path,
        data_dir=data_dir,
        num_frames=num_frames,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    typer.echo(f"Top-1 Accuracy: {results['top1_accuracy']:.2f}%")
    typer.echo(f"Top-5 Accuracy: {results['top5_accuracy']:.2f}%")


@app.command("h5")
def h5(
    h5_path: Annotated[Path, typer.Argument(help="Path to HDF5 logits file from inference.")],
    manifest: Annotated[Path, typer.Argument(help="Path to manifest.jsonl from preprocessing.")],
    class_map: Annotated[
        Path,
        typer.Option("--class-map", "-c", help="Path to class_map.json."),
    ] = Path("class_map.json"),
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Print per-sample predictions."),
    ] = False,
) -> None:
    """Evaluate inference results from an HDF5 logits file against ground-truth labels."""
    from lpcv.evaluation import evaluate_h5

    results = evaluate_h5(
        h5_path=h5_path,
        manifest_path=manifest,
        class_map_path=class_map,
        verbose=verbose,
    )
    typer.echo(f"Top-1 Accuracy: {results['top1_accuracy']:.2f}%")
    typer.echo(f"Top-5 Accuracy: {results['top5_accuracy']:.2f}%")
