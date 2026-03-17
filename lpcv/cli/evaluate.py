"""CLI sub-commands for model evaluation."""

from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

app = typer.Typer(help="Model evaluation operations.")


@app.command("model")
def model(
    data_dir: Annotated[Path, typer.Argument(help="Path to QEVD dataset or cached DatasetDict.")],
    model_path: Annotated[
        str,
        typer.Option("--model-path", "-m", help="Path to the trained model directory."),
    ] = "model",
    num_frames: Annotated[
        int,
        typer.Option("--num-frames", help="Number of frames to sample per video."),
    ] = 16,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", "-b", help="Batch size for inference."),
    ] = 8,
    clips_per_video: Annotated[
        int,
        typer.Option(
            "--clips-per-video",
            help="Number of clips to sample per video for multi-clip aggregation.",
        ),
    ] = 1,
    decoder: Annotated[
        str,
        typer.Option(
            "--decoder",
            help="Video decoder backend: 'pyav', 'torchcodec-cpu', or 'torchcodec-nvdec'.",
        ),
    ] = "torchcodec-nvdec",
) -> None:
    """Evaluate a trained VideoMAE model on the QEVD validation set."""
    from lpcv.datasets.base import load_video_dataset
    from lpcv.datasets.decoder import get_decoder
    from lpcv.evaluation import evaluate_model
    from lpcv.transforms import VAL_PRESET, build_transform

    video_decoder = get_decoder(decoder)
    val_transform = build_transform(VAL_PRESET)
    _, eval_ds = load_video_dataset(
        data_dir=data_dir,
        decoder=video_decoder,
        val_transform=val_transform,
        num_frames=num_frames,
    )

    results = evaluate_model(
        model_path=Path(model_path),
        eval_ds=eval_ds,
        batch_size=batch_size,
        clips_per_video=clips_per_video,
    )
    logger.info(f"Clip  Acc@1: {results['clip_top1_accuracy']:.2f}%")
    logger.info(f"Clip  Acc@5: {results['clip_top5_accuracy']:.2f}%")
    logger.info(f"Video Acc@1: {results['video_top1_accuracy']:.2f}%")
    logger.info(f"Video Acc@5: {results['video_top5_accuracy']:.2f}%")


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
    logger.info(f"Top-1 Accuracy: {results['top1_accuracy']:.2f}%")
    logger.info(f"Top-5 Accuracy: {results['top5_accuracy']:.2f}%")
