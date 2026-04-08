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
        typer.Argument(help="Path to the trained model directory."),
    ],
    num_frames: Annotated[
        int | None,
        typer.Option("--num-frames", help="Override frames per clip from the saved artifact."),
    ] = None,
    model_type: Annotated[
        str | None,
        typer.Option("--model-type", "-m", help="Override model type from the saved artifact."),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", "-b", help="Batch size for inference."),
    ] = 32,
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
    force_override: Annotated[
        bool,
        typer.Option(
            "--force-override",
            help="Allow --model-type/--num-frames to differ from saved artifact metadata.",
        ),
    ] = False,
) -> None:
    """Evaluate a trained model on the QEVD validation set."""
    from lpcv.datasets.base import load_video_dataset
    from lpcv.datasets.decoder import get_decoder
    from lpcv.evaluation import evaluate_model
    from lpcv.models import (
        MODEL_CONFIG_FILENAME,
        VAL_TRANSFORM_FILENAME,
        load_model_config,
        resolve_artifact_model_name,
        resolve_model_config,
    )
    from lpcv.transforms import build_transform, load_val_transform_config

    model_dir = Path(model_path)
    resolved_model_type = resolve_artifact_model_name(
        model_dir,
        model_name=model_type,
        force_override=force_override,
    )
    raw_model_config: dict[str, object] = {}
    if (model_dir / MODEL_CONFIG_FILENAME).is_file():
        raw_model_config = load_model_config(model_dir)
    resolved_model = resolve_model_config(resolved_model_type, raw_model_config)
    resolved_num_frames = resolved_model.num_frames
    if num_frames is not None:
        if num_frames != resolved_num_frames and not force_override:
            raise typer.BadParameter(
                f"--num-frames={num_frames} does not match saved artifact "
                f"num_frames={resolved_num_frames}. "
                "Pass --force-override to ignore the saved metadata."
            )
        resolved_num_frames = num_frames

    val_config_path = model_dir / VAL_TRANSFORM_FILENAME
    val_config = (
        load_val_transform_config(val_config_path)
        if val_config_path.is_file()
        else resolved_model.val_preset
    )
    video_decoder = get_decoder(decoder)
    val_transform = build_transform(val_config)
    _, eval_ds = load_video_dataset(
        data_dir=data_dir,
        decoder=video_decoder,
        val_transform=val_transform,
        num_frames=resolved_num_frames,
    )

    results = evaluate_model(
        model_type=resolved_model_type,
        model_path=model_dir,
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
