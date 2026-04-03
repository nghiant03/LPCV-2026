"""CLI sub-commands for the submission pipeline.

Covers preprocessing, ONNX export, AI Hub compilation, and on-device inference.
"""

from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

app = typer.Typer(help="Submission pipeline: preprocess, export, compile, infer.")


@app.command()
def preprocess(
    data_dir: Annotated[
        Path, typer.Argument(help="Path to videofolder dataset root (must contain val/ split).")
    ],
    output_dir: Annotated[
        Path, typer.Argument(help="Directory to write .npy tensors and manifest.jsonl.")
    ],
    num_frames: Annotated[
        int, typer.Option("--num-frames", help="Number of frames to sample per clip.")
    ] = 16,
    decoder: Annotated[
        str,
        typer.Option("--decoder", help="Decoder backend (pyav, torchcodec-cpu, torchcodec-nvdec)."),
    ] = "torchcodec-nvdec",
    target_fps: Annotated[
        int, typer.Option("--target-fps", help="Target FPS for frame resampling.")
    ] = 4,
) -> None:
    """Decode validation videos to .npy tensors using the competition pipeline."""
    from lpcv.submission import preprocess_dataset

    manifest = preprocess_dataset(
        data_dir=data_dir,
        output_dir=output_dir,
        num_frames=num_frames,
        decoder_name=decoder,
        target_fps=target_fps,
    )
    logger.info(f"Done — manifest: {manifest}")


@app.command()
def export(
    model_path: Annotated[Path, typer.Argument(help="Path to a saved model checkpoint directory.")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output .onnx file path.")] = Path(
        "model.onnx"
    ),
    model_type: Annotated[
        str, typer.Option("--model-type", "-m", help="Registered model type (e.g. videomae).")
    ] = "videomae",
    num_frames: Annotated[
        int, typer.Option("--num-frames", help="Temporal dimension of input.")
    ] = 16,
    opset: Annotated[int, typer.Option("--opset", help="ONNX opset version.")] = 18,
    dynamo: Annotated[
        bool,
        typer.Option(
            "--dynamo", help="Use torch.export-based ONNX exporter (flattens control flow)."
        ),
    ] = False,
    decompose: Annotated[
        bool,
        typer.Option(
            "--decompose/--no-decompose",
            help="Replace depthwise 3D convs with 2D+1D decompositions.",
        ),
    ] = True,
) -> None:
    """Export a trained model checkpoint to ONNX with competition adapter."""
    from lpcv.submission import export_onnx

    export_onnx(
        model_path=model_path,
        output_path=output,
        model_type=model_type,
        num_frames=num_frames,
        opset_version=opset,
        dynamo=dynamo,
        decompose=decompose,
    )


@app.command()
def compile(
    onnx_path: Annotated[Path, typer.Argument(help="Path to the ONNX model file.")],
    device_name: Annotated[
        str, typer.Option("--device", "-d", help="Qualcomm AI Hub device name.")
    ] = "Dragonwing IQ-9075 EVK",
    num_frames: Annotated[
        int, typer.Option("--num-frames", help="Temporal dimension for input spec.")
    ] = 16,
    output_dir: Annotated[
        Path, typer.Option("--output-dir", "-o", help="Directory for the compiled .bin.")
    ] = Path("export_assets"),
    no_download: Annotated[
        bool,
        typer.Option("--no-download", help="Skip binary download; print the AI Hub model ID only."),
    ] = False,
    hub_model_id: Annotated[
        str | None,
        typer.Option(
            "--hub-model-id",
            help="Reuse an existing AI Hub model ID instead of uploading.",
        ),
    ] = None,
    name: Annotated[
        str | None,
        typer.Option("--name", "-n", help="Job name on AI Hub."),
    ] = None,
) -> None:
    """Compile an ONNX model on Qualcomm AI Hub and download the binary."""
    from lpcv.submission import compile_on_hub

    result = compile_on_hub(
        model_path=onnx_path,
        device_name=device_name,
        num_frames=num_frames,
        output_dir=output_dir,
        download=not no_download,
        hub_model_id=hub_model_id,
        name=name,
    )
    if no_download:
        logger.info(f"AI Hub model ID: {result}")


@app.command()
def profile(
    compiled_model: Annotated[
        Path,
        typer.Argument(help="Path to the compiled .bin model (ignored with --hub-model-id)."),
    ] = Path("."),
    device_name: Annotated[
        str, typer.Option("--device", "-d", help="Qualcomm AI Hub device name.")
    ] = "Dragonwing IQ-9075 EVK",
    hub_model_id: Annotated[
        str | None,
        typer.Option(
            "--hub-model-id",
            help="Reuse an existing AI Hub model ID instead of uploading.",
        ),
    ] = None,
    name: Annotated[
        str | None,
        typer.Option("--name", "-n", help="Job name on AI Hub."),
    ] = None,
) -> None:
    """Submit a profile job for a compiled model on Qualcomm AI Hub."""
    from lpcv.submission import profile_on_hub

    model_path = None if hub_model_id else compiled_model
    url = profile_on_hub(
        model_path=model_path,
        device_name=device_name,
        hub_model_id=hub_model_id,
        name=name,
    )
    logger.info(f"Profile results: {url}")


@app.command()
def validate(
    config: Annotated[
        Path,
        typer.Argument(help="Path to a model config YAML file."),
    ],
    num_classes: Annotated[
        int, typer.Option("--num-classes", "-c", help="Number of output classes.")
    ] = 15,
    device_name: Annotated[
        str, typer.Option("--device", "-d", help="Qualcomm AI Hub device name.")
    ] = "Dragonwing IQ-9075 EVK",
    opset: Annotated[int, typer.Option("--opset", help="ONNX opset version.")] = 18,
    dynamo: Annotated[
        bool,
        typer.Option(
            "--dynamo", help="Use torch.export-based ONNX exporter (flattens control flow)."
        ),
    ] = False,
    decompose: Annotated[
        bool,
        typer.Option(
            "--decompose/--no-decompose",
            help="Replace depthwise 3D convs with 2D+1D decompositions.",
        ),
    ] = True,
    name: Annotated[
        str | None,
        typer.Option("--name", "-n", help="Job name prefix on AI Hub."),
    ] = None,
) -> None:
    """Build a throwaway model, export, compile, and profile on AI Hub."""
    from lpcv.models import load_model_config
    from lpcv.submission import validate_on_hub

    model_cfg = load_model_config(config)
    model_type = model_cfg["model"]
    arch_params = {k: v for k, v in model_cfg.items() if k != "model"}

    url = validate_on_hub(
        model_type=model_type,
        num_classes=num_classes,
        num_frames=arch_params.get("num_frames", 16),
        device_name=device_name,
        opset_version=opset,
        dynamo=dynamo,
        decompose=decompose,
        name=name,
        model_kwargs=arch_params,
    )
    logger.info(f"Validation complete: {url}")


@app.command()
def infer(
    tensor_dir: Annotated[Path, typer.Argument(help="Directory of preprocessed .npy tensors.")],
    compiled_model: Annotated[
        Path,
        typer.Option(
            "--compiled-model",
            "-c",
            help="Path to the compiled .bin model (ignored with --hub-model-id).",
        ),
    ] = Path("."),
    output_h5: Annotated[
        Path, typer.Option("--output", "-o", help="Output HDF5 logits file.")
    ] = Path("result/dataset-export.h5"),
    device_name: Annotated[
        str, typer.Option("--device", "-d", help="Qualcomm AI Hub device name.")
    ] = "Dragonwing IQ-9075 EVK",
    channel_last: Annotated[
        bool, typer.Option("--channel-last", help="Transpose tensors to NTHWC layout.")
    ] = False,
    hub_model_id: Annotated[
        str | None,
        typer.Option(
            "--hub-model-id",
            help="Reuse an existing AI Hub model ID instead of uploading.",
        ),
    ] = None,
    name: Annotated[
        str | None,
        typer.Option("--name", "-n", help="Job name prefix on AI Hub."),
    ] = None,
) -> None:
    """Upload tensors and run on-device inference via Qualcomm AI Hub."""
    from lpcv.submission import run_inference_on_hub

    run_inference_on_hub(
        compiled_model_path=compiled_model,
        tensor_dir=tensor_dir,
        manifest_path=tensor_dir / "manifest.jsonl",
        output_h5=output_h5,
        device_name=device_name,
        channel_last=channel_last,
        hub_model_id=hub_model_id,
        name=name,
    )
