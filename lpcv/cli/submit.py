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
    spatial_size: Annotated[
        int, typer.Option("--spatial-size", help="Target spatial resolution (H=W).")
    ] = 224,
    decoder: Annotated[
        str,
        typer.Option(
            "--decoder", help="Video decoder backend: 'pyav', 'torchcodec-cpu', 'torchcodec-nvdec'."
        ),
    ] = "pyav",
) -> None:
    """Decode validation videos to .npy tensors and write a manifest.jsonl."""
    from lpcv.datasets.decoder import get_decoder
    from lpcv.submission import preprocess_dataset

    video_decoder = get_decoder(decoder)
    manifest = preprocess_dataset(
        data_dir=data_dir,
        output_dir=output_dir,
        decoder=video_decoder,
        num_frames=num_frames,
        spatial_size=spatial_size,
    )
    logger.info(f"Done — manifest: {manifest}")


@app.command()
def export(
    model_path: Annotated[
        Path, typer.Argument(help="Path to a HuggingFace-saved VideoMAE checkpoint directory.")
    ],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output .onnx file path.")] = Path(
        "model.onnx"
    ),
    num_frames: Annotated[
        int, typer.Option("--num-frames", help="Temporal dimension of input.")
    ] = 16,
    spatial_size: Annotated[
        int, typer.Option("--spatial-size", help="Spatial H=W of input.")
    ] = 224,
    opset: Annotated[int, typer.Option("--opset", help="ONNX opset version.")] = 17,
) -> None:
    """Export a trained VideoMAE checkpoint to ONNX."""
    from lpcv.submission import export_onnx

    export_onnx(
        model_path=model_path,
        output_path=output,
        num_frames=num_frames,
        spatial_size=spatial_size,
        opset_version=opset,
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
    spatial_size: Annotated[
        int, typer.Option("--spatial-size", help="Spatial H=W for input spec.")
    ] = 224,
    output_dir: Annotated[
        Path, typer.Option("--output-dir", "-o", help="Directory for the compiled .bin.")
    ] = Path("export_assets"),
) -> None:
    """Compile an ONNX model on Qualcomm AI Hub and download the binary."""
    from lpcv.submission import compile_on_hub

    compile_on_hub(
        model_path=onnx_path,
        device_name=device_name,
        num_frames=num_frames,
        spatial_size=spatial_size,
        output_dir=output_dir,
    )


@app.command()
def infer(
    compiled_model: Annotated[Path, typer.Argument(help="Path to the compiled .bin model.")],
    tensor_dir: Annotated[Path, typer.Argument(help="Directory of preprocessed .npy tensors.")],
    manifest: Annotated[Path, typer.Argument(help="Path to manifest.jsonl from preprocessing.")],
    output_h5: Annotated[
        Path, typer.Option("--output", "-o", help="Output HDF5 logits file.")
    ] = Path("dataset-export.h5"),
    device_name: Annotated[
        str, typer.Option("--device", "-d", help="Qualcomm AI Hub device name.")
    ] = "Dragonwing IQ-9075 EVK",
    channel_last: Annotated[
        bool, typer.Option("--channel-last", help="Transpose tensors to NTHWC layout.")
    ] = False,
) -> None:
    """Upload tensors and run on-device inference via Qualcomm AI Hub."""
    from lpcv.submission import run_inference_on_hub

    run_inference_on_hub(
        compiled_model_path=compiled_model,
        tensor_dir=tensor_dir,
        manifest_path=manifest,
        output_h5=output_h5,
        device_name=device_name,
        channel_last=channel_last,
    )
