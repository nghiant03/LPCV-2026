"""CLI sub-commands for dataset operations (convert, etc.)."""

from pathlib import Path
from typing import Annotated

import typer


def convert(
    data_dir: Annotated[Path, typer.Argument(help="Root directory of the QEVD dataset.")],
    target_label: Annotated[
        Path | None,
        typer.Option("--target-label", "-t", help="Path to target label JSON file."),
    ] = None,
    source_label: Annotated[
        Path | None,
        typer.Option("--source-label", "-s", help="Path to source label JSON file."),
    ] = None,
    num_workers: Annotated[
        int | None,
        typer.Option("--num-workers", "-w", help="Number of parallel workers."),
    ] = None,
) -> None:
    """Convert QEVD dataset into videofolder format."""
    from lpcv.datasets.qevd import QEVDAdapter

    adapter = QEVDAdapter(
        data_dir=data_dir,
        target_label=target_label,
        source_label_path=source_label,
        num_workers=num_workers,
    )
    adapter.convert()
