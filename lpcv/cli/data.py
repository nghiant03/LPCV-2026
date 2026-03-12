from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help="Dataset operations.")


@app.command()
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


@app.command()
def precompute(
    data_dir: Annotated[Path, typer.Argument(help="Path to QEVD dataset or cached DatasetDict.")],
    output_dir: Annotated[
        Path,
        typer.Argument(help="Directory to save precomputed DatasetDict."),
    ],
    is_cache: Annotated[
        bool,
        typer.Option("--is-cache", help="Treat data_dir as a saved DatasetDict."),
    ] = False,
    num_workers: Annotated[
        int | None,
        typer.Option(
            "--num-workers", "-w", help="Number of parallel workers for preprocessing and saving."
        ),
    ] = None,
) -> None:
    """Precompute dataset: decode video frames and save to disk."""
    from datasets import DatasetDict, load_from_disk

    from lpcv.datasets.precompute import PrecomputedDataset
    from lpcv.datasets.qevd import QEVDAdapter

    if is_cache:
        dataset = load_from_disk(str(data_dir))
        if not isinstance(dataset, DatasetDict):
            raise typer.BadParameter(
                f"Expected DatasetDict at {data_dir}, got {type(dataset).__name__}"
            )
    else:
        adapter = QEVDAdapter(data_dir=data_dir)
        dataset = adapter.load()

    pds = PrecomputedDataset(
        dataset=dataset,
        num_workers=num_workers,
    )
    pds.precompute(output_dir)


@app.command()
def cache(
    data_dir: Annotated[Path, typer.Argument(help="Root directory of the QEVD dataset.")],
    cache_dir: Annotated[
        Path,
        typer.Argument(help="Directory to save/load the cached DatasetDict."),
    ],
) -> None:
    """Cache the QEVD DatasetDict to disk for fast reloading."""
    from lpcv.datasets.qevd import QEVDAdapter

    adapter = QEVDAdapter(data_dir=data_dir)
    adapter.load(cache_dir=cache_dir)
