"""CLI entry point — mounts sub-command groups for data, training, evaluation, and submission."""

import typer

from lpcv.cli.data import app as data_app
from lpcv.cli.evaluate import app as evaluate_app
from lpcv.cli.submit import app as submit_app
from lpcv.cli.train import train as train_command

app = typer.Typer()
app.command(name="train")(train_command)

app.add_typer(data_app, name="data")
app.add_typer(evaluate_app, name="evaluate")
app.add_typer(submit_app, name="submit")


if __name__ == "__main__":
    app()
