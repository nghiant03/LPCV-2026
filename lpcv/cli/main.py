import typer

from lpcv.cli.data import app as data_app
from lpcv.cli.evaluate import app as evaluate_app
from lpcv.cli.train import app as train_app

app = typer.Typer()
app.add_typer(data_app, name="data")
app.add_typer(train_app, name="train")
app.add_typer(evaluate_app, name="evaluate")


if __name__ == "__main__":
    app()
