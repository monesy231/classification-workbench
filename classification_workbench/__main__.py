"""Run one experiment with python -m classification_workbench."""

import argparse
from importlib import import_module
from pathlib import Path


def nonnegative_integer(value):
    number = int(value)
    if not 0 <= number < 2**32:
        raise argparse.ArgumentTypeError("expected an integer from 0 to 2**32 - 1")
    return number


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def main():
    parser = argparse.ArgumentParser(description="Run a classification experiment.")
    commands = parser.add_subparsers(dest="experiment", required=True)
    descriptions = {
        "bank": "Compare a scikit-learn decision tree and categorical ID3.",
        "digits": "Train a PyTorch MLP on MNIST and compare architecture choices.",
        "news": "Compare sparse text classifiers on Chinese headlines.",
    }
    for name, description in descriptions.items():
        command = commands.add_parser(name, help=description, description=description)
        command.add_argument("--seed", type=nonnegative_integer, default=42)
        command.add_argument("--out", type=Path, help="Output directory (default: under outputs/).")
        source = command.add_mutually_exclusive_group(required=name != "digits")
        source.add_argument("--demo", action="store_true", help="Run with offline demo data.")
        source.add_argument("--data-dir", type=Path, help="Directory containing local data files.")
        if name == "digits":
            command.add_argument("--epochs", type=positive_integer, default=5)
            command.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
            command.add_argument("--sweep", action="store_true", help="Compare single-factor MLP variants.")
            command.add_argument(
                "--extensions",
                action="store_true",
                help="Try joint tuning, training-only augmentation, and a small CNN.",
            )

    args = parser.parse_args()
    if args.out is None:
        run_name = f"{args.experiment}-demo" if args.demo else args.experiment
        args.out = Path("outputs") / run_name

    import matplotlib

    matplotlib.use("Agg")
    experiment = import_module(f".{args.experiment}", package=__package__)
    try:
        experiment.run(args)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
