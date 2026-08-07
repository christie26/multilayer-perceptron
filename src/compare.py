import argparse

import matplotlib.pyplot as plt

from io_utils import load_history, load_run_log


def main():
    parser = argparse.ArgumentParser(
        description="Compare validation loss across previously trained runs"
    )
    parser.add_argument(
        "--run_log",
        type=str,
        default="tag.csv",
        help="Path to the CSV log recording every run's tag and hyperparameters",
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        default=None,
        help="Tags to compare (default: all tags in the run log)",
    )
    args = parser.parse_args()

    runs = load_run_log(args.run_log)
    if args.tags is not None:
        runs = [r for r in runs if r["tag"] in args.tags]

    plt.figure(figsize=(7, 5))
    for run in runs:
        try:
            history = load_history(run["history_file"])
        except FileNotFoundError:
            print(f"⚠️  Skipping '{run['tag']}': {run['history_file']} not found")
            continue
        plt.plot(history["val_loss"], label=run["tag"])

    plt.xlabel("Epoch")
    plt.ylabel("Validation loss")
    plt.title("Model comparison")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
