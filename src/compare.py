import argparse

import matplotlib.pyplot as plt

from io_utils import load_dataset
from mlp import MLP


def parse_run(spec):
    """Parse a 'name:hidden1,hidden2:lr:optimizer' run spec."""
    name, hidden, lr, optimizer = spec.split(":")
    hidden_sizes = [int(h) for h in hidden.split(",")]
    return name, hidden_sizes, float(lr), optimizer


def main():
    parser = argparse.ArgumentParser(
        description="Train multiple MLP configurations and compare their validation loss"
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="One spec per run: name:hidden1,hidden2:lr:optimizer",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train", type=str, default="data_train.npz")
    parser.add_argument("--val", type=str, default="data_val.npz")
    args = parser.parse_args()

    X_train, y_train = load_dataset(args.train)
    X_val, y_val = load_dataset(args.val)

    plt.figure(figsize=(7, 5))
    for spec in args.runs:
        name, hidden_sizes, lr, optimizer = parse_run(spec)
        print(f"--- training run '{name}' ---")
        mlp = MLP(
            number_hidden_layer=len(hidden_sizes),
            input_size=X_train.shape[1],
            hidden_sizes=hidden_sizes,
            output_size=2,
            learning_rate=lr,
            batch_size=args.batch_size,
            optimizer=optimizer,
        )
        mlp.train(X_train, y_train, X_val, y_val, epochs=args.epochs)
        plt.plot(mlp.val_loss_history, label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Validation loss")
    plt.title("Model comparison")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
