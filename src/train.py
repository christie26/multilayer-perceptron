import argparse

from io_utils import load_dataset, save_history, save_model
from mlp import MLP


def main():
    parser = argparse.ArgumentParser(description="Train MLP model")
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Mini-batch size used for training"
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="Learning rate for gradient descent"
    )
    parser.add_argument(
        "--hidden",
        nargs="+",
        type=int,
        default=[5, 10],
        help="Sizes of hidden layers (space-separated). Example: --hidden 64 32",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="categoricalCrossentropy",
        choices=["categoricalCrossentropy"],
        help="Loss function",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="sigmoid",
        choices=["sigmoid"],
        help="Hidden layer activation function",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "momentum", "rmsprop", "adam"],
        help="Weight update rule",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Stop early after N epochs without val-loss improvement (0 = disabled)",
    )
    parser.add_argument(
        "--train", type=str, default="data_train.npz", help="Training dataset file"
    )
    parser.add_argument(
        "--val", type=str, default="data_val.npz", help="Validation dataset file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlp_model.npz",
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--history",
        type=str,
        default="mlp_history.json",
        help="Path to save the training history",
    )
    args = parser.parse_args()

    print(f"Epochs : {args.epochs}")
    print(f"Batch size : {args.batch_size}")
    print(f"Learning rate : {args.lr}\n")
    print(f"Sizes of hidden layers : {args.hidden}")
    print(f"Loss : {args.loss}")
    print(f"Activation : {args.activation}")
    print(f"Optimizer : {args.optimizer}\n")

    X_train, y_train = load_dataset(args.train)
    X_val, y_val = load_dataset(args.val)
    print(f"Train file: {args.train}")
    print(f"Validation file: {args.val}")
    print(f"Model path : {args.model}\n")

    input_size = X_train.shape[1]
    mlp = MLP(
        number_hidden_layer=len(args.hidden),
        input_size=input_size,
        hidden_sizes=args.hidden,
        output_size=2,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        optimizer=args.optimizer,
    )

    mlp.train(
        X_train, y_train, X_val, y_val, epochs=args.epochs, patience=args.patience
    )
    mlp.plot_metrics()

    save_model(mlp, args.model)
    save_history(mlp, args.history)
    print(f"✅ Model saved to {args.model}")
    print(f"✅ History saved to {args.history}")


if __name__ == "__main__":
    main()
