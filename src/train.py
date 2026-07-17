import argparse

from mlp import MLP
from io_utils import load_dataset, save_model


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
    args = parser.parse_args()

    print(f"Epochs : {args.epochs}")
    print(f"Batch size : {args.batch_size}")
    print(f"Learning rate : {args.lr}")
    print(f"Sizes of hidden layers : {args.hidden}")
    print(f"Model path : {args.model}\n")

    X_train, y_train = load_dataset(args.train)
    X_val, y_val = load_dataset(args.val)
    print(f"Train file: {args.train}")
    print(f"Validation file: {args.val}\n")

    input_size = X_train.shape[1]
    mlp = MLP(
        number_hidden_layer=len(args.hidden),
        input_size=input_size,
        hidden_sizes=args.hidden,
        output_size=1,
        learning_rate=args.lr,
        batch_size=args.batch_size,
    )

    mlp.train(X_train, y_train, X_val, y_val, epochs=args.epochs)
    mlp.plot_metrics()

    save_model(mlp, args.model)
    print(f"✅ Model saved to {args.model}")


if __name__ == "__main__":
    main()
