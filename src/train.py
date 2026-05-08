import numpy as np
import argparse
from mlp import MLP


def save_model(model, filename):
    arrays = {}
    for i, w in enumerate(model.weights):
        arrays[f"weight_{i}"] = w
    for i, b in enumerate(model.biases):
        arrays[f"bias_{i}"] = b

    np.savez(filename, **arrays)


if __name__ == "__main__":
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
        "--model",
        type=str,
        default="mlp_model.npz",
        help="Path to save or load the trained model",
    )
    args = parser.parse_args()

    print(f"Epochs : {args.epochs}")
    print(f"Batch size : {args.batch_size}")
    print(f"Learning rate : {args.lr}")
    print(f"Sizes of hidden layers : {args.hidden}")
    print(f"Model path : {args.model}\n")

    train_file = "data_train.npz"
    validation_file = "data_val.npz"
    train = np.load(train_file)
    val = np.load(validation_file)
    print(f"Train file: {train_file}")
    print(f"Validation file: {validation_file}\n")

    X_train, y_train = train["X"], train["y"]
    X_val, y_val = val["X"], val["y"]
    # print(f"X_val: {X_val}, y_val: {y_val}")

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
