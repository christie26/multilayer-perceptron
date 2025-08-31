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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--hidden", nargs="+", type=int, default=[5, 10])
    parser.add_argument("--model", type=str, default="mlp_model.npz")
    args = parser.parse_args()

    train = np.load("data_train.npz")
    val = np.load("data_val.npz")
    X_train, y_train = train["X"], train["y"]
    X_val, y_val = val["X"], val["y"]

    input_size = X_train.shape[1]
    mlp = MLP(
        number_hidden_layer=len(args.hidden),
        input_size=input_size,
        hidden_sizes=args.hidden,
        output_size=1,
        learning_rate=args.lr,
    )

    mlp.train(
        X_train, y_train, X_val, y_val, epochs=args.epochs, batch_size=args.batchsize
    )
    mlp.plot_metrics()

    save_model(mlp, args.model)
    print(f"✅ Model saved to {args.model}")
