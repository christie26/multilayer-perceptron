import argparse
import sys
import numpy as np

from io_utils import load_dataset, load_model


def evaluate(predicted_labels, actual_labels):
    """Return accuracy (%) of predicted labels against the ground truth."""
    n = len(actual_labels)
    if n == 0:
        return 0.0
    correct = np.sum(predicted_labels == actual_labels)
    return correct / n * 100


def main():
    parser = argparse.ArgumentParser(description="Predict with a trained MLP model")
    parser.add_argument(
        "--data", type=str, default="data_val.npz", help="Dataset file to predict on"
    )
    parser.add_argument(
        "--model", type=str, default="mlp_model.npz", help="Trained model file"
    )
    args = parser.parse_args()

    X_val, y_val = load_dataset(args.data)
    mlp = load_model(args.model)
    if mlp is None:
        sys.exit(1)
    actual_labels = y_val.flatten()

    # ---- Predict with sigmoid ----
    outputs = mlp.forward(X_val).flatten()
    print(f"from sigmoid {outputs}")
    sigmoid_labels = (outputs >= 0.5).astype(int)
    print(f"✅ Accuracy with sigmoid: {evaluate(sigmoid_labels, actual_labels):.2f}%")

    # ---- Predict with softmax ----
    _, pred_classes = mlp.predict(X_val)
    print(f"from softmax {pred_classes}")
    print(f"✅ Accuracy with softmax: {evaluate(pred_classes, actual_labels):.2f}%")


if __name__ == "__main__":
    main()
