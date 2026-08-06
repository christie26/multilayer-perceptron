import argparse
import sys

import numpy as np

from io_utils import load_dataset, load_model


def confusion_counts(predicted, actual):
    tp = np.sum((predicted == 1) & (actual == 1))
    tn = np.sum((predicted == 0) & (actual == 0))
    fp = np.sum((predicted == 1) & (actual == 0))
    fn = np.sum((predicted == 0) & (actual == 1))
    return tp, tn, fp, fn


def main():
    parser = argparse.ArgumentParser(description="Predict with a trained MLP model")
    parser.add_argument(
        "--data", type=str, default="data_val.npz", help="Dataset file to predict on"
    )
    parser.add_argument(
        "--model", type=str, default="mlp_model.npz", help="Trained model file"
    )
    args = parser.parse_args()

    X, y = load_dataset(args.data)
    mlp = load_model(args.model)
    if mlp is None:
        sys.exit(1)

    actual = np.argmax(y, axis=1)
    outputs = mlp.forward(X)
    p_malignant = outputs[:, 1]
    predicted = np.argmax(outputs, axis=1)

    accuracy = np.mean(predicted == actual) * 100
    bce = -np.mean(
        actual * np.log(p_malignant + 1e-9)
        + (1 - actual) * np.log(1 - p_malignant + 1e-9)
    )

    tp, _tn, fp, fn = confusion_counts(predicted, actual)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"✅ Accuracy: {accuracy:.2f}%")
    print(f"Binary cross-entropy: {bce:.4f}")
    print(f"Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")


if __name__ == "__main__":
    main()
