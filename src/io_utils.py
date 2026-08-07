import csv
import json
import os
import sys

import numpy as np

from mlp import MLP


def load_data(filename):
    """Read the raw diagnosis CSV and return standardized features X and one-hot labels y."""
    data = []
    labels = []
    try:
        with open(filename, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 32:
                    continue
                label = 1 if parts[1] == "M" else 0
                features = list(map(float, parts[2:]))
                data.append(features)
                labels.append(label)
    except FileNotFoundError:
        print(f"❌ Error: File '{filename}' not found.\nMake sure you have right file.")
        sys.exit(1)
    except PermissionError:
        print(
            f"❌ Error: No permission to read '{filename}'.\nMake sure you have permission to read"
        )
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error while reading '{filename}': {e}")
        sys.exit(1)

    if not data:
        print(f"❌ Error: No valid data found in '{filename}'.")
        sys.exit(1)

    X = np.array(data)
    y = np.zeros((len(labels), 2))
    y[np.arange(len(labels)), labels] = 1

    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, y


def train_test_split(X, y, train_ratio=0.8, seed=42):
    """Randomly split X, y into train and validation subsets."""
    num_samples = len(X)
    num_train = int(num_samples * train_ratio)

    rng = np.random.default_rng(seed=seed)
    indices = rng.permutation(num_samples)

    train_idx = indices[:num_train]
    test_idx = indices[num_train:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def load_dataset(filename):
    """Load a prepared .npz dataset and return its X, y arrays."""
    try:
        data = np.load(filename)
    except FileNotFoundError:
        print(f"❌ Dataset file '{filename}' not found. Run prepare_data.py first.")
        sys.exit(1)
    return data["X"], data["y"]


def save_model(model, filename):
    """Persist an MLP's weights, biases, topology and hyperparameters to an .npz file."""
    arrays = {}
    for i, w in enumerate(model.weights):
        arrays[f"weight_{i}"] = w
    for i, b in enumerate(model.biases):
        arrays[f"bias_{i}"] = b

    arrays["hidden_sizes"] = np.array([w.shape[1] for w in model.weights[:-1]])
    arrays["input_size"] = np.array(model.weights[0].shape[0])
    arrays["output_size"] = np.array(model.weights[-1].shape[1])
    arrays["learning_rate"] = np.array(model.learning_rate)
    arrays["batch_size"] = np.array(model.batch_size)
    arrays["optimizer"] = np.array(model.optimizer)

    np.savez(filename, **arrays)


def load_model(filename):
    """Rebuild an MLP (topology + hyperparameters + weights) from a saved .npz file."""
    try:
        data = np.load(filename, allow_pickle=True)
    except FileNotFoundError:
        print(f"❌ Model file '{filename}' not found. Run train.py first.")
        return None

    weights = [data[k] for k in sorted(data.files) if k.startswith("weight_")]
    biases = [data[k] for k in sorted(data.files) if k.startswith("bias_")]

    mlp = MLP(
        number_hidden_layer=len(data["hidden_sizes"]),
        input_size=int(data["input_size"]),
        hidden_sizes=list(data["hidden_sizes"]),
        output_size=int(data["output_size"]),
        learning_rate=float(data["learning_rate"]),
        batch_size=int(data["batch_size"]),
        optimizer=str(data["optimizer"]),
    )
    mlp.weights = weights
    mlp.biases = biases
    return mlp


def save_history(model, filename):
    """Persist per-epoch train/val loss & accuracy history to a JSON file."""
    history = {
        "train_loss": model.train_loss_history,
        "train_acc": model.train_acc_history,
        "val_loss": model.val_loss_history,
        "val_acc": model.val_acc_history,
    }
    with open(filename, "w") as f:
        json.dump(history, f)


def load_history(filename):
    """Load a previously saved training history JSON file."""
    with open(filename, "r") as f:
        return json.load(f)


def append_run_log(csv_path, row):
    """Append one run's tag/hyperparameters/timestamp as a row to the shared run-log CSV."""
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_run_log(csv_path):
    """Read the run-log CSV and return a list of row dicts."""
    with open(csv_path, "r", newline="") as f:
        return list(csv.DictReader(f))
