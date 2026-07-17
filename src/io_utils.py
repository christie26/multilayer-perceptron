import sys
import numpy as np

from mlp import MLP


def load_data(filename):
    """Read the raw diagnosis CSV and return standardized features X and labels y."""
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
                labels.append([label])
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
    y = np.array(labels)

    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, y


def train_test_split(X, y, train_ratio=0.8):
    """Randomly split X, y into train and validation subsets."""
    num_samples = len(X)
    num_train = int(num_samples * train_ratio)

    indices = np.random.permutation(num_samples)
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
    """Persist an MLP's weights and biases to an .npz file."""
    arrays = {}
    for i, w in enumerate(model.weights):
        arrays[f"weight_{i}"] = w
    for i, b in enumerate(model.biases):
        arrays[f"bias_{i}"] = b

    np.savez(filename, **arrays)


def load_model(filename):
    """Rebuild an MLP from a saved .npz file. Returns None if the file is missing."""
    try:
        data = np.load(filename, allow_pickle=True)
    except FileNotFoundError:
        print(f"❌ Model file '{filename}' not found. Run train.py first.")
        return None

    weights = []
    biases = []
    for key in sorted(data.files):
        if key.startswith("weight"):
            weights.append(data[key])
        elif key.startswith("bias"):
            biases.append(data[key])

    mlp = MLP(
        number_hidden_layer=len(weights) - 1,
        input_size=weights[0].shape[0],
        hidden_sizes=[w.shape[1] for w in weights[:-1]],
        output_size=weights[-1].shape[1],
        learning_rate=0.01,
    )
    mlp.weights = list(weights)
    mlp.biases = list(biases)
    return mlp
