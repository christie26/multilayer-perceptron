import numpy as np
import argparse
import sys


def load_data(filename):
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
    num_samples = len(X)
    num_train = int(num_samples * train_ratio)

    indices = np.random.permutation(num_samples)
    train_idx = indices[:num_train]
    test_idx = indices[num_train:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset")
    parser.add_argument("--data", type=str, default="data.csv", help="Input CSV file")
    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="Train/test split ratio"
    )
    args = parser.parse_args()

    X, y = load_data(args.data)
    X_train, y_train, X_test, y_test = train_test_split(X, y, args.train_ratio)

    np.savez("data_train.npz", X=X_train, y=y_train)
    np.savez("data_test.npz", X=X_test, y=y_test)
    print(
        f"filename: {args.data}\ntrain_ratio: {args.train_ratio}\n✅ Saved data_train.npz and data_test.npz"
    )
