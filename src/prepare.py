import argparse
import numpy as np

from io_utils import load_data, train_test_split


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset")
    parser.add_argument("--data", type=str, default="data.csv", help="Input CSV file")
    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="Train/test split ratio"
    )
    parser.add_argument(
        "--train_out", type=str, default="data_train.npz", help="Output train file"
    )
    parser.add_argument(
        "--val_out", type=str, default="data_val.npz", help="Output validation file"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed to shuffle"
    )
    args = parser.parse_args()

    X, y = load_data(args.data)
    X_train, y_train, X_val, y_val = train_test_split(X, y, args.train_ratio, args.seed)

    np.savez(args.train_out, X=X_train, y=y_train)
    np.savez(args.val_out, X=X_val, y=y_val)
    print(
        f"Total dataset: {len(X)}\n"
        f"train_ratio: {args.train_ratio}\n"
        f"Saved {args.train_out}, {len(X_train)}\n"
        f"Saved {args.val_out}, {len(X_val)}"
    )


if __name__ == "__main__":
    main()
