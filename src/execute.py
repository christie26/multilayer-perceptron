import numpy as np
from mlp import MLP


def load_data(filename):
    data = []
    labels = []

    with open(filename, "r") as f:
        for line in f.readlines():
            parts = line.strip().split(",")
            if len(parts) < 32:  # ID + Label + 30 Features
                continue
            # Label 처리 (M=1, B=0)
            label = 1 if parts[1] == "M" else 0
            features = list(map(float, parts[2:]))  # Feature 30개
            data.append(features)
            labels.append([label])

    X = np.array(data)
    y = np.array(labels)
    return X, y


if __name__ == "__main__":

    X, y = load_data("../data.csv")
    print("X", X)
    print("y", y)

    # Train
    mlp = MLP(input_size=30, hidden_size=16, output_size=1, learning_rate=0.01)
    mlp.train(X, y, epochs=2000)

    # Predict
    print("\nPredictions for first 5 samples:")
    for i in range(5):
        output = mlp.forward(X[i])
        predicted_label = 1 if output >= 0.5 else 0
        print(
            f"Input {i+1}: Predicted={predicted_label}, Actual={y[i][0]}, Raw={output.round(3)}"
        )
