import numpy as np
from mlp import MLP


def load_model(filename):
    try:
        data = np.load(filename, allow_pickle=True)
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
    except FileNotFoundError:
        print(f"❌ Model file '{filename}' not found.")
        return None, None


if __name__ == "__main__":
    test = np.load("data_val.npz")
    X_val, y_val = test["X"], test["y"]
    mlp = load_model("mlp_model.npz")
    actual_labels = y_val.flatten()

    # Predict with sigmoid ================================
    outputs = mlp.forward(X_val).flatten()
    print(f"from sigmoid {outputs}")
    predicted_labels = (outputs >= 0.5).astype(int)
    correct = (predicted_labels == actual_labels).sum()

    accuracy = correct / len(X_val) * 100 if len(X_val) > 0 else 0
    print(f"✅ Accuracy with sigmoid: {accuracy:.2f}%")

    # Predict with softmax ================================
    probs, pred_classes = mlp.predict(X_val)
    print(f"from softmax {pred_classes}")

    correct = np.sum(pred_classes == actual_labels)
    accuracy = correct / len(X_val) * 100 if len(X_val) > 0 else 0
    print(f"✅ Accuracy with softmax: {accuracy:.2f}%")
