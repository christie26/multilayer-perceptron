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
    test = np.load("data_test.npz")
    X_test, y_test = test["X"], test["y"]

    mlp = load_model("mlp_model.npz")

    correct = 0
    for i in range(len(X_test)):
        output = mlp.forward(X_test[i])
        predicted_label = 1 if output >= 0.5 else 0
        actual_label = y_test[i][0]
        correct += predicted_label == actual_label

    accuracy = correct / len(X_test) * 100 if len(X_test) > 0 else 0
    print(f"✅ Accuracy on Test Set: {accuracy:.2f}%")
