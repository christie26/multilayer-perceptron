import numpy as np
from mlp import MLP

def load_model(filename):
    with open(filename, "rb") as f:
        weights1 = np.load(f)
        weights2 = np.load(f)
        bias1 = np.load(f)
        bias2 = np.load(f)
    return weights1, weights2, bias1, bias2

def load_data(filename):
    data = []
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 32:
                continue
            features = list(map(float, parts[1:]))  # Assuming no labels in prediction mode
            data.append(features)
    return np.array(data)

if __name__ == "__main__":
    # Load the trained model's parameters
    weights1, weights2, bias1, bias2 = load_model("mlp_model.npy")

    # Prepare the MLP model
    input_size = 30  # Adjust according to the input features
    hidden_size1 = 5
    hidden_size2 = 10
    output_size = 1
    mlp = MLP(input_size, hidden_size1, hidden_size2, output_size, learning_rate=0)

    # Manually assign the weights and biases to the MLP model
    mlp.weights1 = weights1
    mlp.weights2 = weights2
    mlp.bias1 = bias1
    mlp.bias2 = bias2

    # Load new data for prediction
    X_new = load_data("new_data.csv")  # Use the filename of your new data

    # Make predictions
    print("\n🔍 Predictions:")
    for i, x in enumerate(X_new):
        output = mlp.forward(x)
        predicted_label = 1 if output >= 0.5 else 0
        print(f"Input {i+1}: Predicted={predicted_label}, Raw Output={output.round(3)}")
