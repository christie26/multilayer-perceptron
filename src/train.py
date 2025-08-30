# train.py
import numpy as np
from mlp import MLP

def load_data(filename):
    data = []
    labels = []
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 32:
                continue
            label = 1 if parts[1] == "M" else 0
            features = list(map(float, parts[2:]))
            data.append(features)
            labels.append([label])
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

def save_model(model, filename):
    with open(filename, "wb") as f:
        for i in range(len(model.weights)):
            np.save(f, model.weights[i])
            np.save(f, model.biases[i])

if __name__ == "__main__":
    X, y = load_data("data.csv")
    X_train, y_train, X_test, y_test = train_test_split(X, y, train_ratio=0.8)

    input_size = X.shape[1]
    number_hidden_layer = 2
    hidden_sizes = [5,10]
    output_size = 1
    learning_rate = 0.01
    epochs = 10000

    mlp = MLP(number_hidden_layer = number_hidden_layer, input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size, learning_rate=learning_rate)
    mlp.train(X_train, y_train, epochs=epochs)

    # Evaluation on the test set
    correct = 0
    for i in range(len(X_test)):
        output = mlp.forward(X_test[i])
        predicted_label = 1 if output >= 0.5 else 0
        actual_label = y_test[i][0]
        is_correct = predicted_label == actual_label
        correct += is_correct

    accuracy = correct / len(X_test) * 100 if len(X_test) > 0 else 0
    print(f"✅ Accuracy on Test Set: {accuracy:.2f}%")

    # Save model weights to file
    save_model(mlp, "mlp_model.npy")

    # Log training session details
    with open("training_log.txt", "a") as log_file:
        log_file.write("Training Session\n")
        log_file.write(f"Epochs: {epochs}\n")
        log_file.write(f"Learning Rate: {learning_rate}\n")
        log_file.write(f"Hidden Layer Neurons: {hidden_sizes[0]}, {hidden_sizes[1]}\n")
        log_file.write(f"Test Accuracy: {accuracy:.2f}%\n")
        log_file.write("-" * 30 + "\n")
