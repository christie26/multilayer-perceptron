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
            features = list(map(float, parts[2:]))  # 30 features
            data.append(features)
            labels.append([label])
    X = np.array(data)
    y = np.array(labels)

    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, y

def train_test_split(X, y, train_ratio=0.8):
    num_samples = len(X)
    num_train = int(num_samples * train_ratio)
    # Shuffle data
    indices = np.random.permutation(num_samples)
    train_idx = indices[:num_train]
    test_idx = indices[num_train:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

if __name__ == "__main__":
    X, y = load_data("data.csv")
    
    # Split into 80% train and 20% test
    X_train, y_train, X_test, y_test = train_test_split(X, y, train_ratio=0.8)

    # Define hyperparameters
    input_size = X.shape[1]
    hidden_size1 = 10
    hidden_size2 = 10
    output_size = 1
    learning_rate = 0.01
    epochs = 10000

    # Initialize and train MLP
    mlp = MLP(input_size=input_size, hidden_size1=hidden_size1, hidden_size2=hidden_size2, output_size=output_size, learning_rate=learning_rate)
    mlp.train(X_train, y_train, epochs=epochs)

    # Evaluation
    print(f"\n🔍 Evaluation on Test Set")
    print(f"➡ Epochs: {epochs}")
    print(f"➡ Learning Rate: {learning_rate}")
    print(f"\nPredictions:")

    correct = 0
    for i in range(len(X_test)):
        output = mlp.forward(X_test[i])
        predicted_label = 1 if output >= 0.5 else 0
        actual_label = y_test[i][0]
        is_correct = predicted_label == actual_label
        correct += is_correct
        print(f"Input {i+1}: Predicted={predicted_label}, Actual={actual_label}, Raw={output.round(3)}, {'✅' if is_correct else '❌'}")

    accuracy = correct / len(X_test) * 100 if len(X_test) > 0 else 0
    print(f"➡ Epochs: {epochs}")
    print(f"➡ Learning Rate: {learning_rate}")
    print(f"\n✅ Accuracy on Test Set: {accuracy:.2f}%")

    with open("training_log.txt", "a") as log_file:
        log_file.write("Training Session\n")
        log_file.write(f"Epochs: {epochs}\n")
        log_file.write(f"Learning Rate: {learning_rate}\n")
        log_file.write(f"Hidden Layer Neurons: {hidden_size1}, {hidden_size2}\n")
        log_file.write(f"Test Accuracy: {accuracy:.2f}%\n")
        log_file.write("-" * 30 + "\n")