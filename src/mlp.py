import numpy as np
import matplotlib.pyplot as plt


# Activation and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def accuracy(y_true, y_pred):
    predictions = (y_pred > 0.5).astype(int)
    correct = np.sum(predictions == y_true)
    return correct / len(y_true)


class MLP:
    def __init__(
        self,
        number_hidden_layer: int,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
        learning_rate=0.1,
    ):
        self.learning_rate = learning_rate

        if number_hidden_layer != len(hidden_sizes):
            raise ValueError(
                f"Number of hidden layers ({number_hidden_layer}) does not match the length of hidden_sizes ({len(hidden_sizes)})."
            )

        layer_sizes = [input_size] + hidden_sizes + [output_size]

        weights, biases = self.initialize_weights(layer_sizes)

        self.weights = [weights[i].T for i in range(len(weights))]
        self.biases = [biases[i].T for i in range(len(biases))]

        # for weights in self.weights:
        #     print(weights.shape)
        # print("----------------------")

    def initialize_weights(self, layer_sizes, initialization="he"):
        """
        Initialize weights for an MLP.

        Parameters:
            layer_sizes (list): Sizes of each layer including input and output layers.
            initialization (str): 'xavier', 'he', or 'random'

        Returns:
            weights (list of np.ndarray): List of weight matrices.
            biases (list of np.ndarray): List of bias vectors.
        """
        weights = []
        biases = []

        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]

            if initialization == "xavier":
                limit = np.sqrt(6 / (fan_in + fan_out))
                W = np.random.uniform(-limit, limit, (fan_out, fan_in))
            elif initialization == "he":
                std = np.sqrt(2 / fan_in)
                W = np.random.randn(fan_out, fan_in) * std
            elif initialization == "random":
                W = np.random.randn(fan_out, fan_in) * 0.01
            else:
                raise ValueError("Unsupported initialization method")

            b = np.zeros((fan_out, 1))

            weights.append(W)
            biases.append(b)

        return weights, biases

    def forward(self, X):
        self.inputs = [X]
        self.activations = []

        for i in range(len(self.weights)):
            input_layer = np.dot(self.inputs[-1], self.weights[i]) + self.biases[i]
            activation_layer = sigmoid(input_layer)
            self.inputs.append(input_layer)
            self.activations.append(activation_layer)

        return self.activations[-1]

    def backward(self, X, y, output):
        error = y - output
        delta = error * sigmoid_derivative(output)

        for i in reversed(range(len(self.weights))):

            self.weights[i] += self.inputs[i].T.dot(delta) * self.learning_rate
            self.biases[i] += np.sum(delta, axis=0, keepdims=True) * self.learning_rate

            if i != 0:
                delta = delta.dot(self.weights[i].T) * sigmoid_derivative(
                    self.activations[i - 1]
                )

    def train(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
        epochs=10000,
        batch_size=32,
        shuffle=True,
    ):
        n_samples = X.shape[0]

        self.train_loss_history = []
        self.train_acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []

        for epoch in range(epochs):
            if shuffle:
                indices = np.arange(n_samples)
                np.random.shuffle(indices)
                X = X[indices]
                y = y[indices]

            # ---- Mini-batch training ----
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X[start:end]
                y_batch = y[start:end]

                output = self.forward(X_batch)
                self.backward(X_batch, y_batch, output)

            # ---- Metrics after epoch ----
            # Train metrics
            train_output = self.forward(X)
            train_loss = np.mean((y - train_output) ** 2)
            train_acc = accuracy(y, train_output)
            self.train_loss_history.append(train_loss)
            self.train_acc_history.append(train_acc)

            # Validation metrics (if provided)
            if X_val is not None and y_val is not None:
                val_output = self.forward(X_val)
                val_loss = np.mean((y_val - val_output) ** 2)
                val_acc = accuracy(y_val, val_output)
                self.val_loss_history.append(val_loss)
                self.val_acc_history.append(val_acc)

            if epoch % 10 == 0:
                if X_val is not None:
                    print(
                        f"Epoch {epoch}, Train Loss: {train_loss:.4f}, "
                        f"Train Acc: {train_acc:.4f}, "
                        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                    )
                else:
                    print(
                        f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
                    )

    def plot_metrics(self):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 5))

        # ---- Loss curve ----
        plt.subplot(1, 2, 1)
        plt.plot(self.train_loss_history, label="Train Loss")
        if len(self.val_loss_history) > 0:
            plt.plot(self.val_loss_history, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.legend()

        # ---- Accuracy curve ----
        plt.subplot(1, 2, 2)
        plt.plot(self.train_acc_history, label="Train Accuracy")
        if len(self.val_acc_history) > 0:
            plt.plot(self.val_acc_history, label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Curve")
        plt.legend()

        plt.tight_layout()
        plt.show()
