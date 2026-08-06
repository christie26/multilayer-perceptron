import matplotlib.pyplot as plt
import numpy as np


# Activation and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(a):
    return a * (1 - a)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # prevent overflow
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))


def accuracy(y_true, y_pred):
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))


class MLP:
    def __init__(
        self,
        number_hidden_layer: int,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
        learning_rate=0.1,
        batch_size=32,
        optimizer="sgd",
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer = optimizer

        if number_hidden_layer != len(hidden_sizes):
            raise ValueError(
                f"Number of hidden layers ({number_hidden_layer}) does not match the length of hidden_sizes ({len(hidden_sizes)})."
            )

        layer_sizes = [input_size] + list(hidden_sizes) + [output_size]

        weights, biases = self.initialize_weights(layer_sizes)

        self.weights = [weights[i].T for i in range(len(weights))]
        self.biases = [biases[i].T for i in range(len(biases))]
        self._init_optimizer_state()

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

    def _init_optimizer_state(self):
        self.t = 0
        self.mW = [np.zeros_like(w) for w in self.weights]
        self.mB = [np.zeros_like(b) for b in self.biases]
        self.vW = [np.zeros_like(w) for w in self.weights]
        self.vB = [np.zeros_like(b) for b in self.biases]

    def forward(self, X):
        self.activations = [X]

        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            a = softmax(z) if i == len(self.weights) - 1 else sigmoid(z)
            self.activations.append(a)

        return self.activations[-1]

    def _apply_update(self, i, grad_w, grad_b):
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        if self.optimizer == "momentum":
            self.vW[i] = beta1 * self.vW[i] + (1 - beta1) * grad_w
            self.vB[i] = beta1 * self.vB[i] + (1 - beta1) * grad_b
            self.weights[i] += self.learning_rate * self.vW[i]
            self.biases[i] += self.learning_rate * self.vB[i]
        elif self.optimizer == "rmsprop":
            self.vW[i] = beta2 * self.vW[i] + (1 - beta2) * grad_w**2
            self.vB[i] = beta2 * self.vB[i] + (1 - beta2) * grad_b**2
            self.weights[i] += self.learning_rate * grad_w / (np.sqrt(self.vW[i]) + eps)
            self.biases[i] += self.learning_rate * grad_b / (np.sqrt(self.vB[i]) + eps)
        elif self.optimizer == "adam":
            self.mW[i] = beta1 * self.mW[i] + (1 - beta1) * grad_w
            self.mB[i] = beta1 * self.mB[i] + (1 - beta1) * grad_b
            self.vW[i] = beta2 * self.vW[i] + (1 - beta2) * grad_w**2
            self.vB[i] = beta2 * self.vB[i] + (1 - beta2) * grad_b**2

            mW_hat = self.mW[i] / (1 - beta1**self.t)
            mB_hat = self.mB[i] / (1 - beta1**self.t)
            vW_hat = self.vW[i] / (1 - beta2**self.t)
            vB_hat = self.vB[i] / (1 - beta2**self.t)

            self.weights[i] += self.learning_rate * mW_hat / (np.sqrt(vW_hat) + eps)
            self.biases[i] += self.learning_rate * mB_hat / (np.sqrt(vB_hat) + eps)
        else:  # sgd
            self.weights[i] += self.learning_rate * grad_w
            self.biases[i] += self.learning_rate * grad_b

    def backward(self, y, output):
        # softmax output + categorical cross-entropy: combined gradient is (output - y);
        # we track the error direction y - output so weight updates stay additive.
        delta = y - output

        if self.optimizer == "adam":
            self.t += 1

        for i in reversed(range(len(self.weights))):
            a_prev = self.activations[i]
            grad_w = a_prev.T.dot(delta)
            grad_b = np.sum(delta, axis=0, keepdims=True)

            if i != 0:
                # propagate error through the pre-update weights, before this layer's own update
                delta = delta.dot(self.weights[i].T) * sigmoid_derivative(self.activations[i])

            self._apply_update(i, grad_w, grad_b)

    def train(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
        epochs=10000,
        shuffle=True,
        patience=0,
    ):
        n_samples = X.shape[0]

        self.train_loss_history = []
        self.train_acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []

        best_val_loss = np.inf
        no_improve = 0

        for epoch in range(epochs):
            if shuffle:
                indices = np.arange(n_samples)
                np.random.shuffle(indices)
                X = X[indices]
                y = y[indices]

            # ---- Mini-batch training ----
            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch = X[start:end]
                y_batch = y[start:end]

                output = self.forward(X_batch)
                self.backward(y_batch, output)

            # ---- Metrics after epoch ----
            train_output = self.forward(X)
            train_loss = cross_entropy(y, train_output)
            train_acc = accuracy(y, train_output)
            self.train_loss_history.append(train_loss)
            self.train_acc_history.append(train_acc)

            val_loss = None
            if X_val is not None and y_val is not None:
                val_output = self.forward(X_val)
                val_loss = cross_entropy(y_val, val_output)
                val_acc = accuracy(y_val, val_output)
                self.val_loss_history.append(val_loss)
                self.val_acc_history.append(val_acc)

            if val_loss is not None:
                print(f"epoch {epoch + 1:02d}/{epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")
            else:
                print(f"epoch {epoch + 1:02d}/{epochs} - loss: {train_loss:.4f}")

            if patience > 0 and val_loss is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"early stopping at epoch {epoch + 1}")
                        break

    def plot_metrics(self):

        hidden_layers = len(self.weights) - 1  # exclude output layer
        hidden_sizes = [
            w.shape[1] for w in self.weights[:-1]
        ]  # number of neurons in each hidden layer
        hidden_layer_info = f"Hidden Layers: {hidden_layers}, Sizes: {hidden_sizes}"
        learning_info = (
            f", Batch Size: {self.batch_size}, LR: {self.learning_rate}, Optimizer: {self.optimizer}"
        )

        plt.figure(figsize=(12, 5))

        # ---- Loss curve ----
        plt.subplot(1, 2, 1)
        plt.plot(self.train_loss_history, label="Train Loss")
        if len(self.val_loss_history) > 0:
            plt.plot(self.val_loss_history, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss Curve\n{hidden_layer_info}\n{learning_info}")
        plt.legend()

        # ---- Accuracy curve ----
        plt.subplot(1, 2, 2)
        plt.plot(self.train_acc_history, label="Train Accuracy")
        if len(self.val_acc_history) > 0:
            plt.plot(self.val_acc_history, label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy Curve\n{hidden_layer_info}\n{learning_info}")
        plt.legend()

        plt.tight_layout()
        plt.show()
