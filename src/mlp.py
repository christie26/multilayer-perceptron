import numpy as np


# Activation and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


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

    def train(self, X, y, epochs=10000):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

            if epoch % 1000 == 0:
                loss = np.mean((y - output) ** 2)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
