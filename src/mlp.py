import numpy as np

# Activation and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


class MLP:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, learning_rate=0.1):
        self.learning_rate = learning_rate

        # Use initialize_weights function
        layer_sizes = [input_size, hidden_size1, hidden_size2, output_size]
        weights, biases = self.initialize_weights(layer_sizes)

        # Assign weights and biases to layer connections
        self.weights_input_hidden1 = weights[0].T
        self.bias_hidden1 = biases[0].T

        self.weights_hidden1_hidden2 = weights[1].T
        self.bias_hidden2 = biases[1].T

        self.weights_hidden2_output = weights[2].T
        self.bias_output = biases[2].T

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

            b = np.zeros((fan_out, 1))  # biases usually start at 0

            weights.append(W)
            biases.append(b)

        return weights, biases

    def forward(self, X):
        # Input -> Hidden Layer 1
        self.hidden_input1 = np.dot(X, self.weights_input_hidden1) + self.bias_hidden1
        self.hidden_output1 = sigmoid(self.hidden_input1)

        # Hidden Layer 1 -> Hidden Layer 2
        self.hidden_input2 = np.dot(self.hidden_output1, self.weights_hidden1_hidden2) + self.bias_hidden2
        self.hidden_output2 = sigmoid(self.hidden_input2)

        # Hidden Layer 2 -> Output
        self.final_input = np.dot(self.hidden_output2, self.weights_hidden2_output) + self.bias_output
        self.final_output = sigmoid(self.final_input)

        return self.final_output

    def backward(self, X, y, output):
        # Output error and delta
        error_output = y - output
        d_output = error_output * sigmoid_derivative(output)

        # Hidden layer 2 error and delta
        error_hidden2 = d_output.dot(self.weights_hidden2_output.T)
        d_hidden2 = error_hidden2 * sigmoid_derivative(self.hidden_output2)

        # Hidden layer 1 error and delta
        error_hidden1 = d_hidden2.dot(self.weights_hidden1_hidden2.T)
        d_hidden1 = error_hidden1 * sigmoid_derivative(self.hidden_output1)

        # Update weights and biases
        self.weights_hidden2_output += self.hidden_output2.T.dot(d_output) * self.learning_rate
        self.bias_output += np.sum(d_output, axis=0, keepdims=True) * self.learning_rate

        self.weights_hidden1_hidden2 += self.hidden_output1.T.dot(d_hidden2) * self.learning_rate
        self.bias_hidden2 += np.sum(d_hidden2, axis=0, keepdims=True) * self.learning_rate

        self.weights_input_hidden1 += X.T.dot(d_hidden1) * self.learning_rate
        self.bias_hidden1 += np.sum(d_hidden1, axis=0, keepdims=True) * self.learning_rate

    def train(self, X, y, epochs=10000):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            if epoch % 1000 == 0:
                loss = np.mean((y - output) ** 2)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
