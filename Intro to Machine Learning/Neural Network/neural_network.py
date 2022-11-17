import numpy as np
from activation_functions import *

class Layer:
    def __init__(self):
        pass

    def forward(self, inputs: list) -> list:
        """Calculate the activation of the layer for a given input.

        Args:
            inputs: List of inputs
        """
        # Add a bias input
        inputs = np.append(np.array(inputs), 1)

        return self.activation_function.activate(np.dot(self.weights, inputs))

    def backwards(self, inputs: list, error: list, learning_rate: float = 0.01) -> list:
        """Calculate the the derivative of error with respect to the weights for a given input.

        Args:
            inputs: Input point to optimize
            error: Change in error with respect to the layer output
            learning_rate: Learning rate hyperparameter
        """
        error, inputs = np.array(error), np.array(inputs)
        # inputs = np.append(inputs, 1)
        # Calculate the change in error with respect to the activation function
        error = np.multiply(self.activation_function.derivative(inputs), error)

        # Update the weights down the gradient of the error with respect to the weights
        error_by_weights = learning_rate*np.dot(inputs.T, error)
        # Return the change in error with respect to the inputs
        print(error)
        error_by_inputs = np.dot(error, self.weights.T)

        self.weights -= error_by_weights
        return error_by_inputs

    @property
    def num_inputs(self) -> int:
        # Subtract one to remove the bias weights
        return self.weights.shape[1]-1

    @property
    def num_outputs(self) -> int:
        return self.weights.shape[0]

    @classmethod
    def randomized(cls, num_inputs: int, num_outputs: int, activation_function=None):
        """Return a new layer with random weights.

        Args:
            num_inputs: Integer number of inputs to the layer
            num_outputs: Integer number of outputs of the layer
            activation_function: Activation function used by the layer
        """
        new_layer = cls()
        # Intialize an array of random weights
        weights = 1-2*np.random.rand(num_outputs, num_inputs)
        # Create the bias weights
        new_layer.weights = np.concatenate((weights, np.zeros((num_outputs, 1))), axis=1)
        # Choose the identity if no activation function provided
        new_layer.activation_function = activation_function if activation_function is not None else identity
        return new_layer

    @classmethod
    def from_weights(cls, weights: list, activation_function=None):
        """Return a new layer with provided weights.

        Args:
            weights: List of weights including column for biases
            activation_function: Activation function used by the layer
        """
        new_layer = cls()
        new_layer.weights = weights
        # Choose the identity if no activation function provided
        new_layer.activation_function = activation_function if activation_function is not None else identity
        return new_layer

    def __str__(self) -> str:
        return f"Layer with {self.num_inputs} inputs and {self.num_outputs} outputs"

class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def forward_propgate(self, inputs: list) -> list:
        """Return the output layers from a given output.

        Args:
            inputs: List of input data
        """
        # Add the input as the output of the first layer
        layer_outputs = [inputs]
        for layer in self.layers:
            # Activate each layer on the output of the previous layer
            layer_outputs.append(layer.forward(layer_outputs[-1]))

        return layer_outputs

    def backwards_propogate(self, layer_outputs: list, target: list, learning_rate: float = 0.01):
        """Train the network on a single input.

        Args:
            layer_outputs: List of each layer's output in the forward propogation step
            target: Target output
            learning_rate: Learning rate hyperparameter 
        """
        # Calulate the starting error
        error = self.loss_derivative(layer_outputs[-1], target)
        # Loop through the layers in reverse order
        for layer in self.layers[::-1]:
            error = layer.backwards(error, learning_rate)

    def loss(self, output: list, target: list) -> float:
        """Return the squared sum loss of the network on a single input point.

        Args:
            output: Network output
            target: Target output
        """
        return np.mean(np.square(lables-inputs))

    def loss_derivative(self, output: list, target: list) -> float:
        """Return change in error with respect to the output of the network.

        Args:
            output: Network output
            target: Target output
        """
        return 2*(output-target)/len(output)

    def train(self, x_train: list, y_train: list, epochs: int = 50, learning_rate: float = 0.01):
        samples = len(x_train)
        for epoch in range(epochs):
            error = 0
            for n, (x, y) in enumerate(zip(x_train, y_train)):
                layer_outputs = self.forward_propgate(x)
                self.backwards_propogate(layer_outputs, y, learning_rate)

                # Store the error to average across each epoch
                error += self.loss(layer_outputs[-1], y)

            error /= samples
            print(f"Epoch:{epoch+1} of {epochs}, error:{error}")

    def classify(self, inputs: list) -> int:
        """Return the index of the highest output neuron from a given input.

        Args:
            inputs: List of input data
        """
        return np.argmax(self.activate(inputs)[-1])

    def __call__(self, inputs: list) -> list:
        return self.forward_propgate(inputs)[-1]

    @property
    def num_inputs(self) -> int:
        return self.layers[0].num_inputs

    @property
    def num_outputs(self) -> int:
        return self.layers[-1].num_outputs

    @property
    def shape(self) -> list:
        """Return the size of each layer in the network."""
        return [self.num_inputs]+[layer.num_outputs for layer in self.layers]

    @classmethod
    def randomized(cls, layers: list, activation_functions = None, layer_cls = Layer):
        """Return a new neural network with random weights.

        Args:
            layers: Output size of each layer including the input layer
            activation_functions: List of activation_functions used by the layers excluding the input layer
            layer_cls: Layer class to use
        """
        new_network = cls()
        # Choose the identity if no activation functions provided
        if activation_functions is None:
            activation_functions = [None for _ in layers]
        # Create layers with random weights
        for inputs, outputs, activation_function in zip(layers, layers[1:], activation_functions):
            new_network.layers.append(layer_cls.randomized(inputs, outputs, activation_function))

        return new_network

    @classmethod
    def from_weights(cls, layer_weights: list, activation_functions = None, layer_cls = Layer):
        """Return a new neural network with provided weights.

        Args:
            layer_weights: List of weight matrixes used by each layer including bias columns
            activation_function: List of activation_functions used by the layers excluding the input layer
            layer_cls: Layer class to use
        """
        new_network = cls()
        # Choose the identity if no activation functions provided
        if activation_functions is None:
            activation_functions = [None for _ in layers]
        # Create layers with the provided weights
        for weights, activation_function in zip(layer_weights, activation_functions):
            new_network.layers.append(layer_cls.from_weights(weights, activation_function))


if __name__ == '__main__':
    # XOR training data
    x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
    y_train = np.array([[[0, 1]], [[1, 0]], [[1, 0]], [[0, 1]]])

    # New network
    nn = NeuralNetwork.randomized([2, 3, 2], [Tanh(), Tanh()])
    nn.train(x_train, y_train, 1000, 0.1)

