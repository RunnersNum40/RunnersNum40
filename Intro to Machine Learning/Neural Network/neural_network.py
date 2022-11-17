import numpy as np
from activation_functions import *

from progress.bar import Bar

class Layer:
    def __init__(self):
        pass

    def forward(self):
        pass

    def backwards(self):
        pass

class ActivationLayer(Layer):
    def __init__(self, activation_function):
        self.activation_function = activation_function

    def forward(self, inputs: list) -> list:
        """Calculate the activation of the layer for a given input.

        Args:
            inputs: List of inputs
        """
        self.input = inputs
        return self.activation_function.activate(inputs)

    def backwards(self, error: list, learning_rate: float = 1) -> list:
        return self.activation_function.derivative(self.input)*error

    def __str__(self) -> str:
        return f"Activation layer with {len(self.input)} inputs"

class DenseLayer(Layer):
    def __init__(self):
        pass

    def forward(self, inputs: list) -> list:
        """Calculate the activation of the layer for a given input.

        Args:
            inputs: List of inputs
        """
        # Add a bias input
        inputs = np.append(np.array(inputs), 1)
        self.input = inputs
        return np.dot(self.weights, inputs)

    def backwards(self, error: list, learning_rate: float = 0.01) -> list:
        """Calculate the the derivative of error with respect to the weights for a given input.

        Args:
            inputs: Input point to optimize
            error: Change in error with respect to the layer output
            learning_rate: Learning rate hyperparameter
        """
        error, inputs = np.array(error), np.expand_dims(self.input, 1)

        # Return the change in error with respect to the inputs
        error_by_inputs = np.dot(error, self.weights)
        # Update the weights down the gradient of the error with respect to the weights
        error_by_weights = np.dot(inputs, error).T

        self.weights -= learning_rate*error_by_weights
        return np.delete(error_by_inputs, -1, axis=1)

    @property
    def num_inputs(self) -> int:
        # Subtract one to remove the bias weights
        return self.weights.shape[1]-1

    @property
    def num_outputs(self) -> int:
        return self.weights.shape[0]

    @classmethod
    def randomized(cls, num_inputs: int, num_outputs: int):
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
        new_layer.weights = np.concatenate((weights, np.ones((num_outputs, 1))), axis=1)
        return new_layer

    @classmethod
    def from_weights(cls, weights: list):
        """Return a new layer with provided weights.

        Args:
            weights: List of weights including column for biases
            activation_function: Activation function used by the layer
        """
        new_layer = cls()
        new_layer.weights = weights
        return new_layer

    def __str__(self) -> str:
        return f"Layer with {self.num_inputs} inputs and {self.num_outputs} outputs"

class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward_propagate(self, inputs: list) -> list:
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

    def backwards_propagate(self, layer_outputs: list, target: list, learning_rate: float = 0.01):
        """Train the network on a single input.

        Args:
            layer_outputs: List of each layer's output in the forward propagate step
            target: Target output
            learning_rate: Learning rate hyperparameter 
        """
        # Calulate the starting error change in loss by change in the output
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
        return np.mean(np.square(target-output))

    def loss_derivative(self, output: list, target: list) -> float:
        """Return change in error with respect to the output of the network.

        Args:
            output: Network output
            target: Target output
        """
        return 2*(output-target)/len(output)

    def train(self, x_train: list, y_train: list, epochs: int = 50, learning_rate: float = 0.01, progress_update: bool = False, progress_bar: bool = False):
        """Train the neural network on a set of training data.

        Args:
            x_train: Training points
            y_train: Training outputs
            epoch: Integer number of epochs to run
            learning_rate: Learning rate hyperparameter
            progress_update: Boolean flag to print updates
            progress_bar: Boolean flag to display a progress bar
        """
        if progress_bar: bar = Bar("Epoch", max=epochs, suffix = r"%(index)d/%(max)d epochs - %(eta)ds eta - %(error)f error")
        samples = len(x_train)
        for epoch in range(epochs):
            error = 0

            for x, y in zip(x_train, y_train):
                layer_outputs = self.forward_propagate(x)
                self.backwards_propagate(layer_outputs, y, learning_rate)

                # Store the error to average across each epoch
                error += self.loss(layer_outputs[-1], y)

            self.error = error/samples
            if progress_update: print(f"Epoch: {epoch+1}/{epochs}, loss: {round(self.error, 5)}")
            if progress_bar:
                bar.error = self.error
                bar.next()
        if progress_bar: bar.finish()

    def classify(self, inputs: list) -> int:
        """Return the index of the highest output neuron from a given input.

        Args:
            inputs: List of input data
        """
        return np.argmax(self.forward_propgate(inputs)[-1])

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

if __name__ == '__main__':
    # XOR training data
    x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
    y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

    # New network
    nn = NeuralNetwork()
    nn.add(DenseLayer.randomized(2, 100))
    nn.add(ActivationLayer(Tanh()))
    nn.add(DenseLayer.randomized(100, 50))
    nn.add(ActivationLayer(Tanh()))
    nn.add(DenseLayer.randomized(50, 1))
    nn.add(ActivationLayer(Tanh()))

    for x, y in zip(x_train, y_train):
        print(x, nn(x), y)

    nn.train(x_train, y_train, 4, 1000, 0.1)

    for x, y in zip(x_train, y_train):
        print(x, nn(x), y)