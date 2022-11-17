import numpy as np

class Preceptron:
    """A single neuron capable of linear classification"""
    def __init__(self, weights: list):
        self.weights = np.array(weights)
        # Create an extra weight to use as the bias
        self.weights = np.append(self.weights, 0)

    def activate(self, inputs: list):
        """Activate the preceptron on an input array"""
        # Create an extra feature to use as the bias
        inputs = np.append(np.array(inputs), 1)
        return -1 if np.sum(np.multiply(self.weights, inputs)) < 0 else 1

    def train(self, training_inputs: list, training_lables: list, learning_rate: float = 1.0, max_epochs: int = 100):
        """Train the preceptron until the weights don't update.

        Args:
            training_inputs: Array of training points
            training_lables: Array of classifications
            learning_rate: Floating point learning rate
            max_epochs: Integer number of epochs to halt learning at
        """
        old_weights = None
        epochs = 0
        # Until the weights are not updated in a step
        while not np.array_equal(self.weights, old_weights) and epochs < max_epochs:
            old_weights = np.copy(self.weights)
            epochs += 1
            for x, y in zip(training_inputs, training_lables):
                x = np.array(x)
                # If the classification is incorrect
                if self.activate(x)*y <= 0:
                    # Create an extra feature to use as the bias
                    x = np.append(x, 1)
                    # Updated the weights
                    self.weights += learning_rate*y*x

        return epochs

    def evaluate(self, test_inputs: list, test_lables: list):
        """Return the number of points incorrectly classified.

        Args:
            test_inputs: Array of test points
            test_lables: Array of classifications
        """
        errors = []
        for x, y in zip(test_inputs, test_lables):
            x = np.array(x)
            errors.append(self.activate(x)*y <= 0)
        return sum(errors)/len(errors)

    @classmethod
    def randomized(cls, num_inputs: int):
        """Return a preceptron with random weights.

        Args:
            num_inputs: Integer number of inputs
        """
        return cls(1-2*np.random.rand(num_inputs))

    @property
    def num_inputs(self):
        # Subtract one to remove the bias
        return len(self.weights)-1


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    input_shape = 2
    points = 100

    # Generate random data
    center = np.random.randint(1, 6, size=(input_shape,))
    X1 = center+np.random.normal(0, 2, (points//2, input_shape))
    X2 = -center+np.random.normal(0, 2, (points//2, input_shape))
    X = np.concatenate((X1, X2))

    Y1 = np.ones(points//2)
    Y2 = -Y1
    Y = np.concatenate((Y1, Y2))

    # Split the date for testing
    x_train, x_test, y_train, y_test = train_test_split(X, Y)

    preceptron = Preceptron.randomized(input_shape)

    Y_eval_untrained = np.array([preceptron.activate(x) for x in X])

    # Train the preceptron on the data
    print("Error before training:", f"{round(preceptron.evaluate(x_test, y_test)*100, 1)}%")
    print("Trained over", preceptron.train(x_train, y_train, learning_rate=0.1, max_epochs=200), "epochs")
    print("Error after training:", f"{round(preceptron.evaluate(x_test, y_test)*100, 1)}%")

    Y_eval_trained = np.array([preceptron.activate(x) for x in X])

    # If the data can be plotted easily
    if input_shape == 2:
        # Create and format subplots to display results
        fig, axs = plt.subplots(1, 3)
        for ax in axs:
            ax.set(adjustable="box", aspect="equal")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        # Plot the data
        axs[0].set_title("Data", loc="left")
        axs[0].scatter(X[Y==1][:,0], X[Y==1][:,1], color="red")
        axs[0].scatter(X[Y==-1][:,0], X[Y==-1][:,1], color="blue")
        # Plot the prediction of the preceptron before training
        axs[1].set_title("Untrained", loc="left")
        axs[1].scatter(X[Y_eval_untrained==1][:,0], X[Y_eval_untrained==1][:,1], color="red")
        axs[1].scatter(X[Y_eval_untrained==-1][:,0], X[Y_eval_untrained==-1][:,1], color="blue")
        # Plot the prediction of the preceptron after training
        axs[2].set_title("Trained", loc="left")
        axs[2].scatter(X[Y_eval_trained==1][:,0], X[Y_eval_trained==1][:,1], color="red")
        axs[2].scatter(X[Y_eval_trained==-1][:,0], X[Y_eval_trained==-1][:,1], color="blue")

        plt.show()