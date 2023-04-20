import numpy as np

from neural_network import *
from activation_functions import *

from keras.datasets import mnist
from keras.utils import np_utils


# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# training data : 60000 samples
# reshape and normalize input data
x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255
# encode output which is a number in range [0,9] into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = np_utils.to_categorical(y_train)

# same for test data : 10000 samples
x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = np_utils.to_categorical(y_test)

nn = NeuralNetwork()
nn.add(DenseLayer.randomized(28*28, 100))
nn.add(ActivationLayer(Tanh()))
nn.add(DenseLayer.randomized(100, 50))
nn.add(ActivationLayer(Tanh()))
nn.add(DenseLayer.randomized(50, 10))
nn.add(ActivationLayer(Tanh()))

y_train, y_test = np.expand_dims(y_train, 1), np.expand_dims(y_test, 1)

nn.train(x_train[:2000], y_train[:2000], epochs=1000, learning_rate=0.05, progress_bar=True)

results = np.array([nn.classify(x) == np.argmax(y) for x, y in zip(x_test, y_test)])
print(f"{round(np.mean(results)*100)}% accurate on test data")