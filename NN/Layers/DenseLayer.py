"""Dense Layer Class"""

from NN.Layers.Layer import Layer

import numpy.typing as npt
import numpy as np


class DenseLayer(Layer):
    """

    Attributes:
    input: npt.ArrayLike
        - attribute inherited from base class

    weights: npt.ArrayLike
        - numpy array which contains weights of connections in layer

    biases: npt.ArrayLike
        - numpy array which contains biases of connections in layer

    Methods:
    feed_forward(self, input: npt.ArrayLike) -> npt.ArrayLike
        - method inherited from base class

    back_propagate(self, gradient: npt.ArrayLike, learning_rate: np.float64)
        - method inherited from base class

    """

    weights: npt.ArrayLike
    biases: npt.ArrayLike

    def __init__(self, input_neurons: np.int64, output_neurons: np.int64) -> None:
        self.weights = np.random.randn(input_neurons, output_neurons)
        self.biases = np.random.randn(1, output_neurons)

    def feed_forward(self, input: npt.ArrayLike) -> npt.ArrayLike:
        self.input = input
        return np.dot(input, self.weights) + self.biases

    def back_propagate(
        self, gradient: npt.ArrayLike, learning_rate: np.float64
    ) -> npt.ArrayLike:
        weights_gradient = np.dot(self.input.T, gradient)
        input_gradient = np.dot(gradient, self.weights.T)
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * gradient
        return input_gradient
