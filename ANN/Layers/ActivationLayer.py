"""Activation Layer Abstract Class"""

from ANN.Layers.Layer import Layer

from typing import Callable
import numpy.typing as npt
import numpy as np


class ActivationLayer(Layer):
    """

    Attributes:
    input: npt.ArrayLike
        - attribute inherited from base class

    activation: Callable[[np.float64], np.float64]
        - function which calculates value of activation function

    activation_derivative: Callable[[np.float64], np.float64]
        - function which calculates value of derivative of activation function

    Methods:
    feed_forward(self, input: npt.ArrayLike) -> npt.ArrayLike
        - method inherited from base class

    back_propagate(self, gradient: npt.ArrayLike, learning_rate: np.float64)
        - method inherited from base class

    """

    activation: Callable[[np.float64], np.float64]
    activation_derivative: Callable[[np.float64], np.float64]

    def __init__(
        self,
        activation: Callable[[np.float64], np.float64],
        activation_derivative: Callable[[np.float64], np.float64],
    ) -> None:
        self.activation = activation
        self.activation_derivative = activation_derivative

    def feed_forward(self, input: npt.ArrayLike) -> npt.ArrayLike:
        self.input = input
        return self.activation(self.input)

    def back_propagate(
        self, gradient: npt.ArrayLike, learning_rate: np.float64
    ) -> npt.ArrayLike:
        return np.multiply(gradient, self.activation_derivative(self.input))
