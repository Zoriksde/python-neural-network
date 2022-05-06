"""Swish Activation Layer Class"""

from ANN.Layers.ActivationLayer import ActivationLayer
import numpy as np


class SwishActivationLayer(ActivationLayer):
    def __init__(self, beta: np.float64 = 1.0) -> None:
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        activation = lambda x: x * sigmoid(beta * x)
        activation_derivative = lambda x: self.activation(x) + sigmoid(beta * x) * (1 - self.activation(x))
        super().__init__(activation, activation_derivative)
