"""Sigmoid Activation Layer Class"""

from NN.Layers.Activation.ActivationLayer import ActivationLayer
import numpy as np


class SigmoidActivationLayer(ActivationLayer):
    def __init__(self) -> None:
        activation = lambda x: 1 / (1 + np.exp(-x))
        activation_derivative = lambda x: (1 - activation(x)) * activation(x)
        super().__init__(activation, activation_derivative)
