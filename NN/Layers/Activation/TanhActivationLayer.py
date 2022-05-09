"""Tanh Activation Layer Class"""

from NN.Layers.Activation.ActivationLayer import ActivationLayer
import numpy as np


class TanhActivationLayer(ActivationLayer):
    def __init__(self) -> None:
        activation = lambda x: np.tanh(x)
        activation_derivative = lambda x: 1 - np.tanh(x) ** 2
        super().__init__(activation, activation_derivative)
