"""ReLU Activation Layer Class"""

from ANN.Layers.ActivationLayer import ActivationLayer
import numpy as np


class ReLUActivationLayer(ActivationLayer):
    def __init__(self) -> None:
        activation = lambda x: np.maximum(0, x)
        activation_derivative = lambda x: np.maximum(0, x) * 1 / x
        super().__init__(activation, activation_derivative)
