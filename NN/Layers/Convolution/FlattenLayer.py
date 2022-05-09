"""Flatten Layer Class"""

from typing import Tuple
from NN.Layers.Layer import Layer

import numpy.typing as npt
import numpy as np


class FlattenLayer(Layer):
    """

    Attributes:
    input: npt.ArrayLike
        - attribute inherited from base class

    input_shape: Tuple[np.int64, np.int64, np.int64]
        - tuple with shape of input in flatten layer

    Methods:
    feed_forward(self, input: npt.ArrayLike) -> npt.ArrayLike
        - method inherited from base class

    back_propagate(self, gradient: npt.ArrayLike, learning_rate: np.float64)
        - method inherited from base class

    """

    def __init__(self, input_shape: Tuple[np.int64, np.int64, np.int64]) -> None:
        self.input_shape = input_shape

    def feed_forward(self, input: npt.ArrayLike) -> npt.ArrayLike:
        self.input = input
        return np.reshape(input, (1, -1))

    def back_propagate(
        self, gradient: npt.ArrayLike, learning_rate: np.float64
    ) -> npt.ArrayLike:
        return np.reshape(gradient, self.input_shape)
