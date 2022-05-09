"""Layer Abstract Class"""

from abc import ABC, abstractmethod
import numpy.typing as npt
import numpy as np


class Layer(ABC):
    """

    Attributes:
    input: npt.ArrayLike
        - numpy array which contains input values passed to layer

    Methods:
    feed_forward(self, input: npt.ArrayLike) -> npt.ArrayLike
        - method which passes current input forward to next layer

    back_propagate(self, gradient: npt.ArrayLike, learning_rate: np.float64)
        - method which passes current gradient backward to previous layer

    """

    input: npt.ArrayLike

    @abstractmethod
    def feed_forward(self, input: npt.ArrayLike) -> npt.ArrayLike:
        pass

    @abstractmethod
    def back_propagate(
        self, gradient: npt.ArrayLike, learning_rate: np.float64
    ) -> npt.ArrayLike:
        pass
