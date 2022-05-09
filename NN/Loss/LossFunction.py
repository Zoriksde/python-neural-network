"""Loss Function Abstract Class"""

from abc import ABC, abstractmethod
import numpy.typing as npt
import numpy as np


class LossFunction(ABC):
    """

    Methods:
    calculate_loss(self, actual_output: npt.ArrayLike, predicted_output: npt.ArrayLike) -> npt.float64
        - function which calculates loss

    calculate_loss_derivative(self, actual_output: npt.ArrayLike, predicted_output: npt.ArrayLike) -> npt.float64
        - function which calculates loss derivative

    """
    
    def __init__(self) -> None:
        pass

    @abstractmethod
    def calculate_loss(
        self, actual_output: npt.ArrayLike, predicted_output: npt.ArrayLike
    ) -> np.float64:
        pass

    @abstractmethod
    def calculate_loss_derivative(
        self, actual_output: npt.ArrayLike, predicted_output: npt.ArrayLike
    ) -> np.float64:
        pass
