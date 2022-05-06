"""MSE Loss Function Class"""
from ANN.Loss.LossFunction import LossFunction

import numpy.typing as npt
import numpy as np

class MSELossFunction(LossFunction):
    """

    Methods:
    calculate_loss(self, actual_output: npt.ArrayLike, predicted_output: npt.ArrayLike) -> npt.float64
        - method inherited from base class

    calculate_loss_derivative(self, actual_output: npt.ArrayLike, predicted_output: npt.ArrayLike) -> npt.float64
        - method inherited from base class

    """

    def __init__(self) -> None:
        pass

    def calculate_loss(self, actual_output: npt.ArrayLike, predicted_output: npt.ArrayLike) -> np.float64:
        return np.mean(np.power(actual_output - predicted_output, 2))

    def calculate_loss_derivative(self, actual_output: npt.ArrayLike, predicted_output: npt.ArrayLike) -> np.float64:
        return 2 * (predicted_output - actual_output) / np.size(actual_output)