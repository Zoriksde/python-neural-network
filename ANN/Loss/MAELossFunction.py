"""MAE Loss Function Class"""
from ANN.Loss.LossFunction import LossFunction

import numpy.typing as npt
import numpy as np

class MAELossFunction(LossFunction):
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
        return np.mean(np.abs(actual_output - predicted_output))

    def calculate_loss_derivative(self, actual_output: npt.ArrayLike, predicted_output: npt.ArrayLike) -> np.float64:
        """
        
        This function value of loss derivative
        
        dy/dx = 1 if predicted_output > actual_output else -1
        
        However predicted_output > actual_output return binary matrix f.e [1, 0, 1, 1]
        Let's scale it

        X - Matrix
        B - Binary Matrix

        f(X) = (predicted_output > actual_output)
        f(X): X -> B
        2 * f(x): X -> 2 * B
        2 * f(x): X -> 2 * B - 1
        2 * B - 1 = [1, -1, 1, 1]
        
        """
        
        return 2 * (predicted_output > actual_output) - 1