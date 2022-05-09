"""BCE Loss Function Class"""
from NN.Loss.LossFunction import LossFunction

import numpy.typing as npt
import numpy as np

class BCELossFunction(LossFunction):
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
        """
        
        Note:
        Binary Cross Entropy Loss function is special case of cross entropy loss function, its
        assuming we have only two outputs of our neural network either F or S where F and S are different
        categories.

        Here we are assuming our F = 0, S = 1 or otherwise.
        Then if our actual output is 1 the term (1 - actual_output) * np.log(1 - predicted_output)
        is cancelled out, while if our actual output is 0 then the term 
        actual_output * np.log(precited_output) also cancels out.

        Thus, the binary cross entropy can be simplified in special way to cross entropy.
        
        Also, it's well to mention that logarithmic function which is used here doesn't accept
        negative values, thus our activation function in last layer should return non-negative
        results.

        f.e sigmoid or softmax

        """

        return -np.mean(actual_output * np.log(predicted_output) + (1 - actual_output) * np.log(1 - predicted_output))

    def calculate_loss_derivative(self, actual_output: npt.ArrayLike, predicted_output: npt.ArrayLike) -> np.float64: 
        return ((1 - actual_output) / (1 - predicted_output) - actual_output / predicted_output) / np.size(actual_output)