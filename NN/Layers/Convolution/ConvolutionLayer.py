"""Convolution Layer Class"""

from NN.Layers.Layer import Layer

from typing import Tuple
from scipy import signal

import numpy.typing as npt
import numpy as np

class ConvolutionLayer(Layer):
    """

    Attributes:
    input: npt.ArrayLike
        - attribute inherited from base class
    
    kernels: npt.ArrayLike
        - numpy array which contains each kernel used for either corellation or convolution
    
    biases: npt.ArrayLike
        - numpy array which contains biases of kernels in layer

    depth: np.int64
        - attribute which stores depth of convolution layer

    input_depth: np.int64
        - attribute which stores depth of input

    kernels_shape: Tuple[np.int64, np.int64, np.int64, np.int64]
        - tuple with shape of kernels in convolution layer

    input_shape: Tuple[np.int64, np.int64, np.int64]
        - tuple with shape of input in convolution layer

    Methods:
    feed_forward(self, input: npt.ArrayLike) -> npt.ArrayLike
        - method inherited from base class

    back_propagate(self, gradient: npt.ArrayLike, learning_rate: np.float64)
        - method inherited from base class

    Note:
    cross correlation is element wise matrix multiplication between input and kernel matrices,
    while convolution operation is element wise matrix multiplication between input and rotated
    kernel matrices.

    valid cross correlation is generating downsized feature map, while same cross correlation is
    generating same size feature map.

    feature_map_size = ((input_size - kernel_size + 2 * padding_size) / stride) + 1

    let's assume that we want to perform valid cross correlation, thus padding_size is 0, and
    let's assume that stride is 1, thus:
    feature_map_size = input_size - kernel_size + 1

    What's more shapes are defined as below:
    (x, y, z, v) - is four dimensional tuple where x is depth of whole layer, y is depth
    of current kernel, z and v determine matrix shape

    (x, y, z) - is three dimensional tuple where x is depth of whole input, 
    y and z determine matrix shape
    
    """

    kernels: npt.ArrayLike
    biases: npt.ArrayLike
    depth: np.int64
    input_depth: np.int64
    kernels_shape: Tuple[np.int64, np.int64, np.int64, np.int64]
    input_shape: Tuple[np.int64, np.int64, np.int64]

    def __init__(self, input_shape: Tuple[np.int64, np.int64], kernel_size: np.int64, depth: np.int64 = 1) -> None:
        """
        
        Note:
        Let's assume that single input matrix is square matrix, also notice that depth hyperparameter
        is equivalent to number of kernels we want to use.
        
        """

        input_depth, input_size = input_shape
        feature_map_shape = (depth, input_size - kernel_size + 1, input_size - kernel_size + 1)

        self.depth = depth
        self.input_depth = input_depth

        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.input_shape = (input_depth, input_size, input_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*feature_map_shape)

    def feed_forward(self, input: npt.ArrayLike) -> npt.ArrayLike:
        self.input = input
        output = np.copy(self.biases)

        for i in range(self.depth):
            for j in range(self.input_depth):
                output[i] += signal.correlate2d(input[j], self.kernels[i, j], "valid")

        return output

    def back_propagate(self, gradient: npt.ArrayLike, learning_rate: np.float64) -> npt.ArrayLike:
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(gradient[i], self.kernels[i, j], "full")
        
        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * gradient
        return input_gradient