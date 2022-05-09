"""Main file"""

from NN.Layers.Convolution.ConvolutionLayer import ConvolutionLayer
from NN.Layers.Convolution.FlattenLayer import FlattenLayer
from NN.Layers.DenseLayer import DenseLayer
from NN.Loss.BCELossFunction import BCELossFunction
from NN.Loss.MAELossFunction import MAELossFunction
from NN.Loss.MSELossFunction import MSELossFunction
from NN.Network.NeuralNetwork import NeuralNetwork
from NN.Layers.Activation.ReLUActivationLayer import ReLUActivationLayer
from NN.Layers.Activation.SigmoidActivationLayer import SigmoidActivationLayer
from NN.Layers.Activation.SwishActivationLayer import SwishActivationLayer
from NN.Layers.Activation.TanhActivationLayer import TanhActivationLayer

import numpy as np
from sklearn.datasets import load_digits

"""Ex. of Convolutional Neural Network with digits dataset from sklearn"""

digits = load_digits(n_class=2)
X = np.reshape(digits.images[:300], (300, 1, 8, 8))
y = np.reshape(digits.target[:300], (300, 1, 1))
X = X.astype(dtype=np.double)

"""

Note: 

This project is simple implementation of Neural Network, notice assumptions listed below:

1) There is no any pooling layer implemented yet.
2) For large value of x, np.exp(-x) can cause overflow error, 
there are alternatives of np.exp(x) however it's not implemented here.
3) ReLU Activation function is commonly used in CNNs with its alternatives f.e PReLU,
please notice that for any positive value of x ReLU(x) = x, so it can cause overflow error
when passed to f.e sigmoid function
4) Project is created with main aim of helping others to understand basis of Neural Networks,
it is not optimized f.e no backpropagation optimizer like Adam and so on...

Enjoy!

"""

cnn_loss_function = MAELossFunction()
cnn_network = NeuralNetwork(cnn_loss_function)
cnn_network.add_layer(ConvolutionLayer((1, 8), 2, 5))
cnn_network.add_layer(ReLUActivationLayer())
cnn_network.add_layer(FlattenLayer((5, 7, 7)))
cnn_network.add_layer(DenseLayer(5 * 7 * 7, 100))
cnn_network.add_layer(SigmoidActivationLayer())
cnn_network.add_layer(DenseLayer(100, 1))
cnn_network.add_layer(SigmoidActivationLayer())
cnn_network.train(X, y, epochs=30)

cnn_network.predict(X[0]) # actual_value = 0
cnn_network.predict(X[1]) # actual_value = 1
