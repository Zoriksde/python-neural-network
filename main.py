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

cnn_loss_function = BCELossFunction()
cnn_network = NeuralNetwork(cnn_loss_function)
cnn_network.add_layer(ConvolutionLayer((1, 8), 2, 5))
cnn_network.add_layer(SigmoidActivationLayer())
cnn_network.add_layer(FlattenLayer((5, 7, 7)))
cnn_network.add_layer(DenseLayer(5 * 7 * 7, 100))
cnn_network.add_layer(SigmoidActivationLayer())
cnn_network.add_layer(DenseLayer(100, 1))
cnn_network.add_layer(SigmoidActivationLayer())
cnn_network.train(X, y, epochs=30)

cnn_network.predict(X[0]) # 0
cnn_network.predict(X[1]) # 1
