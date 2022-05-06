"""Example set given by sklearn - make moons """

from ANN.Loss.MSELossFunction import MSELossFunction
from ANN.Network.NeuralNetwork import NeuralNetwork
from ANN.Layers.DenseLayer import DenseLayer
from ANN.Layers.SigmoidActivationLayer import SigmoidActivationLayer
from ANN.Layers.SwishActivationLayer import SwishActivationLayer

import numpy as np
from sklearn.datasets import make_moons

X, y = make_moons(500, noise=0.1)
X = np.reshape(X, (500, 1, 2))
y = np.reshape(y, (500, 1, 1))

loss_function = MSELossFunction()

network = NeuralNetwork(loss_function)
network.add_layer(DenseLayer(2, 4))
network.add_layer(SwishActivationLayer(beta=0.8))
network.add_layer(DenseLayer(4, 4))
network.add_layer(SwishActivationLayer(beta=0.8))
network.add_layer(DenseLayer(4, 4))
network.add_layer(SwishActivationLayer(beta=0.8))
network.add_layer(DenseLayer(4, 1))
network.add_layer(SigmoidActivationLayer())

network.train(X, y, 100, 0.1)
