"""Neural Network Class"""

from NN.Layers.Layer import Layer
from NN.Loss.LossFunction import LossFunction

from typing import List
import numpy.typing as npt
import numpy as np

class NeuralNetwork:
    """

    Attributes:
    layers: List[Layer]
        - list of layers to proceed during training process

    loss_function: LossFunction
        - loss function which calculates current loss

    Methods:
    add_layer(self, layer: Layer) -> None:
        - method which appends new layer to proceed

    train(self, inputs: npt.ArrayLike, actual_outputs: npt.ArrayLike, 
        epochs: np.int64 = 100, learning_rate: np.float64 = 0.1) -> None:
        - method which trains model to predict with highest accuracy
    
    predict(self, input: npt.ArrayLike) -> None:
        - method which predicts output for given input

    """

    layers: List[Layer]
    loss_function: LossFunction

    def __init__(self, loss_function: LossFunction) -> None:
        self.layers = []
        self.loss_function = loss_function
        pass

    def add_layer(self, layer: Layer) -> None:
        self.layers.append(layer)

    def train(
        self,
        inputs: npt.ArrayLike,
        actual_outputs: npt.ArrayLike,
        epochs: np.int64 = 100,
        learning_rate: np.float64 = 0.1,
    ) -> None:

        for epoch in range(epochs):
            loss_value = 0

            for input, output in zip(inputs, actual_outputs):
                current_input = input

                for layer in self.layers:
                    current_input = layer.feed_forward(current_input)

                loss_value += self.loss_function.calculate_loss(output, current_input)
                gradient = self.loss_function.calculate_loss_derivative(
                    output, current_input
                )

                for layer in reversed(self.layers):
                    gradient = layer.back_propagate(gradient, learning_rate)

            print(
                f"Training... [{epoch + 1} / {epochs} epoch] with {loss_value} value of loss"
            )
            
    def predict(self, input: npt.ArrayLike) -> None:
        current_input = input
        
        for layer in self.layers:
            current_input = layer.feed_forward(current_input)
        
        print(f'Prediction: {current_input}')