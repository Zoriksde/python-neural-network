# Neural Network

Neural Network project that helps others to understand basic ideas behind neural networks.
The only requirement of understanding concepts of this project is basic knowledge of linear algebra and calculus.
There are several activation functions and loss functions to use.
Project is written in Python.

The structure of neural network is divided into layers of several different types:

- Dense Layer - which creates synapses between each node from input to each node from output.
- Activation Layer - which calculates value of activation function for given input.
- Convolution Layer - which performs correlation and convolution operations on images.
- Flatten Layer - which reshapes given input into one dimension output.

Project is written using OOP Principles and different Design Patterns, which makes it easy to maintain, extend, fix and change.
Each activation function is implemented as single class which inherits from abstract one (Strategy Design Pattern).
The same concept is used in loss functions.

Ex. of activation layer:

```python
class ActivationLayer(Layer):

    activation: Callable[[np.float64], np.float64]
    activation_derivative: Callable[[np.float64], np.float64]

    def __init__(
        self,
        activation: Callable[[np.float64], np.float64],
        activation_derivative: Callable[[np.float64], np.float64],
    ) -> None:
        self.activation = activation
        self.activation_derivative = activation_derivative

    def feed_forward(self, input: npt.ArrayLike) -> npt.ArrayLike:
        self.input = input
        return self.activation(self.input)

    def back_propagate(
        self, gradient: npt.ArrayLike, learning_rate: np.float64
    ) -> npt.ArrayLike:
        return np.multiply(gradient, self.activation_derivative(self.input))
```

There is also Neural Network class which is responsible for training model and predicting values.
There is no any optimizers implemented.

Ex. of training model

```python
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
```

Activation functions that are covered here:

| Function | Vanishing Gradient Issue | Formula                            | Derivative Formula                      |
| -------- | ------------------------ | ---------------------------------- | --------------------------------------- |
| ReLU     | No                       | f(x) = max(0, x)                   | d(x) = f(x) \* 1 / x                    |
| Sigmoid  | Yes                      | f(x) = 1 / (1 + e^-x)              | d(x) = f(x) \* (1 - f(x))               |
| Tanh     | Yes                      | f(x) = (e^x - e^-x) / (e^x + e^-x) | d(x) = 1 - f^2(x)                       |
| Swish    | No                       | f(x) = x \* sigmoid(Bx)            | d(x) = f(x) + sigmoid(Bx) \* (1 - f(x)) |

Loss functions that are covered here:

- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- BCE (Binary Cross Entropy)

I do recommend this [article](https://arxiv.org/pdf/1811.03378.pdf) to read about activation functions with some chart visualization.

There is implemented basic CNN on ```main.py```.

Project is written with main aim to help anyone who is interested in these topics. Enjoy playing around!
