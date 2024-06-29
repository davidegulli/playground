import numpy as np


class ActivationFunction:
    def __init__(self, function):
        self.function = function

    def call(self, value):
        if self.function == 'sigmoid':
            return self.sigmoid(value)

        return self.relu(value)

    def call_derivative(self, value):
        if self.function == 'sigmoid':
            return self.sigmoid_derivative(value)

        return self.relu_derivative(value)

    def sigmoid(self, value):
        return 1 / (1 + np.exp(-value))

    def sigmoid_derivative(self, value):
        return value * (1 - value)

    def relu(self, value):
        return np.maximum(0, value)

    def relu_derivative(self, value):
        return np.where(value > 0, 1, 0)