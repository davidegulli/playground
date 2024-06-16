import numpy as np
from graphs import plot_descendent_gradient

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, debug=False, activation_function='sigmoid'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation_function = activation_function
        self.debug = debug

        self.weights_hidden_input = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2.0 / (self.input_size + self.hidden_size))
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2.0 / (self.output_size + self.hidden_size))
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))

        # Initialize velocity terms for momentum
        self.velocity_hidden_input = np.zeros_like(self.weights_hidden_input)
        self.velocity_hidden_output = np.zeros_like(self.weights_hidden_output)
        self.velocity_bias_hidden = np.zeros_like(self.bias_hidden)
        self.velocity_bias_output = np.zeros_like(self.bias_output)

        self.hidden_output = 0
        self.predicted_output = 0


    def activation(self, value):
        if self.activation_function == 'sigmoid':
            return self.sigmoid(value)

        return self.relu(value)

    def activation_derivative(self, value):
        if (self.activation_function == 'sigmoid'):
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

    def predict(self, data):
        # Hidden Layer
        hidden_activation = np.dot(data, self.weights_hidden_input) + self.bias_hidden
        self.hidden_output = self.activation(hidden_activation)

        # Output Layer
        output_activation = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.predicted_output = self.activation(output_activation)

        return self.predicted_output

    def back_propagation(self, data, targets, learning_rate, momentum):
        # Output Layer
        output_error = targets - self.predicted_output
        output_delta = output_error * self.activation_derivative(self.predicted_output)

        # Update velocities and weights with momentum
        self.velocity_hidden_output = momentum * self.velocity_hidden_output + np.dot(self.hidden_output.T, output_delta) * learning_rate
        self.weights_hidden_output += self.velocity_hidden_output
        self.velocity_bias_output = momentum * self.velocity_bias_output + np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.bias_output += self.velocity_bias_output

        # Hidden Layer
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.activation_derivative(self.hidden_output)

        # Update velocities and weights with momentum
        self.velocity_hidden_input = momentum * self.velocity_hidden_input + np.dot(data.T, hidden_delta) * learning_rate
        self.weights_hidden_input += self.velocity_hidden_input
        self.velocity_bias_hidden = momentum * self.velocity_bias_hidden + np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate
        self.bias_hidden += self.velocity_bias_hidden

    def train(self, data, targets, epochs, learning_rate=0.00001, momentum=0.9,):
        y_losses = []
        x_epochs = []

        for epoch in range(epochs):
            output = self.predict(data)
            self.back_propagation(data, targets, learning_rate, momentum)
            if epoch % (epochs / 10) == 0:
                x_epochs.append(epoch)
                loss = np.mean(np.square(targets - output))
                y_losses.append(loss)
                if self.debug:
                    print(f"Epoch {epoch}, Loss: {loss}")

        plot_descendent_gradient(x_epochs, y_losses)
        return np.min(y_losses)
