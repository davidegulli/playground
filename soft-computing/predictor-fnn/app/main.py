import numpy as np
from neural_network import NeuralNetwork
from dataset import Dataset
from graphs import plot_losses_graph

#dataset = Dataset(file_path="../files/wizmir.txt", columns_number=9)
dataset = Dataset(file_path="../files/ele-2.txt", columns_number=4)
#dataset = Dataset(file_path="../files/house_prices.txt", columns_number=6)

training_data, test_data, training_targets, test_targets, = dataset.load()

x_neurons = []
y_training_losses = []
y_test_losses = []

for neurons in range(5, 51, 5):
    print('--------------------------------------------------')
    print(f"Number of neurons in the hidden layers: {neurons}")

    x_neurons.append(neurons)

    # Training the model
    nn = NeuralNetwork(
        input_size=dataset.columns_number,
        hidden_size=neurons,
        output_size=1
    )

    training_loss = nn.train(
        training_data,
        training_targets.reshape(len(training_targets), 1),
        epochs=10000,
        learning_rate=0.00001,
        momentum=0.9
    )

    y_training_losses.append(training_loss)
    print(f"Training Loss: {training_loss}")

    # Testing the model
    outputs = nn.predict(test_data)
    output_loss = np.mean(np.square(test_targets - outputs.reshape(-1)))
    y_test_losses.append(output_loss)
    print(f"Testing Loss: {output_loss}")

plot_losses_graph(x_neurons, y_training_losses, y_test_losses)