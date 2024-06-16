import matplotlib.pyplot as plt


def plot_losses_graph(x_neurons, y_training_losses, y_test_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(x_neurons, y_training_losses, marker='o', label='Errore quadratico Training')
    plt.plot(x_neurons, y_test_losses, marker='o', label='Errore quadratico Testing')

    plt.xlabel('Numero di Neuroni')
    plt.ylabel('Errori')
    plt.title('Errore quadratico su Training e Testing set rispetto al numero di Neuroni')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_descendent_gradient(x_epochs, y_training_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(x_epochs, y_training_losses, marker='o', label='Loss')

    plt.xlabel('Numero di Epoche')
    plt.ylabel('Loss')
    plt.title('Loss per epoche')
    plt.legend()
    plt.grid(True)
    plt.show()