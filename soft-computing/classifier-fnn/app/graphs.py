import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_binary_classification_outcome(average_losses, average_accuracies, average_recalls, dataset_names):
    bar_width = 0.25
    plt.subplots(figsize=(14, 10))

    # Set position of bar on X axis
    br1 = np.arange(len(average_losses))
    br2 = [x + bar_width for x in br1]
    br3 = [x + bar_width for x in br2]

    plt.bar(br1, average_losses, color='#ea5545', width=bar_width, edgecolor='grey', label='Loss')
    plt.bar(br2, average_accuracies, color='#bdcf32', width=bar_width, edgecolor='grey', label='Accuracy')
    plt.bar(br3, average_recalls, color='#ef9b20', width=bar_width, edgecolor='grey', label='Recall')

    plt.xlabel('Datasets', fontweight='bold', fontsize=15)
    plt.ylabel('Values', fontweight='bold', fontsize=15)
    plt.xticks([r + bar_width for r in range(len(average_losses))], dataset_names)

    plt.legend()
    plt.show()


def plot_multi_class_classification_outcome(average_loss, average_accuracy, average_tpr, dataset_names):
    bar_width = 0.25
    plt.subplots(figsize=(10, 6))

    plt.bar(1, average_loss, color='#ea5545', width=bar_width, edgecolor='grey', label='Loss')
    plt.bar(1.25, average_accuracy, color='#bdcf32', width=bar_width, edgecolor='grey', label='Accuracy')
    plt.bar(1.5, average_tpr, color='#ef9b20', width=bar_width, edgecolor='grey', label='TPR')

    # Adding Xticks
    plt.xlabel(dataset_names[0], fontweight='bold', fontsize=15)
    plt.ylabel('Values', fontweight='bold', fontsize=15)
    plt.xticks([1.25], dataset_names)

    plt.legend()
    plt.show()


def plot_confusion_matrix(conf_mat):
    displ = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
    displ.plot()
    plt.show()