import numpy as np


class Dataset:

    def __init__(self, file_path, columns_number, training_set_ratio=0.7):
        self.file_path = file_path
        self.columns_number = columns_number
        self.training_set_ratio = training_set_ratio

    def load(self):
        dataset = np.loadtxt(self.file_path, dtype=np.float64, delimiter=',')
        data, targets = dataset[:, :-1], dataset[:, -1]
        return self.__split_datasets(
            self.__normalize_data(data),
            self.__normalize_data(targets)
        )

    def __normalize_data(self, data):
        return (data - data.mean(axis=0)) / data.std(axis=0)

    def __split_datasets(self, data, targets):
        n_train = int(self.training_set_ratio * len(data))
        indices = np.random.permutation(len(data))
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        return data[train_indices], data[test_indices], targets[train_indices], targets[test_indices]