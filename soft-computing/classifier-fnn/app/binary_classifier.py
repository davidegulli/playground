import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall


class BinaryClassifier:

    def train_and_evaluate_model(self, dataset, targets):
        fold_no = 1
        accuracies = []
        losses = []
        recalls = []

        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        for train_index, test_index in kfold.split(dataset):
            training_dataset, testing_dataset = dataset.iloc[train_index], dataset.iloc[test_index]
            training_targets, testing_targets = targets.iloc[train_index], targets.iloc[test_index]

            model = self.__create_model(training_dataset.shape[1])

            model.fit(
                training_dataset,
                training_targets,
                epochs=50,
                batch_size=32,
                validation_data=(testing_dataset, testing_targets),
                verbose=0
            )

            scores = model.evaluate(testing_dataset, testing_targets, verbose=0)
            print(f'Fold {fold_no}: Loss = {scores[0]:.4f}, Accuracy = {scores[1]:.4f}, Recall = {scores[2]:.4f}')

            accuracies.append(scores[1])
            losses.append(scores[0])
            recalls.append(scores[2])

            fold_no += 1

        # Calculating and printing evaluation metrics
        print('\nAverage scores:')
        class_counts = targets.value_counts().sort_index()
        positive_ratio = class_counts[1] / (class_counts[0] + class_counts[1])
        print(f"Positive ratio: {positive_ratio:.2%}")

        average_loss = np.mean(losses)
        print(f'Loss: {average_loss:.4f}')

        average_accuracy = np.mean(accuracies)
        print(f'Accuracy: {average_accuracy:.4f} (+/- {np.std(accuracies):.4f})')

        average_recall = np.mean(recalls)
        print(f'Recall: {average_recall:.4f} (+/- {np.std(recalls):.4f})\n')

        return positive_ratio, average_loss, average_accuracy, average_recall

    def __create_model(self, input_shape):
        model = Sequential([
            Input(shape=(input_shape,)),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.01),
            loss='binary_crossentropy',
            metrics=['accuracy', Recall()]
        )

        return model
