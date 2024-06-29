import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, TruePositives, FalseNegatives
from sklearn.metrics import confusion_matrix
from graphs import plot_confusion_matrix


class MultiClassifier:

    def train_and_evaluate_model(self, dataset, targets):
        fold_no = 1
        accuracies = []
        losses = []
        recalls = []
        tprs = []

        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        for train_index, test_index in kfold.split(dataset):
            training_dataset, testing_dataset = dataset.iloc[train_index], dataset.iloc[test_index]
            training_targets, testing_targets = targets.iloc[train_index], targets.iloc[test_index]

            model = self.__create_model(training_dataset.shape[1])

            model.fit(
                training_dataset,
                training_targets,
                epochs=100,
                batch_size=32,
                validation_data=(testing_dataset, testing_targets),
                verbose=0
            )

            scores = model.evaluate(testing_dataset, testing_targets, verbose=0)

            loss, accuracy, recall, true_positive, false_negative = scores
            tpr = true_positive / (true_positive + false_negative)

            print(f'Fold: {fold_no}: Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, TPR:{tpr:.2f}')

            accuracies.append(scores[1])
            losses.append(scores[0])
            recalls.append(scores[2])
            tprs.append(tpr)

            y_pred = model.predict(testing_dataset)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true = np.argmax(testing_targets.values, axis=1)

            cm = confusion_matrix(y_true, y_pred_classes)
            plot_confusion_matrix(cm)

            fold_no += 1

        # Calculating and printing evaluation metrics
        print('\nAverage scores:')

        average_loss = np.mean(losses)
        print(f'Loss: {average_loss:.4f}')

        average_accuracy = np.mean(accuracies)
        print(f'Accuracy: {average_accuracy:.4f} (+/- {np.std(accuracies):.4f})')

        average_recall = np.mean(recalls)
        print(f'Recall: {average_recall:.4f} (+/- {np.std(recalls):.4f})')

        average_tpr = np.mean(tprs)
        print(f'TPR: {average_tpr:.2%} (+/- {np.std(tprs):.4f})\n')

        return average_loss, average_accuracy, average_recall, average_tpr

    def __create_model(self, input_shape):
        model = Sequential([
            Input(shape=(input_shape,)),
            Dense(15, activation='relu'),
            Dense(5, activation='softmax')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.01),
            loss='categorical_crossentropy',
            metrics=['accuracy', Recall(), TruePositives(), FalseNegatives()]
        )

        return model
