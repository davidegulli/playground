import os
import pandas as pd
from binary_classifier import BinaryClassifier
from create_arff_dataset import generate_arff_files_from_csv_files
from graphs import plot_binary_classification_outcome, plot_multi_class_classification_outcome
from multi_classifier import MultiClassifier
from split_files import split_csv_by_action
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def load_arff_files(file_path):
    return pd.read_csv(file_path, comment='@', header=None)


def normalize_data(features):
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(features), columns=features.columns)


def preprocess_binary_class_data(dataset):
    features = dataset.iloc[:, :-1]
    targets = dataset.iloc[:, -1]

    return normalize_data(features), targets


def preprocess_multi_class_data(dataset):
    # Removing the header and the timestamp column
    dataset = dataset.iloc[1:]
    dataset = dataset.reset_index(drop=True)
    dataset.drop(columns=dataset.columns[0], axis=1, inplace=True)

    # Applying one-hot encoding to the categorical columns
    encoder = OneHotEncoder(sparse_output=False)
    targets = encoder.fit_transform(dataset[[7]])
    targets = pd.DataFrame(targets)

    # Removing action column from the features
    features = dataset.iloc[:, :-1]

    return normalize_data(features), targets


def run_binary_classifier_evaluation():
    folder_path = '../files/human_actions_datasets'
    dataset_names = []
    average_losses = []
    average_accuracies = []
    average_recalls = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.arff'):
            dataset_path = os.path.join(folder_path, filename)
            print(f'--- File: {filename} ---------------------------')

            dataframe = load_arff_files(dataset_path)
            normalized_dataset, targets = preprocess_binary_class_data(dataframe)

            positive_ratio, average_loss, average_accuracy, average_recall = (
                BinaryClassifier().train_and_evaluate_model(normalized_dataset, targets)
            )

            dataset_name = filename.replace("_", " ").replace(".arff", '').title()
            dataset_names.append(f'{dataset_name} \n Pos. Ratio: {positive_ratio:.2%}')
            average_losses.append(average_loss)
            average_accuracies.append(average_accuracy)
            average_recalls.append(average_recall)

    plot_binary_classification_outcome(average_losses, average_accuracies, average_recalls, dataset_names)


def run_multi_class_classifier_evaluation():
    print(f'--- File: Fall Dataset ---------------------------')
    file_path = '../files/human_actions_datasets/fall_dataset.csv'
    dataframe = load_arff_files(file_path)
    features, targets = preprocess_multi_class_data(dataframe)

    classifier = MultiClassifier()
    average_loss, average_accuracy, average_recall, average_tper = (
        classifier.train_and_evaluate_model(features, targets)
    )

    plot_multi_class_classification_outcome(average_loss, average_accuracy, average_tper, ['Fall Dataset'])


split_csv_by_action()
generate_arff_files_from_csv_files()
run_binary_classifier_evaluation()
run_multi_class_classifier_evaluation()