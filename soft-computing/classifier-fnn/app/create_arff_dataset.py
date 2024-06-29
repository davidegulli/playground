import os
import csv
import arff


def generate_arff_files_from_csv_files():

    datasets_folder = '../files/human_actions_datasets/'
    filelist = os.listdir(datasets_folder)

    for filename in filelist:
        if filename.endswith('.csv') and filename != 'fall_dataset.csv':
            csv_path = os.path.join(datasets_folder, filename)
            arff_path = os.path.join(datasets_folder, filename.replace('.csv', '.arff'))

            with open(csv_path, 'r') as csvfile:
                reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
                next(reader)
                data = list(reader)

            for row in data:
                row.append(1)

            filter_function = lambda item: not item.endswith(filename) and item.endswith('.csv') and item != 'fall_dataset.csv'

            negative_class_files = filter(filter_function, filelist)
            for negative_class_file in negative_class_files:
                negative_class_csv_path = os.path.join(datasets_folder, negative_class_file)
                with open(negative_class_csv_path, 'r') as csvfile:
                    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
                    next(reader)
                    negative_data = list(reader)
                    for row in negative_data:
                        row.append(0)

                    data = data + negative_data

            relation = filename.replace('.csv', '')
            columns = [
                'X-Gyroscope',
                'Y-Gyroscope',
                'Z-Gyroscope',
                'X-Accelerometer',
                'Y-Accelerometer',
                'Z-Accelerometer',
                'Target'
            ]

            arff.dump(arff_path, data, relation=relation, names=columns)

            print(f"Converted {filename} to ARFF format.")
