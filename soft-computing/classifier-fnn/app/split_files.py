import csv
import os


def split_csv_by_action():
    input_file = '../files/human_actions_datasets/fall_dataset.csv'
    output_dir = '../files/human_actions_datasets'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    action_files = {}

    with open(input_file, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        columns = reader.fieldnames

        for row in reader:
            action = row['Action']

            if action not in action_files:
                file_path = os.path.join(output_dir, f"{action.replace(' ', "_").lower()}.csv")
                action_files[action] = open(file_path, 'w', newline='')

            writer = csv.DictWriter(action_files[action], fieldnames=columns[1:7])
            row.pop('Timestamp')
            row.pop('Action')
            writer.writerow(row)

    # Close all the files
    for file in action_files.values():
        file.close()
