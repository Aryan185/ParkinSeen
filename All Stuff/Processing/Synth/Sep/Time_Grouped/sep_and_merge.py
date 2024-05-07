import csv
import os

# Function to create a new CSV file for each row in a given CSV file
def split_csv_file(csv_file, output_folder):
    with open(csv_file, 'r', newline='') as file:
        reader = csv.reader(file)
        headers = next(reader)  # Get the header row
        for i, row in enumerate(reader, start=1):
            # Create a new filename
            base_name, ext = os.path.splitext(csv_file)
            new_file_name = f"{base_name}_{i}{ext}"

            # Write the row to the new CSV file
            with open(os.path.join(output_folder, new_file_name), 'w', newline='') as new_file:
                writer = csv.writer(new_file)
                writer.writerow(headers)
                writer.writerow(row)


# Function to merge CSV files with the same {i}
def merge_csv_files(folder, output_folder):
    merged_files = {}

    for file in os.listdir(folder):
        if file.endswith('.csv'):
            base_name, ext = os.path.splitext(file)
            if "_" in base_name:
                i = base_name.split("_")[-1]
                if i not in merged_files:
                    merged_files[i] = []
                merged_files[i].append(file)

    for i, files in merged_files.items():
        merged_data = []
        headers_written = False
        for file in files:
            with open(os.path.join(folder, file), 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                if not headers_written:
                    headers = next(reader)
                    merged_data.append(headers)
                    headers_written = True
                for row in reader:
                    merged_data.append(row)

        # Write merged data to new CSV
        with open(os.path.join(output_folder, f"merged_{i}.csv"), 'w', newline='') as merged_csv:
            writer = csv.writer(merged_csv)
            writer.writerows(merged_data)


# Function to process all CSV files in a folder
def process_csv_folder(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in os.listdir(input_folder):
        if file.endswith('.csv'):
            file_path = os.path.join(input_folder, file)
            split_csv_file(file_path, output_folder)

    merge_csv_files(input_folder, output_folder)


# Provide the path to the input folder containing CSV files
input_folder_path = "C:/Users/aryan/PycharmProjects/pythonProject/Processing/Synth/Sep/Time_Grouped/GroupedData"

# Provide the path to the output folder where you want to save the merged CSV files
output_folder_path = "C:/Users/aryan/PycharmProjects/pythonProject/Processing/Synth/Sep/Time_Grouped/SeperateData"

# Call the function to process all CSV files in the input folder and save output in the output folder
process_csv_folder(input_folder_path, output_folder_path)
