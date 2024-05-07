import os
import pandas as pd

def sort_csv_files_by_first_column(folder_path):
    # Get a list of CSV files in the folder
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

    for file in csv_files:
        # Read each CSV file into a DataFrame
        df = pd.read_csv(os.path.join(folder_path, file))

        # Sort the DataFrame by the first column
        df_sorted = df.sort_values(by=df.columns[0])

        # Write the sorted DataFrame back to the CSV file
        df_sorted.to_csv(os.path.join(folder_path, file), index=False)

# Provide the folder path containing CSV files
folder_path = 'C:/Users/aryan/PycharmProjects/pythonProject/Processing/Synth/Sep/RemoveHead/Removed_Heading'
sort_csv_files_by_first_column(folder_path)
