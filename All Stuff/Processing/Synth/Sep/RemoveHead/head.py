import os
import pandas as pd

# Function to remove odd rows from a DataFrame
def remove_odd_rows(df):
    return df.iloc[::2]

# Path to the folder containing CSV files
folder_path = "C:/Users/aryan/PycharmProjects/pythonProject/Processing/Synth/Sep/RemoveHead"

# Create a new folder for the modified CSV files
output_folder = os.path.join(folder_path, "Removed_Heading")
os.makedirs(output_folder, exist_ok=True)

# List all CSV files in the folder
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# Iterate through each CSV file
for file in csv_files:
    file_path = os.path.join(folder_path, file)

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Remove odd rows
    df = remove_odd_rows(df)

    # Save the modified DataFrame back to CSV
    new_file_path = os.path.join(output_folder, f"modified_{file}")
    df.to_csv(new_file_path, index=False)
