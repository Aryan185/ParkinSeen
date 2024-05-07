# import os
# import pandas as pd
# 
# def calculate_acceleration(v1, v2, t1, t2):
#     return (v2 - v1) / (t2 - t1)
# 
# def calculate_acceleration_data(input_folder, output_folder):
#     # Create output folder if it doesn't exist
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
# 
#     # Get list of CSV files in input folder
#     csv_files = [file for file in os.listdir(input_folder) if file.endswith('.csv')]
# 
#     # Iterate over each CSV file
#     for csv_filename in csv_files:
#         try:
#             df = pd.read_csv(os.path.join(input_folder, csv_filename), names=['Time', 'Left_X', 'Left_Y', 'FOG'])
#         except FileNotFoundError:
#             print(f"Error: File '{csv_filename}' not found.")
#             continue
# 
#         if len(df) < 2:
#             print(f"Error: Insufficient data in CSV file '{csv_filename}'.")
#             continue
# 
#         df['Left_X'] = pd.to_numeric(df['Left_X'], errors='coerce')
#         df['Left_Y'] = pd.to_numeric(df['Left_Y'], errors='coerce')
#         df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
#         df['FOG'] = pd.to_numeric(df['FOG'], errors='coerce')
# 
#         accelerations = []
# 
#         for i in range(1, len(df)):
#             row = df.iloc[i]
#             prev_row = df.iloc[i-1]
# 
#             time_interval = row['Time'] - prev_row['Time']
# 
#             vx = (row['Left_X'] - prev_row['Left_X']) / time_interval
#             vy = (row['Left_Y'] - prev_row['Left_Y']) / time_interval
# 
#             ax = calculate_acceleration(prev_row['Left_X'], row['Left_X'], prev_row['Time'], row['Time'])
#             ay = calculate_acceleration(prev_row['Left_Y'], row['Left_Y'], prev_row['Time'], row['Time'])
# 
#             accelerations.append((prev_row['Time'], ax, ay, 0))
# 
#         # Save output file
#         output_filename = os.path.join(output_folder, f"acceleration_readings_{os.path.splitext(csv_filename)[0]}.csv")
#         accelerations_df = pd.DataFrame(accelerations, columns=['Time', 'Acceleration_x', 'Acceleration_y', 'FOG'])
#         accelerations_df.to_csv(output_filename, index=False)
# 
#     print("Acceleration calculations completed.")
# 
# # Specify input and output folders
# input_folder = 'C:/Users/aryan/PycharmProjects/pythonProject/Processing/extract/0'
# output_folder = 'C:/Users/aryan/PycharmProjects/pythonProject/Processing/accel_files'
# 
# # Perform calculations
# calculate_acceleration_data(input_folder, output_folder)


import csv


def remove_rows_with_null(input_file, output_file):
    with open(input_file, 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Write the header
        writer.writerow(next(reader))

        # Iterate through each row in the input CSV
        for row in reader:
            # Check if any element in the row is None or an empty string
            if not any(cell is None or cell == '' for cell in row):
                # Write the row to the output CSV if no null value is found
                writer.writerow(row)


# Example usage:
input_file = "C:/Users/aryan/Downloads/merge-csv.com__663105fae7e66.csv"# Replace 'input.csv' with your input CSV file
output_file = 'C:/Users/aryan/PycharmProjects/pythonProject/Processing/final.csv'  # Replace 'output.csv' with your desired output CSV file
remove_rows_with_null(input_file, output_file)
