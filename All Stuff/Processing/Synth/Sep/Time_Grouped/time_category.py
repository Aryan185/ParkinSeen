import pandas as pd

# Read the CSV file
df = pd.read_csv('C:/Users/aryan/PycharmProjects/pythonProject/Processing/Synth/standard_synthetic_data.csv')

# Get unique values in 'Time' column
unique_times = df['Time'].unique()

# Iterate over unique values and create separate CSV files
for time_value in unique_times:
    # Filter the dataframe for each unique 'Time' value
    filtered_df = df[df['Time'] == time_value]

    # Generate the filename
    filename = f'{time_value}_data.csv'

    # Write the filtered dataframe to a new CSV file
    filtered_df.to_csv(filename, index=False)
    print(f"File {filename} generated successfully.")