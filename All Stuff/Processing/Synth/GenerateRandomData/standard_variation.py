import pandas as pd
import numpy as np

# Load the original CSV file
original_data = pd.read_csv("C:/Users/aryan/PycharmProjects/pythonProject/Processing/Synth/11_12_71_merged.csv")

# Group by Time and calculate mean Left_X and Left_Y for each Time group
grouped_data = original_data.groupby('Time').agg({'Left_X': 'mean', 'Left_Y': 'mean'}).reset_index()

# Repeat the Time column to match the desired number of rows
num_rows = 80850
time_column = np.repeat(grouped_data['Time'], num_rows // len(grouped_data))

# Generate synthetic data for Left_X and Left_Y columns based on the mean values
synthetic_left_x = np.repeat(grouped_data['Left_X'], num_rows // len(grouped_data))
synthetic_left_y = np.repeat(grouped_data['Left_Y'], num_rows // len(grouped_data))

# Add some randomness around the mean values
std_scale = 5  # Adjust this value to control the randomness
synthetic_left_x += np.random.normal(scale=std_scale, size=num_rows)
synthetic_left_y += np.random.normal(scale=std_scale, size=num_rows)

# Create a DataFrame for the synthetic data
synthetic_data = pd.DataFrame({
    'Time': time_column,
    'Left_X': synthetic_left_x,
    'Left_Y': synthetic_left_y,
    'FOG': np.ones(num_rows)  # Set all FOG values to 1
})

# Save the synthetic data to a new CSV file
synthetic_data.to_csv("C:/Users/aryan/PycharmProjects/pythonProject/Processing/Synth/standard_synthetic_data.csv", index=False)