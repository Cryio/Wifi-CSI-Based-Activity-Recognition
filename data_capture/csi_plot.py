import os
import numpy as np
import pandas as pd
from CSIKit.tools.batch_graph import BatchGraph
import re

# Directory containing CSI files
csi_directory = 'data_capture/csi'
# Directory to save the plots
plot_directory = 'data_capture/csi_plot'
os.makedirs(plot_directory, exist_ok=True)

# Function to parse the CSI data string
def parse_csi_string(csi_string):
    # Extract the values between the square brackets
    match = re.search(r'\[(.*?)\]', csi_string)
    if match:
        # Split the extracted string into individual values and convert to float
        csi_values = list(map(float, match.group(1).split()))
        return csi_values
    return []

# Process and plot CSI data
def process_and_plot_csi_files(csi_directory, plot_directory):
    for filename in os.listdir(csi_directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(csi_directory, filename)
            print(f"Processing file: {file_path}")

            # Read the CSV file
            csi_data = pd.read_csv(file_path, header=None)
            
            # Print the number of columns
            print(f"Number of columns in {filename}: {len(csi_data.columns)}")

            # Initialize a list to hold parsed CSI matrices
            csi_matrices = []

            # Iterate over each row to extract CSI data
            for index, row in csi_data.iterrows():
                # Check if the column exists
                if 25 < len(row):
                    # The CSI data is in the 26th column (index 25)
                    csi_values = parse_csi_string(row.iloc[25])
                    if csi_values:
                        csi_matrices.append(csi_values)
                else:
                    print(f"Column 27 does not exist in row {index} of {filename}. Skipping...")

            # Convert the list of CSI matrices to a numpy array
            csi_matrix = np.array(csi_matrices)

            # Debug: Check if csi_matrix contains valid data
            if csi_matrix.size == 0:
                print(f"Warning: {filename} contains no valid CSI data. Skipping...")
                continue
            
            print(f"csi_matrix shape: {csi_matrix.shape}")

            # Determine output file path
            output_file = os.path.join(plot_directory, filename.replace(".csv", "_heatmap.png"))

            # Plot heatmap using BatchGraph
            BatchGraph.plot_heatmap(csi_matrix, range(len(csi_matrix)), save_path=output_file)
            
            print(f"Plot saved at: {output_file}")

if __name__ == "__main__":
    process_and_plot_csi_files(csi_directory, plot_directory)
