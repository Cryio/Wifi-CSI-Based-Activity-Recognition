import os
import numpy as np
import pandas as pd
from CSIKit.filters.passband import lowpass
from CSIKit.filters.statistical import running_mean
from CSIKit.util.filters import hampel
from CSIKit.tools.batch_graph import BatchGraph

# Directory containing CSI files
csi_directory = 'data_capture/csi'
# Directory to save the plots
plot_directory = 'data_capture/csi_plot'
os.makedirs(plot_directory, exist_ok=True)

# Apply filters and plot CSI data
def process_and_plot_csi_files(csi_directory, plot_directory):
    for filename in os.listdir(csi_directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(csi_directory, filename)
            print(f"Processing file: {file_path}")

            # Read CSI data from CSV
            csi_data = pd.read_csv(file_path)
            
            # Assuming that CSI data is in columns named like 'CSI_DATA_1', 'CSI_DATA_2', ...
            csi_columns = [col for col in csi_data.columns if col.startswith('CSI_DATA')]
            csi_matrix = csi_data[csi_columns].values

            # Assuming there are multiple frames, each row is a frame, and columns are subcarriers
            no_frames, no_subcarriers = csi_matrix.shape

            # Apply filters
            for x in range(no_frames):
                csi_matrix[x] = lowpass(csi_matrix[x], 10, 100, 5)
                csi_matrix[x] = hampel(csi_matrix[x], 10, 3)
                csi_matrix[x] = running_mean(csi_matrix[x], 10)

            # Determine output file path
            output_file = os.path.join(plot_directory, filename.replace(".csv", "_heatmap.png"))

            # Check if plot already exists
            if not os.path.exists(output_file):
                # Plot heatmap and save
                BatchGraph.plot_heatmap(csi_matrix, csi_data.index, save_to_disk=True, output_file=output_file)
                print(f"Plot saved at: {output_file}")
            else:
                print(f"Plot already exists: {output_file}")

if __name__ == "__main__":
    process_and_plot_csi_files(csi_directory, plot_directory)


# Compare 2 time series data - csi and audio signal - establish the relation between - cross-corelation matrix/function (ranges from negative to positive) - value of the cross-corelation matrix
# Deadline by tomorrow morning