import os
import pandas as pd

def extract_44th_subcarrier_from_directory(input_dir, output_dir):
    # Ensure the input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: The directory {input_dir} does not exist.")
        return

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through all CSV files in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".csv"):
            input_csv_path = os.path.join(input_dir, file_name)
            
            try:
                # Read the CSI data from the CSV file
                csi_data = pd.read_csv(input_csv_path, header=None)

                # Ensure there are enough columns (at least 44)
                if csi_data.shape[1] < 44:
                    print(f"Skipping {file_name}: Not enough columns for the 44th subcarrier.")
                    continue
                
                # Extract the 44th subcarrier (column 43 in 0-based index)
                subcarrier_44th = csi_data.iloc[:, 43]

                # Prepare the output file path
                output_file = os.path.join(output_dir, file_name)

                # Save the extracted subcarrier to a new CSV file
                subcarrier_44th.to_csv(output_file, index=False, header=False)
                print(f"44th subcarrier data from {file_name} saved to {output_file}")

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

# Example usage
input_dir = 'data_capture/csi_amplitude'  # Replace with the path to your CSI directory
output_dir = 'data_capture/csi_amplitude_44th'

extract_44th_subcarrier_from_directory(input_dir, output_dir)
