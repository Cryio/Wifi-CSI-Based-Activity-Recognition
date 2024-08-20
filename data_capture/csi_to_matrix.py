import os
import numpy as np
import csv

def process_csi_data(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)  # Skip header
        csi_data = []
        for row in reader:
            csi_data_row = []
            
            # Extract the CSI data from the 26th field (index 25)
            csi_string = row[25].strip().replace('[', '').replace(']', '')
            csi_values = csi_string.split()
            
            for i in range(0, len(csi_values), 2):
                try:
                    # Pairing each magnitude with its corresponding phase to form a complex number
                    real_part = int(csi_values[i])
                    imag_part = int(csi_values[i+1])
                    amplitude = np.sqrt(real_part**2 + imag_part**2)  # Calculate amplitude
                    csi_data_row.append(amplitude)  # Store only amplitude
                except (ValueError, IndexError) as e:
                    csi_data_row.append(np.nan)  # Handle malformed or incomplete pairs
            
            csi_data.append(csi_data_row)  # Append the row even if it has NaN values
    
    # Check for consistent row lengths
    row_lengths = {len(row) for row in csi_data}
    if len(row_lengths) > 1:
        print(f"Warning: Inconsistent row lengths found: {row_lengths}")

    # Use dtype=object to handle varying lengths of rows
    return np.array(csi_data, dtype=object)

def save_matrix_to_csv(matrix, csv_file):
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in matrix:
            writer.writerow(row)

def convert_csi_files(csi_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(csi_dir):
        if filename.endswith(".csv"):
            input_path = os.path.join(csi_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            if os.path.exists(output_path):
                print(f"Skipping already converted file: {output_path}")
                continue
            
            csi_matrix = process_csi_data(input_path)
            save_matrix_to_csv(csi_matrix, output_path)
            print(f"Converted and saved: {output_path}")

if __name__ == "__main__":
    csi_dir = 'data_capture/csi'
    output_dir = 'data_capture/csi_matrix'
    convert_csi_files(csi_dir, output_dir)
