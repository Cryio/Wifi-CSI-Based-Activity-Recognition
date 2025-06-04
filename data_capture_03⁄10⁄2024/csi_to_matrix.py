import os
import numpy as np
import csv

def process_csi_data(file_path):
    amplitudes = []
    phases = []
    combined = []

    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)  # Skip header
        for row in reader:
            amplitude_row = []
            phase_row = []
            combined_row = []
            
            # Extract the CSI data from the 26th field (index 25)
            csi_string = row[25].strip().replace('[', '').replace(']', '')
            csi_values = csi_string.split()
            
            for i in range(0, len(csi_values), 2):
                try:
                    real_part = int(csi_values[i])
                    imag_part = int(csi_values[i + 1])
                    
                    amplitude = np.sqrt(real_part**2 + imag_part**2)
                    phase = np.arctan2(imag_part, real_part)
                    
                    amplitude_row.append(amplitude)
                    phase_row.append(phase)
                    combined_row.append(complex(real_part, imag_part))
                except (ValueError, IndexError) as e:
                    amplitude_row.append(np.nan)
                    phase_row.append(np.nan)
                    combined_row.append(np.nan)
            
            amplitudes.append(amplitude_row)
            phases.append(phase_row)
            combined.append(combined_row)
    
    return np.array(amplitudes, dtype=object), np.array(phases, dtype=object), np.array(combined, dtype=object)

def save_matrix_to_csv(matrix, csv_file):
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in matrix:
            writer.writerow(row)

def convert_csi_files(csi_dir, output_dir_amplitude, output_dir_phase, output_dir_combined):
    os.makedirs(output_dir_amplitude, exist_ok=True)
    os.makedirs(output_dir_phase, exist_ok=True)
    os.makedirs(output_dir_combined, exist_ok=True)
    
    for filename in os.listdir(csi_dir):
        if filename.endswith(".csv"):
            input_path = os.path.join(csi_dir, filename)
            output_path_amplitude = os.path.join(output_dir_amplitude, filename)
            output_path_phase = os.path.join(output_dir_phase, filename)
            output_path_combined = os.path.join(output_dir_combined, filename)
            
            if os.path.exists(output_path_amplitude) and os.path.exists(output_path_phase) and os.path.exists(output_path_combined):
                print(f"Skipping already converted file: {filename}")
                continue
            
            amplitudes, phases, combined = process_csi_data(input_path)
            
            save_matrix_to_csv(amplitudes, output_path_amplitude)
            save_matrix_to_csv(phases, output_path_phase)
            save_matrix_to_csv(combined, output_path_combined)
            print(f"Converted and saved: {filename}")

if __name__ == "__main__":
    csi_dir = 'data_capture/csi'
    output_dir_amplitude = 'data_capture/csi_amplitude'
    output_dir_phase = 'data_capture/csi_phase'
    output_dir_combined = 'data_capture/csi_combined'
    convert_csi_files(csi_dir, output_dir_amplitude, output_dir_phase, output_dir_combined)
