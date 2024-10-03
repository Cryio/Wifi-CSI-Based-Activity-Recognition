import os
import numpy as np
import csv
import re
from datetime import datetime
import matplotlib.pyplot as plt

# Directory paths
csi_directory = 'data_capture/csi'
amplitude_directory = 'data_capture/csi_amplitude'
phase_directory = 'data_capture/csi_phase'
plot_directory = 'data_capture/csi_plot'
os.makedirs(plot_directory, exist_ok=True)

# Function to parse the CSI data string (assuming complex values: real + j*imaginary)
def parse_csi_string(csi_string):
    match = re.search(r'\[(.*?)\]', csi_string)
    if match:
        csi_values = list(map(complex, match.group(1).split()))
        return csi_values
    return []

# Function to load CSI data and timestamps from CSV file
def load_csi_data(csi_file):
    csi_data = []
    timestamps = []

    with open(csi_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            csi_data.append(row['CSI_Data'])
            timestamp_str = row['Timestamp']
            try:
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S-%f")
                timestamps.append(timestamp)
            except ValueError:
                print(f"Error parsing timestamp: {timestamp_str}")
                continue

    return np.array(csi_data), timestamps

# Function to calculate the CSI sampling rate
def calculate_csi_sampling_rate(timestamps):
    time_differences = np.diff([ts.timestamp() for ts in timestamps])
    avg_time_interval = np.mean(time_differences)
    csi_frequency = 1 / avg_time_interval
    print(f"Average time interval: {avg_time_interval:.5f} seconds")
    print(f"CSI sampling rate: {csi_frequency:.2f} Hz")
    return csi_frequency

# Function to plot amplitude for all subcarriers
def plot_amplitude_for_all_subcarriers(csi_matrix, time_steps, output_file):
    plt.figure(figsize=(18, 10))
    for subcarrier in range(csi_matrix.shape[1]):
        amplitude_data = np.abs(csi_matrix[:, subcarrier])
        plt.plot(time_steps, amplitude_data)
    plt.title('Amplitude vs Time for All Subcarriers')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()

# Function to plot phase for all subcarriers
def plot_phase_for_all_subcarriers(csi_matrix, time_steps, output_file):
    plt.figure(figsize=(18, 10))
    for subcarrier in range(csi_matrix.shape[1]):
        phase_data = np.angle(csi_matrix[:, subcarrier])
        plt.plot(time_steps, phase_data)
    plt.title('Phase vs Time for All Subcarriers')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Phase (radians)')
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()

# Function to load amplitude and phase data from CSV files
def load_phase_amp_data_from_csv(file_path):
    data_list = []
    with open(file_path, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split(',')))
            data_list.append(values)
    return np.array(data_list)

# Function to plot heatmap of amplitude and phase data
def plot_heatmap(amplitude_data, phase_data, time_steps, output_file):
    plt.figure(figsize=(18, 10))

    # Amplitude heatmap
    plt.subplot(2, 1, 1)
    plt.imshow(amplitude_data.T, aspect='auto', cmap='inferno', 
               extent=[time_steps[0], time_steps[-1], 0, amplitude_data.shape[0]])
    plt.colorbar(label='Amplitude')
    plt.title('Amplitude Heatmap of Subcarriers')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Subcarriers')

    # Phase heatmap
    plt.subplot(2, 1, 2)
    plt.imshow(phase_data.T, aspect='auto', cmap='Reds', vmin=-np.pi, vmax=np.pi, 
               extent=[time_steps[0], time_steps[-1], 0, phase_data.shape[0]])
    plt.colorbar(label='Phase (radians)')
    plt.title('Phase Heatmap of Subcarriers')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Subcarriers')

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Combined heatmap saved at: {output_file}")

# Function to process and plot graphs for amplitude and phase data
def process_and_plot_graphs(amplitude_directory, phase_directory, plot_directory, time_steps, filename):
    # Check if output files already exist
    output_file_heatmap = os.path.join(plot_directory, f"{filename.replace('.csv', '')}_heatmap.png")
    output_file_amp = os.path.join(plot_directory, f"{filename.replace('.csv', '')}_amp_all_subcarriers.png")
    output_file_phase = os.path.join(plot_directory, f"{filename.replace('.csv', '')}_phase_all_subcarriers.png")

    if (os.path.exists(output_file_heatmap) and 
        os.path.exists(output_file_amp) and 
        os.path.exists(output_file_phase)):
        print(f"Files for {filename} already processed. Skipping...")
        return

    amplitude_file_path = os.path.join(amplitude_directory, filename)
    phase_file_path = os.path.join(phase_directory, filename)

    amplitude_data = load_phase_amp_data_from_csv(amplitude_file_path)
    phase_data = load_phase_amp_data_from_csv(phase_file_path)

    # Ensure amplitude and phase data have the same shape
    if amplitude_data.shape != phase_data.shape or amplitude_data.size == 0 or phase_data.size == 0:
        print(f"Warning: Shape mismatch or empty data for {filename}. Skipping...")
        return

    # Calculate time steps for heatmap based on amplitude data
    time_steps_for_heatmap = np.linspace(time_steps[0], time_steps[-1], amplitude_data.shape[0])

    # Save plots
    plot_heatmap(amplitude_data, phase_data, time_steps_for_heatmap, output_file_heatmap)
    plot_amplitude_for_all_subcarriers(amplitude_data, time_steps_for_heatmap, output_file_amp)
    plot_phase_for_all_subcarriers(phase_data, time_steps_for_heatmap, output_file_phase)

# Main function to process CSI files
def process_and_plot_csi_files(csi_directory, plot_directory):
    for filename in os.listdir(csi_directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(csi_directory, filename)
            print(f"Processing file: {file_path}")

            # Check if output files already exist
            output_file_heatmap = os.path.join(plot_directory, f"{filename.replace('.csv', '')}_heatmap.png")
            output_file_amp = os.path.join(plot_directory, f"{filename.replace('.csv', '')}_amp_all_subcarriers.png")
            output_file_phase = os.path.join(plot_directory, f"{filename.replace('.csv', '')}_phase_all_subcarriers.png")

            if (os.path.exists(output_file_heatmap) and 
                os.path.exists(output_file_amp) and 
                os.path.exists(output_file_phase)):
                print(f"Files for {filename} already processed. Skipping...")
                continue

            csi_data, timestamps = load_csi_data(file_path)

            if len(timestamps) < 2:
                print(f"Insufficient valid timestamps in {filename}. Skipping...")
                continue

            sampling_rate_csi = calculate_csi_sampling_rate(timestamps)
            csi_matrices = [parse_csi_string(csi_entry) for csi_entry in csi_data]
            csi_matrix = np.array(csi_matrices)

            if csi_matrix.size == 0:
                print(f"Warning: {filename} contains no valid CSI data. Skipping...")
                continue

            time_steps = np.arange(csi_matrix.shape[0]) / sampling_rate_csi
            process_and_plot_graphs(amplitude_directory, phase_directory, plot_directory, time_steps, filename)

if __name__ == "__main__":
    process_and_plot_csi_files(csi_directory, plot_directory)
