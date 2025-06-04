import os
import numpy as np
import csv
import re
from datetime import datetime
import matplotlib.pyplot as plt

# Directory containing CSI files
csi_directory = 'data_capture/csi'
# Directory to save the plots
plot_directory = 'data_capture/csi_plot'
os.makedirs(plot_directory, exist_ok=True)

# Function to parse the CSI data string (assuming complex values: real + j*imaginary)
def parse_csi_string(csi_string):
    # Extract the values between the square brackets
    match = re.search(r'\[(.*?)\]', csi_string)
    if match:
        # Split the extracted string into individual values and convert to complex numbers
        csi_values = list(map(complex, match.group(1).split()))
        return csi_values
    return []

def load_csi_data(csi_file):
    """
    Load CSI data and timestamps from a CSV file.

    Args:
        csi_file (str): Path to the CSI data file.

    Returns:
        csi_data (np.ndarray): CSI data array.
        timestamps (list of datetime): List of timestamps as datetime objects for each CSI sample.
    """
    csi_data = []
    timestamps = []

    # Open the CSI file and parse it
    with open(csi_file, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Get the CSI data and timestamp from each row
            csi_data.append(row['CSI_Data'])
            
            # Parse the timestamp string into a datetime object
            timestamp_str = row['Timestamp']
            try:
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S-%f")
                timestamps.append(timestamp)
            except ValueError as e:
                print(f"Error parsing timestamp: {timestamp_str}")
                continue

    # Convert CSI data to a numpy array
    csi_data = np.array(csi_data)

    return csi_data, timestamps

def calculate_csi_sampling_rate(timestamps):
    """
    Calculate the CSI frequency (sampling rate) based on the time differences between timestamps.

    Args:
        timestamps (list of datetime): List of timestamps as datetime objects for each CSI sample.

    Returns:
        csi_frequency (float): Estimated CSI sampling rate in Hz.
    """
    # Calculate time differences between consecutive samples
    time_differences = np.diff([ts.timestamp() for ts in timestamps])

    # Calculate average sampling interval in seconds
    avg_time_interval = np.mean(time_differences)

    # The CSI sampling rate is the inverse of the average sampling interval
    csi_frequency = 1 / avg_time_interval

    print(f"Average time interval: {avg_time_interval} seconds")
    print(f"CSI sampling rate: {csi_frequency:.2f} Hz")

    return csi_frequency

# Function to plot amplitude for all subcarriers in a single plot
def plot_amplitude_for_all_subcarriers(csi_matrix, time_steps, output_file):
    plt.figure(figsize=(12, 8))
    for subcarrier in range(csi_matrix.shape[1]):
        amplitude_data = np.abs(csi_matrix[:, subcarrier])
        plt.plot(time_steps, amplitude_data, label=f'Subcarrier {subcarrier}')
    plt.title('Amplitude vs Time for All Subcarriers')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()

# Function to plot phase for all subcarriers in a single plot
def plot_phase_for_all_subcarriers(csi_matrix, time_steps, output_file):
    plt.figure(figsize=(12, 8))
    for subcarrier in range(csi_matrix.shape[1]):
        phase_data = np.angle(csi_matrix[:, subcarrier])
        plt.plot(time_steps, phase_data, label=f'Subcarrier {subcarrier}')
    plt.title('Phase vs Time for All Subcarriers')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Phase (radians)')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()

# Function to plot a heatmap of amplitude values
def plot_heatmap(csi_matrix, time_steps, output_file):
    plt.figure(figsize=(12, 8))
    plt.imshow(np.abs(csi_matrix.T), aspect='auto', cmap='viridis', extent=[time_steps[0], time_steps[-1], 0, csi_matrix.shape[1]])
    plt.colorbar(label='Amplitude')
    plt.title('Amplitude Heatmap of Subcarriers')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Subcarriers')
    plt.savefig(output_file)
    plt.close()

# Process and plot CSI data (Amplitude, Phase, and Heatmap)
def process_and_plot_csi_files(csi_directory, plot_directory):
    for filename in os.listdir(csi_directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(csi_directory, filename)
            print(f"Processing file: {file_path}")

            # Load CSI data and timestamps using the load_csi_data function
            csi_data, timestamps = load_csi_data(file_path)

            if len(timestamps) < 2:
                print(f"Insufficient valid timestamps in {filename}. Skipping...")
                continue

            # Calculate the CSI sampling rate from timestamps
            sampling_rate_csi = calculate_csi_sampling_rate(timestamps)

            # Parse CSI data into a matrix
            csi_matrices = [parse_csi_string(csi_entry) for csi_entry in csi_data]
            csi_matrix = np.array(csi_matrices)

            # Check if csi_matrix contains valid data
            if csi_matrix.size == 0:
                print(f"Warning: {filename} contains no valid CSI data. Skipping...")
                continue

            # Print the shape of the CSI matrix (time samples x subcarriers)
            print(f"csi_matrix shape: {csi_matrix.shape}")

            # Generate the time steps based on the CSI sampling rate
            time_steps = np.arange(csi_matrix.shape[0]) / sampling_rate_csi

            # Plot amplitude for all subcarriers
            output_file_amp = os.path.join(plot_directory, f"{filename.replace('.csv', '')}_amp_all_subcarriers.png")
            plot_amplitude_for_all_subcarriers(csi_matrix, time_steps, output_file_amp)
            print(f"Amplitude plot saved at: {output_file_amp}")

            # Plot phase for all subcarriers
            output_file_phase = os.path.join(plot_directory, f"{filename.replace('.csv', '')}_phase_all_subcarriers.png")
            plot_phase_for_all_subcarriers(csi_matrix, time_steps, output_file_phase)
            print(f"Phase plot saved at: {output_file_phase}")

            # Plot heatmap of amplitude for all subcarriers
            output_file_heatmap = os.path.join(plot_directory, f"{filename.replace('.csv', '')}_amplitude_heatmap.png")
            plot_heatmap(csi_matrix, time_steps, output_file_heatmap)
            print(f"Heatmap saved at: {output_file_heatmap}")

if __name__ == "__main__":
    process_and_plot_csi_files(csi_directory, plot_directory)
