import os
import numpy as np
import csv
import librosa
from datetime import datetime
from scipy import signal
from scipy.interpolate import interp1d
from scipy.fftpack import fft, ifft

def wav_to_matrix_librosa(wav_file, target_sr, method="linear"):
    """
    Load WAV file using librosa, resample it to the target sampling rate using the selected method.

    Args:
        wav_file (str): Path to the input WAV file.
        target_sr (int): The target sampling rate to resample the audio.
        method (str): Resampling method. Choices are: 'nearest', 'linear', 'polynomial', 'sinc', 'fft', 'bspline', 'polyphase'.

    Returns:
        audio_array (np.ndarray): The resampled audio data.
        sr (int): The actual sampling rate used by librosa (should match target_sr).
    """
    # Load the audio file
    audio_array, sr = librosa.load(wav_file, sr=None)  # Load at original sampling rate

    if sr == target_sr:
        return audio_array, sr  # No resampling needed

    if method == "nearest":
        # Nearest-neighbor resampling
        audio_resampled = signal.resample_poly(audio_array, target_sr, sr, window=('boxcar'))
    elif method == "linear":
        # Linear interpolation
        duration = len(audio_array) / sr
        time_old = np.linspace(0, duration, len(audio_array))
        time_new = np.linspace(0, duration, int(duration * target_sr))
        audio_resampled = np.interp(time_new, time_old, audio_array)
    elif method == "polynomial":
        # Polynomial interpolation (e.g., cubic)
        duration = len(audio_array) / sr
        time_old = np.linspace(0, duration, len(audio_array))
        time_new = np.linspace(0, duration, int(duration * target_sr))
        interpolator = interp1d(time_old, audio_array, kind='cubic')
        audio_resampled = interpolator(time_new)
    elif method == "sinc":
        # Sinc interpolation (using signal.resample)
        audio_resampled = signal.resample(audio_array, int(len(audio_array) * target_sr / sr))
    elif method == "fft":
        # FFT-based resampling
        audio_fft = fft(audio_array)
        audio_resampled_fft = signal.resample(audio_fft, int(len(audio_array) * target_sr / sr))
        audio_resampled = np.real(ifft(audio_resampled_fft))
    elif method == "bspline":
        # B-spline interpolation
        audio_resampled = signal.resample_poly(audio_array, target_sr, sr, window='hamming')
    elif method == "polyphase":
        # Polyphase filtering
        audio_resampled = signal.resample_poly(audio_array, target_sr, sr)
    else:
        raise ValueError(f"Unknown resampling method: {method}")

    return audio_resampled, target_sr

def choose_resampling_method():
    """
    Prompt the user to select a resampling method by displaying options with descriptions.

    Returns:
        method (str): The chosen resampling method.
    """
    print("Choose a resampling method:")
    print("1. Nearest-neighbor: Fastest but lowest fidelity, often used for low-latency applications.")
    print("2. Linear interpolation: Moderate fidelity and speed, good for general-purpose use.")
    print("3. Polynomial interpolation: Higher fidelity but slower than linear, suitable for smooth data.")
    print("4. Sinc interpolation: High fidelity, ideal for high-quality audio but slower to compute.")
    print("5. FFT-based resampling: High precision, especially for large signals, but computationally expensive.")
    print("6. B-spline interpolation: High fidelity, with smooth transitions, good for continuous signals.")
    print("7. Polyphase filtering: Efficient for multirate systems, a good trade-off between speed and quality.")

    choice = input("Enter the number corresponding to your choice (1-7): ")

    methods = {
        "1": "nearest",
        "2": "linear",
        "3": "polynomial",
        "4": "sinc",
        "5": "fft",
        "6": "bspline",
        "7": "polyphase"
    }

    return methods.get(choice, "linear")  # Default to linear if invalid choice

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

def save_matrix_to_csv(matrix, csv_file):
    """
    Save the audio matrix as a CSV file.

    Args:
        matrix (np.ndarray): The audio data to save.
        csv_file (str): Path to the output CSV file.
    """
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)

        if matrix.ndim == 1:
            for value in matrix:
                writer.writerow([value])
        else:
            for row in matrix:
                writer.writerow(row)

def convert_all_wavs_to_csv(audio_dir, output_dir, csi_dir, resampling_method):
    """
    Convert all WAV files in a directory to CSV files after resampling to the target CSI sampling rate.

    Args:
        audio_dir (str): Directory containing the WAV files.
        output_dir (str): Directory to save the converted CSV files.
        csi_dir (str): Directory containing the corresponding CSI data files.
        resampling_method (str): Resampling method to use.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(audio_dir):
        if filename.endswith(".wav"):
            csv_filename = filename.replace(".wav", ".csv")
            csi_filename = "csi_data_" + csv_filename

            wav_path = os.path.join(audio_dir, filename)
            csi_path = os.path.join(csi_dir, csi_filename)

            if not os.path.exists(csi_path):
                print(f"No matching CSI file found for {filename}, skipping.")
                continue

            # Load the CSI data and calculate the sampling rate
            csi_data, timestamps = load_csi_data(csi_path)
            csi_sampling_rate = calculate_csi_sampling_rate(timestamps)
            print(f"Calculated CSI sampling rate for {filename}: {csi_sampling_rate:.2f} Hz")

            # Resample the audio to match the CSI sampling rate
            audio_matrix, sr = wav_to_matrix_librosa(wav_path, int(csi_sampling_rate), method=resampling_method)

            # Save the matrix as a CSV file
            output_csv_path = os.path.join(output_dir, csv_filename)
            save_matrix_to_csv(audio_matrix, output_csv_path)
            print(f"Converted and saved: {output_csv_path} (Resampled to {csi_sampling_rate:.2f} Hz using {resampling_method})")

if __name__ == "__main__":
    # Define directories for audio files, CSI data, and output
    audio_dir = 'data_capture/audio'
    csi_dir = 'data_capture/csi'
    output_dir = 'data_capture/audio_matrix'

    # User chooses the resampling method
    resampling_method = choose_resampling_method()

    # Convert all wav files to CSV, resampling to match CSI sampling rate
    convert_all_wavs_to_csv(audio_dir, output_dir, csi_dir, resampling_method)
