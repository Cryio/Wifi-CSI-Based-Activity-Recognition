import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

def normalize(data):
    """
    Normalize data to be within the range of -1 to 1.
    
    Args:
        data (np.ndarray): The data to normalize.

    Returns:
        np.ndarray: Normalized data.
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data)) * 2 - 1

# Directory paths
audio_directory = 'data_capture/audio'
audio_matrix_directory = 'data_capture/audio_matrix'

plot_directory = 'data_capture/audio_plot'

# Ensure the plot directory exists
os.makedirs(plot_directory, exist_ok=True)

# Generate plots for each audio file found in the directory and its subdirectories
for root, dirs, files in os.walk(audio_directory):
    audio_files = [f for f in files if f.endswith('.wav')]
    
    for audio_file in audio_files:
        audio_file_path = os.path.join(root, audio_file)
        base_name = os.path.splitext(audio_file)[0]
        audio_matrix_file_name = f"{base_name}.csv"
        audio_matrix_file_path = os.path.join(audio_matrix_directory, audio_matrix_file_name)

        # Skip if the audio matrix file does not exist
        if not os.path.exists(audio_matrix_file_path):
            print(f"[SKIP] Audio matrix not found for {audio_file}, skipping plot generation.")
            continue

        # Define output plot file paths
        spectrogram_plot_path = os.path.join(plot_directory, f"{base_name}_spectrogram.png")
        time_series_plot_path = os.path.join(plot_directory, f"{base_name}_time_series.png")
        
        # Skip if both plots already exist
        if os.path.exists(spectrogram_plot_path) and os.path.exists(time_series_plot_path):
            print(f"[OK] Plots for {audio_file} already exist.")
            continue

        print(f"[PROCESS] Generating plots for {audio_file}...")

        # Load audio data with librosa
        audio_data, sr = librosa.load(audio_file_path, sr=None)

        # Spectrogram
        print(f"  - Generating spectrogram...")
        plt.figure(figsize=(10, 6))
        S = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_dB, sr=sr, fmax=8000, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"Mel-Spectrogram: {audio_file}")
        plt.savefig(spectrogram_plot_path)
        plt.close()

        # Load and normalize matrix
        print(f"  - Loading and normalizing audio matrix...")
        audio_matrix = pd.read_csv(audio_matrix_file_path, header=None).values.flatten()
        audio_matrix_normalized = normalize(audio_matrix)

        # Interpolation + Smoothing
        time_axis_audio = np.linspace(0, len(audio_data) / sr, num=len(audio_data))
        matrix_time_axis = np.linspace(0, len(audio_data) / sr, num=len(audio_matrix_normalized))
        interp_fn = interp1d(matrix_time_axis, audio_matrix_normalized, kind='linear', fill_value="extrapolate")
        smooth_time_axis = np.linspace(0, len(audio_data) / sr, num=1000)
        smooth_audio_matrix = gaussian_filter1d(interp_fn(smooth_time_axis), sigma=0.5)

        # Time-series plot
        print(f"  - Generating time-series plot...")
        plt.figure(figsize=(10, 6))
        plt.plot(time_axis_audio, audio_data, label="Audio Waveform", color='blue', linewidth=0.8)
        plt.plot(smooth_time_axis, smooth_audio_matrix, color='green', label='Smoothed Audio Matrix Curve', linewidth=1.5)
        plt.scatter(matrix_time_axis, audio_matrix_normalized, color='red', s=10, label='Normalized Audio Matrix Points')
        plt.title(f"Time-Series Plot: {audio_file}")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.savefig(time_series_plot_path)
        plt.close()
