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
        audio_matrix_file_name = f"{os.path.splitext(audio_file)[0]}.csv"  # Assuming CSV has the same base name as WAV
        audio_matrix_file_path = os.path.join(audio_matrix_directory, audio_matrix_file_name)

        # Define output plot file paths
        spectrogram_plot_path = os.path.join(plot_directory, f"{os.path.splitext(audio_file)[0]}_spectrogram.png")
        time_series_plot_path = os.path.join(plot_directory, f"{os.path.splitext(audio_file)[0]}_time_series.png")
        
        # Check if the plots already exist
        if not (os.path.exists(spectrogram_plot_path) and os.path.exists(time_series_plot_path)):
            print(f"Generating plots for {audio_file}...")

            # Load audio data with librosa
            audio_data, sr = librosa.load(audio_file_path, sr=None)  # sr=None keeps the original sample rate
            
            # Generate the spectrogram plot
            print(f"Generating spectrogram for {audio_file}...")
            plt.figure(figsize=(10, 6))
            S = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128, fmax=8000)
            S_dB = librosa.power_to_db(S, ref=np.max)
            librosa.display.specshow(S_dB, sr=sr, fmax=8000, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f"Mel-Spectrogram: {audio_file}")
            plt.savefig(spectrogram_plot_path)
            plt.close()  # Close the plot to save memory

            # Load the audio matrix
            print(f"Loading audio matrix from {audio_matrix_file_name}...")
            audio_matrix = pd.read_csv(audio_matrix_file_path, header=None).values.flatten()

            # Normalize the audio matrix
            audio_matrix_normalized = normalize(audio_matrix)

            # Time axis for the waveform
            time_axis_audio = np.linspace(0, len(audio_data) / sr, num=len(audio_data))

            # Time axis for the audio matrix with equal intervals over the audio duration
            matrix_time_axis = np.linspace(0, len(audio_data) / sr, num=len(audio_matrix_normalized))

            # Interpolate the normalized audio matrix points for a smoother curve
            interpolation_function = interp1d(matrix_time_axis, audio_matrix_normalized, kind='linear', fill_value="extrapolate")
            
            # Create a new time axis for smooth plotting
            smooth_time_axis = np.linspace(0, len(audio_data) / sr, num=1000)  # 1000 points for smoothness
            smooth_audio_matrix = interpolation_function(smooth_time_axis)

            # Apply Gaussian smoothing to the interpolated audio matrix
            sigma = 0.5  # Increased value for more smoothing
            smooth_audio_matrix = gaussian_filter1d(smooth_audio_matrix, sigma=sigma)

            # Generate the time-series plot
            print(f"Generating time-series plot for {audio_file}...")
            plt.figure(figsize=(10, 6))

            # Plot audio waveform
            plt.plot(time_axis_audio, audio_data, label="Audio Waveform", color='blue', linewidth=0.8)

            # Plot the smoothed normalized audio matrix as a curve
            plt.plot(smooth_time_axis, smooth_audio_matrix, color='green', label='Smoothed Audio Matrix Curve', linewidth=1.5)

            # Plot the original normalized audio matrix as points
            plt.scatter(matrix_time_axis, audio_matrix_normalized, color='red', s=10, label='Normalized Audio Matrix Points')

            # Add labels and legend
            plt.title(f"Time-Series Plot: {audio_file}")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Amplitude")
            plt.legend()

            # Save the time-series plot
            plt.savefig(time_series_plot_path)
            plt.close()  # Close the plot to save memory

        else:
            print(f"Plots for {audio_file} already exist.")