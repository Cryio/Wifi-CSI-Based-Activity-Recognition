import os
from pyAudioProcessing import plot
from pyAudioProcessing.extract_features import get_features

# Directory paths
audio_directory = 'data_capture/audio'
plot_directory = 'data_capture/audio_plot'

# Ensure the plot directory exists
os.makedirs(plot_directory, exist_ok=True)

# List all .wav files in the audio directory
audio_files = [f for f in os.listdir(audio_directory) if f.endswith('.wav')]

# Generate plots for each audio file
for audio_file in audio_files:
    audio_file_path = os.path.join(audio_directory, audio_file)
    
    # Define output plot file paths
    spectrogram_plot_path = os.path.join(plot_directory, f"{os.path.splitext(audio_file)[0]}_spectrogram.png")
    time_series_plot_path = os.path.join(plot_directory, f"{os.path.splitext(audio_file)[0]}_time_series.png")
    
    # Check if the plots already exist
    if not (os.path.exists(spectrogram_plot_path) and os.path.exists(time_series_plot_path)):
        print(f"Generating plots for {audio_file}...")

        # Spectrogram plot
        plot.spectrogram(
            audio_file_path,    
            show=False,
            save_to_disk=True,
            output_file=spectrogram_plot_path
        )

        # Time-series plot
        plot.time(
            audio_file_path,
            show=False,
            save_to_disk=True,
            output_file=time_series_plot_path
        )
    else:
        print(f"Plots for {audio_file} already exist.")
