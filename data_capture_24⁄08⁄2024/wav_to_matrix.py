import os
import numpy as np
import wave
import csv

def wav_to_matrix(wav_file):
    with wave.open(wav_file, 'rb') as wf:
        n_channels = wf.getnchannels()
        n_frames = wf.getnframes()

        # Read audio data
        audio_data = wf.readframes(n_frames)

        # Convert to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)

        # If the audio has more than one channel, we need to reshape
        if n_channels > 1:
            audio_array = audio_array.reshape(-1, n_channels)

        return audio_array

def save_matrix_to_csv(matrix, csv_file):
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # If the matrix is 1D (single channel), wrap each item in a list
        if matrix.ndim == 1:
            for value in matrix:
                writer.writerow([value])
        else:
            # Otherwise, write each row as it is
            for row in matrix:
                writer.writerow(row)

def convert_all_wavs_to_csv(audio_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(audio_dir):
        if filename.endswith(".wav"):
            csv_filename = filename.replace(".wav", ".csv")
            csv_path = os.path.join(output_dir, csv_filename)

            # Skip if the file has already been converted
            if os.path.exists(csv_path):
                print(f"Skipping already converted file: {csv_filename}")
                continue

            wav_path = os.path.join(audio_dir, filename)
            audio_matrix = wav_to_matrix(wav_path)

            # Save the matrix as a CSV file
            save_matrix_to_csv(audio_matrix, csv_path)
            print(f"Converted and saved: {csv_path}")

if __name__ == "__main__":
    audio_dir = 'data_capture/audio'
    output_dir = 'data_capture/audio_matrix'
    convert_all_wavs_to_csv(audio_dir, output_dir)
