import librosa
import librosa.display
import IPython.display as ipd
import os
import numpy as np
import matplotlib.pyplot as plt

# Directory containing audio logs
directory = 'Audio_logs/'

# Print all files in the directory
for file in os.listdir(directory):
    print(file) 

# Load and play a sample audio file
sample_audio_path = os.path.join(directory, '2024-05-27_18-11-03.wav')
x, sr = librosa.load(sample_audio_path, sr=None)
ipd.display(ipd.Audio(data=x, rate=sr))

# Function to extract MFCC features
def feature_extraction(file_path):
    # Load the audio file
    x, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    # Extract MFCC features
    mfcc = np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=50).T, axis=0)
    
    return mfcc

# Dictionary to hold the features
features = {}

# Extract features from each audio file in the directory
for audio in os.listdir(directory):
    if audio.endswith('.wav'):  # Ensure the file is an audio file
        audio_path = os.path.join(directory, audio)
        features[audio_path] = feature_extraction(audio_path)

# Plot the MFCC features for each audio file
for audio_path, mfcc in features.items():
    plt.figure(figsize=(10, 4))
    plt.plot(mfcc)
    plt.title(f'MFCC for {os.path.basename(audio_path)}')
    plt.xlabel('MFCC Coefficients')
    plt.ylabel('Mean Value')
    plt.show()
