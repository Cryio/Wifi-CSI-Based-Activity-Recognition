import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from PIL import Image
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN

# Function to create heatmap for CSI data
def create_csi_heatmap(csi_data, image_file):
    fig, ax = plt.subplots()
    cax = ax.imshow(np.abs(csi_data), aspect='auto', cmap='viridis')  # Plot magnitude
    fig.colorbar(cax)
    plt.title("CSI Data Heatmap")
    plt.savefig(image_file)
    plt.close(fig)

# Function to create spectrogram from audio data
def create_spectrogram(audio_file, image_file):
    y, sr = librosa.load(audio_file)
    ms = librosa.feature.melspectrogram(y=y, sr=sr)
    log_ms = librosa.power_to_db(ms, ref=np.max)
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    librosa.display.specshow(log_ms, sr=sr, x_axis='time', y_axis='mel')
    fig.colorbar(ax.collections[0], ax=ax, format="%+2.0f dB")
    plt.title("Mel Spectrogram")
    plt.savefig(image_file)
    plt.close(fig)

# Function to parse complex numbers from the CSV file
def parse_complex_line(line):
    values = line.strip().split(',')
    return [complex(val.strip().replace('j', 'j')) for val in values]

# Function to process CSI and audio data together
def process_csi_and_audio(csi_input_path, audio_input_path, output_path):
    for csi_file in os.listdir(csi_input_path):
        if csi_file.endswith('.csv'):
            try:
                with open(os.path.join(csi_input_path, csi_file), 'r') as file:
                    csi_data = [parse_complex_line(line) for line in file.readlines()]
                    csi_data = np.array(csi_data)  # Convert to NumPy array
                    if csi_data.size == 0:
                        print(f"No data found in {csi_file}. Skipping.")
                        continue
            except Exception as e:
                print(f"Error reading {csi_file}: {e}")
                continue
            
            # Get sorted file lists
            audio_files = sorted([f for f in os.listdir(audio_input_path) if f.endswith('.wav')])

            # Ensure same number of CSI and audio files
            for audio_file in audio_files:
                print(f"Processing: {csi_file} and {audio_file}")

                # Load CSI data and create heatmap
                csi_output = os.path.join(output_path, csi_file.replace('.csv', '_csi.png'))
                create_csi_heatmap(csi_data, csi_output)

                # Load audio data
                audio_file_path = os.path.join(audio_input_path, audio_file)
                audio_output = os.path.join(output_path, audio_file.replace('.wav', '_audio.png'))

                # Create audio spectrogram
                create_spectrogram(audio_file_path, audio_output)

# Function to load image data from generated CSI heatmaps and audio spectrograms
def load_images(image_paths, target_size):
    images = []
    for path in image_paths:
        img = Image.open(path).convert('L')  # Convert to grayscale
        img = img.resize(target_size)
        images.append(np.array(img))
    return np.array(images)

# Create a multi-input CNN model for classification
def create_combined_model():
    # Input for CSI data (e.g., 64x64 heatmap)
    csi_input = layers.Input(shape=(64, 64, 1))
    csi_branch = layers.Conv2D(32, (3, 3), activation='relu')(csi_input)
    csi_branch = layers.MaxPooling2D((2, 2))(csi_branch)
    csi_branch = layers.Flatten()(csi_branch)

    # Input for audio data (e.g., 128x128 spectrogram)
    audio_input = layers.Input(shape=(128, 128, 1))
    audio_branch = layers.Conv2D(32, (3, 3), activation='relu')(audio_input)
    audio_branch = layers.MaxPooling2D((2, 2))(audio_branch)
    audio_branch = layers.Flatten()(audio_branch)

    # Combine both branches
    combined = layers.concatenate([csi_branch, audio_branch])
    
    # Fully connected layer
    combined = layers.Dense(64, activation='relu')(combined)
    output = layers.Dense(10, activation='softmax')(combined)  # Assuming 10 classes

    # Build model
    model = models.Model(inputs=[csi_input, audio_input], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Example of using this to process data and train the model
def main(csi_input_path, audio_input_path, output_path):
    # Step 1: Generate CSI heatmaps and audio spectrograms
    process_csi_and_audio(csi_input_path, audio_input_path, output_path)
    
    # Step 2: Prepare file paths for images
    csi_image_paths = sorted([os.path.join(output_path, f) for f in os.listdir(output_path) if '_csi.png' in f])
    audio_image_paths = sorted([os.path.join(output_path, f) for f in os.listdir(output_path) if '_audio.png' in f])

    # Step 3: Load CSI heatmaps and audio spectrograms as image data
    csi_images = load_images(csi_image_paths, target_size=(64, 64))
    audio_images = load_images(audio_image_paths, target_size=(128, 128))

    # Step 4: Prepare labels (dummy example: 0-9 for 10 classes)
    labels = np.array([i % 10 for i in range(len(csi_images))])  # Example labels
    labels = tf.keras.utils.to_categorical(labels, num_classes=10)

    # Step 5: Split data into train and test sets
    (csi_train, csi_test, audio_train, audio_test, y_train, y_test) = train_test_split(
        csi_images, audio_images, labels, test_size=0.2, random_state=42
    )

    # Step 6: Instantiate and train the model
    model = create_combined_model()
    model.fit([csi_train, audio_train], y_train, epochs=10, validation_data=([csi_test, audio_test], y_test))

    # Step 7: Evaluate the model
    loss, accuracy = model.evaluate([csi_test, audio_test], y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Run the main function with the appropriate paths
if __name__ == "__main__":
    csi_input_path = 'data_capture/csi_combined'  # Path to CSI CSV files
    audio_input_path = 'data_capture/audio'  # Path to audio WAV files
    output_path = 'data_capture/CNN_images'  # Path to save generated images

    main(csi_input_path, audio_input_path, output_path)

# Clustering Example
def clustering_example():
    # Generate synthetic data
    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

    # K-means Clustering
    kmeans = KMeans(n_clusters=4)
    y_kmeans = kmeans.fit_predict(X)

    # DBSCAN Clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    y_dbscan = dbscan.fit_predict(X)

    # Plotting results
    plt.figure(figsize=(14, 6))

    # K-means plot
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=200, alpha=0.75, marker='X')
    plt.title('K-means Clustering')

    # DBSCAN plot
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=y_dbscan, s=50, cmap='viridis')
    plt.title('DBSCAN Clustering')

    plt.show()

# Run the clustering example
if __name__ == "__main__":
    clustering_example()
