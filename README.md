# WiFi-CSI-Based Human Activity Recognition (HAR)

**Leveraging WiFi Signals for Non-Intrusive Human Activity Recognition**

This project focuses on using WiFi Channel State Information (CSI) to recognize human activities without the privacy concerns associated with camera-based methods. Using the ubiquity of WiFi devices, we aim to develop a non-intrusive monitoring system that collects CSI data for various human activities, processes it, and uses these inputs in a 2D Convolutional Neural Network (CNN) for activity recognition.

![Sample Spectrogram](2024-09-25_15-55-29.664_spectrogram.png)
*Figure 1: Mel-Spectrogram of captured audio for activity recognition.*

---

## Overview

The Internet of Things (IoT) has significantly advanced Human Activity Recognition (HAR), especially in smart home environments. Traditionally, vision-based methods such as cameras have raised privacy concerns when used for continuous monitoring. This project bypasses these concerns by leveraging **WiFi Channel State Information (CSI)** to monitor and classify human activities.

The system works by capturing and processing CSI and audio data from WiFi devices (e.g., Raspberry Pi or ESP32), converting this data into an appropriate format for deep learning models.

---

## Key Features

- **Non-Intrusive Monitoring**: Utilizes CSI data from everyday WiFi devices for activity recognition.
- **IoT Device Support**: Data collection is done via **Raspberry Pi 4** and/or **ESP32**, which ensures compatibility with affordable IoT devices.
- **Deep Learning Integration**: The CSI data is converted into images such as **amplitude/phase heatmaps** or **mel-spectrograms** for input into a **2D Convolutional Neural Network (CNN)**.
- **Multi-Modal Data Fusion**: Optionally combines **audio data** with CSI information for improved recognition accuracy.

---

## Methodology

### 1. **CSI Data Collection**

We use a Raspberry Pi or ESP32 device to capture CSI data over a WiFi network. The collected data includes amplitude and phase information from the WiFi signals, which is processed and transformed into time-series representations.

The `CSICapture` class handles real-time data collection from the ESP32 or other devices, saving this data to CSV files. For example, a typical capture session saves the CSI data like:

```csv
CSI_DATA, PA, MAC, RSSI, Rate, Sig_Mode, MCS, Bandwidth, Smoothing, ..., Timestamp
```

This information is then parsed and used for feature extraction.

### 2. **Audio Data Collection**

Alongside CSI data, we capture audio recordings corresponding to human activities. The `AudioCapture` class ensures synchronized collection with the CSI data, as shown below:

![Spectrogram](2024-09-25_15-55-29.664_spectrogram.png)

*Figure 2: Audio Mel-Spectrogram, used as part of the multi-modal HAR pipeline.*

Audio data is processed to extract features like **Mel-Spectrograms** for input into the 2D CNN, providing complementary information to the CSI data.

### 3. **Data Processing**

- **Amplitude/Phase Heatmap**: The raw CSI data is transformed into visual representations that highlight changes in signal properties over time.
- **Mel-Spectrogram (for audio data)**: Extracts time-frequency information, showing frequency patterns in the sound corresponding to various activities.
  
  The following image is an example of a generated **mel-spectrogram**:

  ![Mel-Spectrogram](2024-09-25_15-55-29.664_spectrogram.png)
  *Figure 3: Sample Mel-Spectrogram generated from recorded audio for HAR.*

### 4. **Model Training**

We use a **2D Convolutional Neural Network (CNN)** to process the CSI-derived images and audio spectrograms. The network is trained to classify different human activities based on these images. The CSI data is treated as a spatio-temporal representation, while the spectrogram captures audio features.

### 5. **Synchronization of Captures**

The synchronized collection of CSI and audio data ensures that both types of information correspond to the same activity. Timestamping is key to ensure that both the CSI and audio streams are aligned, as demonstrated in the capturing code provided in this repository.

---

## Dataset

The dataset comprises synchronized **CSI** and **audio** captures for various human activities. Each activity is recorded with corresponding CSI and audio data, and the data is transformed into images used as inputs to the CNN.

For example, the **Tabla Solo dataset** provides audio information which can be utilized for activity recognition alongside the CSI data. Supporting data:

- [Tabla Solo dataset - ISMIR 2015](https://www.upf.edu/web/mtg/tabla-solo)

---

## Usage

To run the project:

1. **Set up the data capture environment** using a Raspberry Pi or ESP32 device.
2. **Run the synchronized CSI and audio capturing scripts** to collect the data for the activity you wish to recognize.
3. **Convert the captured data into heatmaps or mel-spectrograms** using the provided code.
4. **Train the 2D CNN** with the generated data images for human activity recognition.

---

## Results (Ongoing Work)

- The project is currently in progress, and the initial results show promising recognition of activities such as **walking**, **sitting**, and **standing** using a combination of CSI and audio data.
- **Accuracy and F1 Score** metrics will be updated as the training progresses on the full dataset.

---

## Future Work

- **Real-Time Recognition**: Implementing real-time CSI and audio processing for activity detection on edge devices like the Raspberry Pi.
- **Multimodal Data Fusion**: Further experiments to fuse CSI data with additional environmental sensors, such as motion or light sensors, for more accurate activity detection.

---

## References

1. S. Gupta, A. Srinivasamurthy, M. Kumar, H. A. Murthy, X. Serra. *Discovery of Syllabic Percussion Patterns in Tabla Solo Recordings*. In Proceedings of the 16th International Society for Music Information Retrieval Conference (ISMIR), 2015.
