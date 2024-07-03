# Wifi-CSI-Based-Activity-Recognition

Leveraging the ubiquity of WiFi devices to monitor daily activities without the privacy concerns associated with vision-based methods. Collection CSI data for several different human activities, converting the data into images used as inputs for a 2D Convolutional Neural Network (CNN). The proposed CSI-based HAR system.

## Overview

The Internet of Things (IoT) has brought significant advancements in Human Activity Recognition (HAR), particularly for smart homes. This project focuses on using WiFi Channel State Information (CSI) to recognize human activities. The method leverages the ubiquity of WiFi devices to monitor activities without the privacy concerns associated with camera-based methods.

## Key Features

- Utilizes CSI data from WiFi signals for HAR.
- Collects data using a Raspberry Pi 4 and/or ESP32.
- Converts CSI data to images for use with a 2D Convolutional Neural Network (CNN).

## Dataset

[Tabla Solo dataset](https://zenodo.org/records/1267024?token=eyJhbGciOiJIUzUxMiIsImlhdCI6MTcxOTgzOTcwOSwiZXhwIjoxNzIyMzgzOTk5fQ.eyJpZCI6IjQzZTU5YzUxLTU4MzUtNDY2YS05NTMwLTdiNjNjM2QwM2I4NSIsImRhdGEiOnt9LCJyYW5kb20iOiJmYTdiOWYzYjg0NzRlN2JhZmUxNzM2ZTQyNTI5OWMxNSJ9.dtneEpBKf1MPDqzDS4AFmSkLCOdhLybXuM6lKNulwqyctV1LvJCTURSMXhdxZuiMIIc7uSHD2P1ZYTAHWfkMJQ)

Supporting paper : [S. Gupta, A. Srinivasamurthy, M. Kumar, H. A. Murthy, X. Serra. Discovery of Syllabic Percussion Patterns in Tabla Solo Recordings. In Proc. of the 16th International Society for Music Information Retrieval Conference (ISMIR), 2015.](http://hdl.handle.net/10230/25697)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/csi-har-deep-learning.git
   cd csi-har-deep-learning
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Data Collection**:
   - Use a Raspberry Pi 4 with the Nexmon CSI tool to collect CSI data for different activities.

2. **Data Preprocessing**:
   - Convert the collected CSI data into RGB images using the provided scripts.

3. **Training the Model**:
   - Use the preprocessed data to train the 2D CNN model.
   ```bash
   python train.py --data_dir path/to/data
   ```

4. **Evaluating the Model**:
   - Evaluate the trained model on test data.
   ```bash
   python evaluate.py --model_path path/to/model --data_dir path/to/test_data
   ```

## Results

(under progress)
