# Wifi-CSI-Based-Activity-Recognition

Leveraging the ubiquity of WiFi devices to monitor daily activities without the privacy concerns associated with vision-based methods. Collection CSI data for several different human activities, converting the data into images used as inputs for a 2D Convolutional Neural Network (CNN). The proposed CSI-based HAR system.

## Overview

The Internet of Things (IoT) has brought significant advancements in Human Activity Recognition (HAR), particularly for smart homes. This project focuses on using WiFi Channel State Information (CSI) to recognize human activities. The method leverages the ubiquity of WiFi devices to monitor activities without the privacy concerns associated with camera-based methods.

## Key Features

- Utilizes CSI data from WiFi signals for HAR.
- Collects data using a Raspberry Pi 4.
- Converts CSI data to images for use with a 2D Convolutional Neural Network (CNN).
- Achieves high accuracy (around 95%) for recognizing seven different activities.

## Activities Recognized

The system can recognize the following activities:
- Sit down
- Stand up
- Lie down
- Run
- Walk
- Fall
- Bend

## Dataset

The dataset used in this project consists of CSI data collected for the above activities. The data is processed and converted into RGB images which serve as inputs for the CNN classifier.

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
