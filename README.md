# WiFi-CSI-Based Activity Recognition

Non-intrusive human activity recognition using WiFi Channel State Information (CSI), synchronized audio, and deep learning — no cameras, no wearables.

<div align="center">
  <img src="data_capture/csi_plot/csi_data_2024-09-25_22-01-04.590_amp_all_subcarriers.png?raw=true" width="45%" alt="CSI amplitude — all subcarriers">
  <img src="data_capture/csi_plot/csi_data_2024-09-25_16-27-42.805_heatmap.png?raw=true" width="45%" alt="CSI amplitude + phase heatmap">
</div>
<div align="center">
  <img src="data_capture/audio_plot/2024-09-25_22-01-04.590_spectrogram.png?raw=true" width="45%" alt="Mel-spectrogram of synchronized audio">
  <img src="data_capture/audio_plot/2024-09-25_22-01-04.590_time_series.png?raw=true" width="45%" alt="Audio time-series">
</div>
<div align="center"><em>Left: CSI amplitude heatmap. Right: Synchronized audio mel-spectrogram.</em></div>

## Publication

> **Rai, M. et al.** — *BeatWave: WiFi CSI-Based Human Activity Recognition with Multi-Modal Audio Correlation* (COMSNETS 2026)
> [Download PDF](https://github.com/Cryio/Wifi-CSI-Based-Activity-Recognition/raw/main/m2633-rai%20final.pdf)

## Overview

Traditional activity recognition using cameras raises privacy concerns for continuous indoor monitoring. This project replaces cameras with **WiFi CSI** — the amplitude and phase information already present in every WiFi packet — combined with a synchronized microphone to classify human activities.

An ESP32 captures CSI at 100 Hz over a standard WiFi link and sends it over USB serial at 1 Mbaud. A microphone records in parallel. Both streams are timestamped and aligned, then processed through a pipeline that produces CSI heatmaps and audio mel-spectrograms. Three deep learning models have been developed against this data:

| Model | File | Framework | Purpose |
|-------|------|-----------|---------|
| CSI→Audio Transformer | `data_capture/model.py` | PyTorch | Translates CSI time-series into audio waveforms |
| WGAN-GP | `data_capture/train_gan.py` | PyTorch | CSI-conditioned audio generation / augmentation |
| Multi-input CNN | `data_capture/CNN_model.py` | TensorFlow/Keras | Activity classification from CSI + audio images |

## Repository Structure

```
data_capture/
├── record_both.py             # Synchronized CSI + audio capture
├── csi_to_matrix.py           # Raw CSV → amplitude/phase matrix CSVs
├── wav_to_matrix.py           # WAV → audio matrix CSV
├── high_filter.py             # Bandpass filter on CSI amplitude
├── select_top_subcarriers.py  # Select top-8 most-variant subcarriers
├── model.py                   # CSI→Audio Transformer (PyTorch)
├── train_gan.py               # WGAN-GP trainer (PyTorch)
├── CNN_model.py               # Multi-input CNN (TensorFlow/Keras)
├── inference.py               # Run inference with trained models
├── csi/                       # Raw CSI CSV files
├── audio/                     # Raw audio WAV files
├── csi_amplitude/             # Per-subcarrier amplitude matrices
├── csi_phase/                 # Per-subcarrier phase matrices
├── csi_combined/              # Amplitude + phase combined
├── audio_matrix/              # Audio matrices
├── csi_plot/                  # CSI heatmap images
└── audio_plot/                # Mel-spectrogram images
```

## Quick Start

### Requirements

```bash
pip install -r data_capture/requirements.txt
```

Python packages required: `pyserial`, `pyaudio`, `numpy`, `scipy`, `librosa`, `matplotlib`, `torch`, `tensorflow`, `scikit-learn`, `soundfile`.

### 1. Flash ESP32 Firmware

Flash the [ESP32 CSI Toolkit](https://github.com/StevenMHernandez/esp32-csi-tool) (`active_sta` mode) using ESP-IDF v4.3. Full instructions: [Flashing Firmware](https://github.com/Cryio/Wifi-CSI-Based-Activity-Recognition/wiki/Flashing-Firmware).

### 2. Capture Data

```bash
cd data_capture
python record_both.py
```

Captures CSI from `/dev/ttyUSB0` at 1 Mbaud and audio from the default microphone simultaneously for 10 seconds, saving timestamped CSV and WAV files.

### 3. Process Data

```bash
python csi_to_matrix.py           # Raw CSI → amplitude + phase matrices
python wav_to_matrix.py           # WAV → audio matrix
python high_filter.py             # Bandpass filter CSI amplitude
python select_top_subcarriers.py  # Keep top-8 subcarriers by variance
```

## Hardware

| Component | Details |
|-----------|---------|
| ESP32 DevKit | CSI capture via USB serial at 1 Mbaud |
| Router | Standard 2.4 GHz WiFi AP |
| Microphone | Any USB or 3.5 mm mic supported by PyAudio |
| Host PC / Raspberry Pi | Runs `record_both.py` |

## Wiki

Full documentation — hardware setup, firmware flashing, data processing pipeline, model architecture, training, inference, and troubleshooting — is in the [project wiki](https://github.com/Cryio/Wifi-CSI-Based-Activity-Recognition/wiki).

## References

See the [References](https://github.com/Cryio/Wifi-CSI-Based-Activity-Recognition/wiki/References) wiki page for the full list of academic papers and tools this project builds on.
