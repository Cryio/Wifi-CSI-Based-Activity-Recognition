import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, signal
import os
from datetime import datetime
import torch
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler

class CSIWindowAnalyzer:
    def __init__(self, window_size=100, overlap=0.5, sampling_rate=None):
        """
        Initialize the CSI Window Analyzer.
        
        Args:
            window_size (int): Number of samples in each window
            overlap (float): Overlap between consecutive windows (0 to 1)
            sampling_rate (float): Sampling rate of the CSI data in Hz
        """
        self.window_size = window_size
        self.overlap = overlap
        self.sampling_rate = sampling_rate
        self.hop_size = int(window_size * (1 - overlap))
        
    def extract_windows(self, data):
        """
        Extract windows from the input data with specified overlap.
        
        Args:
            data (np.ndarray): Input CSI data array
            
        Returns:
            np.ndarray: Array of windows
        """
        # Calculate number of windows
        n_samples = len(data)
        n_windows = ((n_samples - self.window_size) // self.hop_size) + 1
        
        # Extract windows
        windows = np.array([
            data[i * self.hop_size : i * self.hop_size + self.window_size]
            for i in range(n_windows)
        ])
        
        return windows
    
    def compute_window_features(self, window):
        """
        Compute statistical and frequency-domain features for a single window.
        
        Args:
            window (np.ndarray): Input window data
            
        Returns:
            dict: Dictionary containing computed features
        """
        # Time-domain features
        features = {
            'mean': np.mean(window, axis=0),
            'std': np.std(window, axis=0),
            'max': np.max(window, axis=0),
            'min': np.min(window, axis=0),
            'median': np.median(window, axis=0),
            'skewness': stats.skew(window, axis=0),
            'kurtosis': stats.kurtosis(window, axis=0),
            'rms': np.sqrt(np.mean(np.square(window), axis=0)),
            'peak_to_peak': np.ptp(window, axis=0),
            'crest_factor': np.max(np.abs(window), axis=0) / np.sqrt(np.mean(np.square(window), axis=0))
        }
        
        # Frequency-domain features
        if self.sampling_rate:
            for i in range(window.shape[1]):  # For each subcarrier
                # Compute FFT
                yf = fft(window[:, i])
                xf = fftfreq(self.window_size, 1/self.sampling_rate)
                
                # Only take positive frequencies
                pos_mask = xf > 0
                yf = np.abs(yf[pos_mask])
                xf = xf[pos_mask]
                
                # Frequency features
                features[f'dominant_freq_{i}'] = xf[np.argmax(yf)]
                features[f'spectral_centroid_{i}'] = np.sum(xf * yf) / np.sum(yf)
                features[f'spectral_bandwidth_{i}'] = np.sqrt(np.sum(((xf - features[f'spectral_centroid_{i}'])**2) * yf) / np.sum(yf))
        
        return features
    
    def analyze_csi_file(self, file_path, output_dir=None):
        """
        Analyze CSI data from a file using windowing.
        
        Args:
            file_path (str): Path to the CSI data file
            output_dir (str): Directory to save the results
            
        Returns:
            tuple: (windows, features, timestamps)
        """
        # Create output directory if needed
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Load CSI data
        csi_data = pd.read_csv(file_path)
        timestamps = []
        
        # Parse timestamps if available
        if 'Timestamp' in csi_data.columns:
            timestamps = [
                datetime.strptime(ts, "%Y-%m-%d_%H-%M-%S-%f")
                for ts in csi_data['Timestamp']
            ]
            
            # Calculate sampling rate if not provided
            if not self.sampling_rate and len(timestamps) > 1:
                time_diffs = np.diff([ts.timestamp() for ts in timestamps])
                self.sampling_rate = 1 / np.mean(time_diffs)
                print(f"Calculated sampling rate: {self.sampling_rate:.2f} Hz")
        
        # Extract CSI data
        if 'CSI_Data' in csi_data.columns:
            # Parse CSI string format [value1 value2 ...]
            csi_values = []
            for row in csi_data['CSI_Data']:
                values = row.strip('[]').split()
                csi_values.append([float(v) for v in values])
            csi_matrix = np.array(csi_values)
        else:
            # Check for subcarrier columns
            subcarrier_cols = [col for col in csi_data.columns if col.startswith('subcarrier_')]
            if subcarrier_cols:
                csi_matrix = csi_data[subcarrier_cols].values
            else:
                # Assume all columns except Timestamp are CSI data
                csi_matrix = csi_data.drop('Timestamp', axis=1, errors='ignore').values
        
        # Normalize CSI data
        scaler = StandardScaler()
        csi_matrix = scaler.fit_transform(csi_matrix)
        
        # Extract windows
        windows = self.extract_windows(csi_matrix)
        
        # Compute features for each window
        all_features = []
        for window in windows:
            features = self.compute_window_features(window)
            all_features.append(features)
        
        # Save results if output directory is provided
        if output_dir:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # Save window features
            for feature_name in all_features[0].keys():
                feature_values = np.array([f[feature_name] for f in all_features])
                feature_file = os.path.join(output_dir, f"{base_name}_{feature_name}.csv")
                np.savetxt(feature_file, feature_values, delimiter=',')
            
            # Plot features
            self.plot_window_features(all_features, base_name, output_dir)
            
            # Plot spectrograms
            self.plot_spectrograms(windows, base_name, output_dir)
        
        return windows, all_features, timestamps
    
    def plot_window_features(self, features, base_name, output_dir):
        """
        Plot window features.
        
        Args:
            features (list): List of feature dictionaries
            base_name (str): Base name for the output files
            output_dir (str): Directory to save the plots
        """
        feature_names = list(features[0].keys())
        n_features = len(feature_names)
        
        # Create a subplot for each feature
        fig, axes = plt.subplots(n_features, 1, figsize=(12, 4*n_features))
        fig.suptitle(f'Window Features - {base_name}')
        
        for i, feature_name in enumerate(feature_names):
            feature_values = np.array([f[feature_name] for f in features])
            
            if n_features > 1:
                ax = axes[i]
            else:
                ax = axes
                
            # Plot feature values over windows
            ax.plot(feature_values)
            ax.set_title(f'{feature_name} over Windows')
            ax.set_xlabel('Window Index')
            ax.set_ylabel(feature_name)
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base_name}_window_features.png"))
        plt.close()
    
    def plot_spectrograms(self, windows, base_name, output_dir):
        """
        Plot spectrograms for each subcarrier.
        
        Args:
            windows (np.ndarray): Array of windows
            base_name (str): Base name for the output files
            output_dir (str): Directory to save the plots
        """
        if not self.sampling_rate:
            return
            
        n_subcarriers = windows.shape[2]
        fig, axes = plt.subplots(n_subcarriers, 1, figsize=(12, 4*n_subcarriers))
        fig.suptitle(f'Spectrograms - {base_name}')
        
        for i in range(n_subcarriers):
            if n_subcarriers > 1:
                ax = axes[i]
            else:
                ax = axes
                
            # Compute spectrogram
            f, t, Sxx = signal.spectrogram(
                windows[:, :, i].flatten(),
                fs=self.sampling_rate,
                nperseg=self.window_size,
                noverlap=self.window_size - self.hop_size
            )
            
            # Plot spectrogram
            im = ax.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
            ax.set_title(f'Subcarrier {i+1} Spectrogram')
            ax.set_ylabel('Frequency [Hz]')
            ax.set_xlabel('Time [sec]')
            plt.colorbar(im, ax=ax, label='Intensity [dB]')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base_name}_spectrograms.png"))
        plt.close()

def process_directory(input_dir, output_dir, window_size=100, overlap=0.5):
    """
    Process all CSI files in a directory.
    
    Args:
        input_dir (str): Input directory containing CSI files
        output_dir (str): Output directory for results
        window_size (int): Window size in samples
        overlap (float): Window overlap ratio (0 to 1)
    """
    analyzer = CSIWindowAnalyzer(window_size=window_size, overlap=overlap)
    
    # Process each CSV file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            print(f"Processing {filename}...")
            file_path = os.path.join(input_dir, filename)
            
            # Create output subdirectory for this file
            file_output_dir = os.path.join(output_dir, os.path.splitext(filename)[0])
            os.makedirs(file_output_dir, exist_ok=True)
            
            # Analyze file
            analyzer.analyze_csi_file(file_path, file_output_dir)

if __name__ == "__main__":
    # Example usage
    input_directory = "data_capture/csi/top_csi_filtered"  # Using the filtered CSI data directory
    output_directory = "data_capture/window_analysis"
    
    # Process all files with 100-sample windows and 50% overlap
    process_directory(input_directory, output_directory, window_size=100, overlap=0.5) 