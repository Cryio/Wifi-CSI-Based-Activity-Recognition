import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate
import os
import traceback
from datetime import datetime

def clean_complex_string(s):
    try:
        if isinstance(s, str):
            s = s.replace(' ', '').replace('*', '')
            if 'j' not in s:
                s += 'j0'
            elif s.endswith('j'):
                s += '0'

            return complex(s)
        elif isinstance(s, (int, float)):
            return complex(float(s), 0)
    
        else:
            return np.nan
    except (ValueError, TypeError):
        return np.nan

def save_cross_corr_with_details(cross_corr_matrix, csi_file, lags, corr_type):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = os.path.splitext(csi_file)[0]
    output_dir = f'data_capture/cross_correlation_{corr_type}'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{base_name}_cross_corr_{corr_type}.csv")
    
    if os.path.exists(output_file):
        print(f"File {output_file} already exists. Skipping.")
        return
    
    subcarrier_indices = np.arange(1, cross_corr_matrix.shape[0] + 1)
    
    # Create a DataFrame to save subcarrier, lag, and cross-correlation value
    data = []
    for i, subcarrier_idx in enumerate(subcarrier_indices):
        for j, lag in enumerate(lags):
            data.append([subcarrier_idx, lag, cross_corr_matrix[i, j]])
    
    df = pd.DataFrame(data, columns=['Subcarrier', 'Lag', 'Cross-Correlation'])
    df.to_csv(output_file, index=False)

    print(f"Saved {corr_type} cross-correlation details to {output_file}")

def save_cross_corr_plot(fig, csi_file, corr_type):
    plot_dir = f'data_capture/cross_correlation_{corr_type}_plot'
    os.makedirs(plot_dir, exist_ok=True)
    base_name = os.path.splitext(csi_file)[0]
    plot_file = os.path.join(plot_dir, f"{base_name}_cross_corr_{corr_type}_plot.png")
    fig.savefig(plot_file)
    print(f"Saved {corr_type} cross-correlation plot to {plot_file}")

def compute_cross_correlation(csi_data, audio_signal, lags, corr_type, csi_file):
    cross_corr_matrix = np.zeros((csi_data.shape[1], 2 * len(audio_signal) - 1))
    strong_correlations = []

    for i in range(csi_data.shape[1]):
        subcarrier_signal = csi_data[:, i]

        # Ensure subcarrier_signal is the same length as audio_signal
        if len(subcarrier_signal) != len(audio_signal):
            min_len = min(len(subcarrier_signal), len(audio_signal))
            subcarrier_signal = subcarrier_signal[:min_len]
            audio_signal = audio_signal[:min_len]

        # Check for NaNs and handle if necessary
        if np.isnan(subcarrier_signal).any() or np.isnan(audio_signal).any():
            print(f"NaNs detected in signals. Skipping this subcarrier.")
            continue

        if np.std(subcarrier_signal) != 0:
            subcarrier_signal = (subcarrier_signal - np.mean(subcarrier_signal)) / np.std(subcarrier_signal)
        else:
            print(f"Standard deviation of subcarrier {i} is zero. Skipping normalization.")

        if np.std(audio_signal) != 0:
            audio_signal = (audio_signal - np.mean(audio_signal)) / np.std(audio_signal)

        if np.isnan(audio_signal).any():
            print("NaNs detected in audio signal after normalization. Skipping processing.")
            continue

        # Compute cross-correlation
        cross_corr = correlate(subcarrier_signal, audio_signal, mode='full')

        # Normalize cross-correlation to get coefficients between -1 and +1
        normalization_factor = len(audio_signal) * np.std(subcarrier_signal) * np.std(audio_signal)
        if normalization_factor != 0:
            cross_corr /= normalization_factor
        cross_corr_matrix[i, :] = cross_corr

        # Find and log strong correlations (close to 1 or -1)
        strong_corrs = np.where(np.abs(cross_corr) >= 0.8)[0]
        for idx in strong_corrs:
            strong_correlations.append((i, lags[idx], cross_corr[idx]))

    # Save the cross-correlation matrix along with subcarrier and lag details
    save_cross_corr_with_details(cross_corr_matrix, csi_file, lags, corr_type)

    # Print out the strong correlations
    if strong_correlations:
        print(f"Strong correlations found in {csi_file}:")
        for subcarrier, lag, value in strong_correlations:
            print(f"  Subcarrier {subcarrier + 1}, Lag {lag}: {value:.3f}")
    else:
        print(f"No strong correlations found in {csi_file}.")

    # Plot each subcarrier's cross-correlation separately in a grid
    num_plots = csi_data.shape[1]
    grid_size = 8
    num_rows = int(np.ceil(num_plots / grid_size))

    fig, axs = plt.subplots(num_rows, grid_size, figsize=(15, 2.0 * num_rows))  # Adjust figsize for smaller plots
    axs = axs.flatten()

    for i in range(num_plots):
        if i < 64:
            axs[i].plot(lags, cross_corr_matrix[i, :])
            axs[i].set_xlabel('Lag', fontsize=6)
            axs[i].set_ylabel('Cross-Correlation', fontsize=6)
            axs[i].set_title(f'Subcarrier {i + 1}\nCorr: {np.max(np.abs(cross_corr_matrix[i, :])):.3f}', fontsize=6)
        else:
            axs[i].axis('off')  # Hide empty plots

        axs[i].tick_params(axis='both', which='major', labelsize=6)

    plt.tight_layout(pad=3.0, h_pad=3.0, w_pad=0.0)
    save_cross_corr_plot(fig, csi_file, corr_type)  # Save the plot
    plt.show()

# Set directories for CSI and audio data
csi_dir_amplitude = 'data_capture/csi_amplitude'
csi_dir_phase = 'data_capture/csi_phase'
audio_dir = 'data_capture/audio_matrix'

csi_files_amplitude = sorted([f for f in os.listdir(csi_dir_amplitude) if f.endswith('.csv')])
csi_files_phase = sorted([f for f in os.listdir(csi_dir_phase) if f.endswith('.csv')])
audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.csv')])

# Process amplitude data
for csi_file, audio_file in zip(csi_files_amplitude, audio_files):
    try:
        print(f"\nProcessing amplitude: {csi_file} and {audio_file}")
        
        csi_data_path = os.path.join(csi_dir_amplitude, csi_file)
        audio_data_path = os.path.join(audio_dir, audio_file)

        # Load CSI amplitude data
        csi_matrix = pd.read_csv(csi_data_path, header=None).values
        print(f"Loaded CSI amplitude data with shape: {csi_matrix.shape}")

        # Load audio data
        audio_signal = pd.read_csv(audio_data_path, header=None).values.flatten()
        print(f"Loaded audio data with shape: {audio_signal.shape}")

        min_samples = min(csi_matrix.shape[0], len(audio_signal))
        csi_matrix = csi_matrix[:min_samples, :]
        audio_signal = audio_signal[:min_samples]

        lags = np.arange(-len(audio_signal) + 1, len(audio_signal))

        # Compute cross-correlation for amplitude data
        compute_cross_correlation(csi_matrix, audio_signal, lags, 'amplitude', csi_file)

    except Exception as e:
        print(f"Error processing amplitude {csi_file} and {audio_file}: {e}")
        traceback.print_exc()

# Process phase data
for csi_file, audio_file in zip(csi_files_phase, audio_files):
    try:
        print(f"\nProcessing phase: {csi_file} and {audio_file}")
        
        csi_data_path = os.path.join(csi_dir_phase, csi_file)
        audio_data_path = os.path.join(audio_dir, audio_file)

        # Load CSI phase data
        csi_matrix = pd.read_csv(csi_data_path, header=None).values
        print(f"Loaded CSI phase data with shape: {csi_matrix.shape}")

        # Load audio data
        audio_signal = pd.read_csv(audio_data_path, header=None).values.flatten()
        print(f"Loaded audio data with shape: {audio_signal.shape}")

        min_samples = min(csi_matrix.shape[0], len(audio_signal))
        csi_matrix = csi_matrix[:min_samples, :]
        audio_signal = audio_signal[:min_samples]

        lags = np.arange(-len(audio_signal) + 1, len(audio_signal))

        # Compute cross-correlation for phase data
        compute_cross_correlation(csi_matrix, audio_signal, lags, 'phase', csi_file)

    except Exception as e:
        print(f"Error processing phase {csi_file} and {audio_file}: {e}")

        traceback.print_exc()
