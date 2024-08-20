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
            return complex(s).real  # Use only the real part
        elif isinstance(s, (int, float)):
            return float(s)
        else:
            return np.nan
    except (ValueError, TypeError):
        return np.nan

def save_cross_corr_with_details(cross_corr_matrix, csi_file, lags):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = os.path.splitext(csi_file)[0]
    output_dir = 'data_capture/cross_correlation'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{base_name}_cross_corr_{timestamp}.csv")
    
    subcarrier_indices = np.arange(1, cross_corr_matrix.shape[0] + 1)
    
    # Create a DataFrame to save subcarrier, lag, and cross-correlation value
    data = []
    for i, subcarrier_idx in enumerate(subcarrier_indices):
        for j, lag in enumerate(lags):
            data.append([subcarrier_idx, lag, cross_corr_matrix[i, j]])
    
    df = pd.DataFrame(data, columns=['Subcarrier', 'Lag', 'Cross-Correlation'])
    df.to_csv(output_file, index=False)
    print(f"Saved cross-correlation details to {output_file}")

# Set directories for CSI and audio data
csi_dir = 'data_capture/csi_matrix'
audio_dir = 'data_capture/audio_matrix'

csi_files = sorted([f for f in os.listdir(csi_dir) if f.endswith('.csv')])
print("CSI Files:", csi_files)
audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.csv')])
print("Audio Files:", audio_files)

for csi_file, audio_file in zip(csi_files, audio_files):
    try:
        print(f"\nProcessing: {csi_file} and {audio_file}")
        
        csi_data_path = os.path.join(csi_dir, csi_file)
        audio_data_path = os.path.join(audio_dir, audio_file)

        # Load CSI data and convert to numeric values
        csi_matrix = pd.read_csv(csi_data_path, header=None).applymap(clean_complex_string).values
        print(f"Loaded CSI data with shape: {csi_matrix.shape}")
        print("CSI Matrix:\n", csi_matrix)

        # Load audio data
        audio_signal = pd.read_csv(audio_data_path, header=None).values.flatten()
        print(f"Loaded audio data with shape: {audio_signal.shape}")
        print("Audio Signal:\n", audio_signal)

        min_samples = min(csi_matrix.shape[0], len(audio_signal))
        csi_matrix = csi_matrix[:min_samples, :]
        audio_signal = audio_signal[:min_samples]

        # Check standard deviation for each subcarrier
        std_devs = np.std(csi_matrix, axis=0)
        print("Standard Deviations of Subcarriers:\n", std_devs)
        varied_subcarriers = np.where(std_devs != 0)[0]
        print("Indices of Subcarriers with Non-Zero Std Dev:", varied_subcarriers)

        if len(varied_subcarriers) == 0:
            print("No varying subcarriers found. Skipping processing for this file.")
            continue

        # Use only subcarriers with non-zero standard deviation
        csi_matrix_varied = csi_matrix[:, varied_subcarriers]
        print("Varied Subcarriers Matrix:\n", csi_matrix_varied)

        # Create cross-correlation matrix
        cross_corr_matrix = np.zeros((csi_matrix_varied.shape[1], 2 * len(audio_signal) - 1))
        lags = np.arange(-len(audio_signal) + 1, len(audio_signal))
        strong_correlations = []

        for i in range(csi_matrix_varied.shape[1]):
            subcarrier_signal = csi_matrix_varied[:, i]

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
            cross_corr_matrix[i, :] = cross_corr

            # Find and log strong correlations (close to 1 or -1)
            strong_corrs = np.where(np.abs(cross_corr) >= 0.8)[0]
            for idx in strong_corrs:
                strong_correlations.append((i, lags[idx], cross_corr[idx]))

        # Save the cross-correlation matrix along with subcarrier and lag details
        save_cross_corr_with_details(cross_corr_matrix, csi_file, lags)

        # Print the cross-correlation matrix
        print("Cross-Correlation Matrix:\n", cross_corr_matrix)

        # Print out the strong correlations
        if strong_correlations:
            print(f"Strong correlations found in {csi_file}:")
            for subcarrier, lag, value in strong_correlations:
                print(f"  Subcarrier {subcarrier + 1}, Lag {lag}: {value:.3f}")
        else:
            print(f"No strong correlations found in {csi_file}.")

        # Plot each subcarrier's cross-correlation separately in an 8x8 grid
        num_plots = len(varied_subcarriers)
        grid_size = 8
        num_rows = int(np.ceil(num_plots / grid_size))

        fig, axs = plt.subplots(num_rows, grid_size, figsize=(20, 2.5 * num_rows))
        axs = axs.flatten()

        for i, subcarrier_idx in enumerate(varied_subcarriers):
            axs[i].plot(lags, cross_corr_matrix[i, :])
            axs[i].set_xlabel('Lag')
            axs[i].set_ylabel('Cross-Correlation')
            axs[i].set_title(f'Subcarrier {subcarrier_idx + 1}')

        # Remove any empty subplots
        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error processing {csi_file} and {audio_file}: {e}")
        traceback.print_exc()
