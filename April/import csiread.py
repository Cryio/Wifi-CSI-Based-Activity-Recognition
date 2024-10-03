import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn style
sns.set(style="whitegrid")

def read_esp32(file):
    """Parse CSI obtained using 'ESP32-CSI-Tool'"""
    dt = {'type': str, 'role': str, 'mac': str, 'rssi': str, 'rate': int,
          'sig_mode': int, 'mcs': int, 'bandwidth': int, 'smoothing': int,
          'not_sounding': int, 'aggregation': int, 'stbc': int,
          'fec_coding': int, 'sgi': int, 'noise_floor': int, 'ampdu_cnt': int,
          'channel': int, 'secondary_channel': int, 'local_timestamp': int,
          'ant': int, 'sig_len': int, 'rx_state': int, 'real_time_set': int,
          'real_timestamp': float, 'len': int, 'csi_data': object}
    names = list(dt.keys())
    data = pd.read_csv(file, header=None, names=names, dtype=dt, skiprows=1)

    # Parse CSI data
    csi_data = data['csi_data'].str.strip('[]').str.split(expand=True)
    csi = csi_data.astype(float).to_numpy()

    return csi

def plot_csi(csifile):
    # Read CSI data from file
    csi = read_esp32(csifile)

    # Plot amplitude
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(np.abs(csi.T), linewidth=1)
    plt.title('Amplitude of CSI Data', fontsize=16)
    plt.xlabel('Subcarrier Index', fontsize=14)
    plt.ylabel('Amplitude', fontsize=14)

    # Plot phase
    plt.subplot(2, 1, 2)
    plt.plot(np.angle(csi.T), linewidth=1)
    plt.title('Phase of CSI Data', fontsize=16)
    plt.xlabel('Subcarrier Index', fontsize=14)
    plt.ylabel('Phase (radians)', fontsize=14)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    csifile = r'D:\Projects\WIFI Research\April\my-experiment-file.csv' 
    plot_csi(csifile)
