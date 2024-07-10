import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Set Seaborn style
sns.set(style="whitegrid")

def read_esp32(file):
    """Parse CSI obtained using 'ESP32-CSI-Tool'"""
    dt = {'type': str, 'role': str, 'mac': str, 'rssi': int, 'rate': int,
          'sig_mode': int, 'mcs': int, 'bandwidth': int, 'smoothing': int,
          'not_sounding': int, 'aggregation': int, 'stbc': int,
          'fec_coding': int, 'sgi': int, 'noise_floor': int, 'ampdu_cnt': int,
          'channel': int, 'secondary_channel': int, 'local_timestamp': int,
          'ant': int, 'sig_len': int, 'rx_state': int, 'real_time_set': int,
          'real_timestamp': float, 'len': int, 'csi_data': object}
    names = list(dt.keys())
    data = pd.read_csv(file, header=None, names=names, dtype=dt,
                       usecols=['csi_data'], engine='c')

    # parse csi
    csi_string = ''.join(data['csi_data'].apply(lambda x: x.strip('[]')))
    csi = np.fromstring(csi_string, dtype=int, sep=' ').reshape(len(data), -1)
    csi = csi[:, 1::2] + csi[:, ::2] * 1.j

    return csi

def animate_plot(csifile):
    csi = read_esp32(csifile)
    num_frames, num_subcarriers = csi.shape
    dframe = np.zeros((num_frames, num_subcarriers))
    y2 = np.zeros((num_frames, num_subcarriers))

    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    lineL, = axs[0].plot([], [], color='b', linewidth=1)
    lineR, = axs[1].plot([], [], color='r', linewidth=1)

    axs[0].set_title('Average Amplitude of CSI Data', fontsize=16)
    axs[0].set_xlabel('Frame Index', fontsize=14)
    axs[0].set_ylabel('Amplitude', fontsize=14)

    axs[1].set_title('Average Phase of CSI Data', fontsize=16)
    axs[1].set_xlabel('Frame Index', fontsize=14)
    axs[1].set_ylabel('Phase (radians)', fontsize=14)

    for frame in range(num_frames):
        dframe[frame] = time.perf_counter()
        if frame > 0:
            y2[frame] = 1000 * np.diff(dframe[frame-1:frame+1], axis=0)
        else:
            y2[frame] = 0

        y2_avg = np.nanmean(y2[:frame+1], axis=0)

        # Average amplitude and phase of all subcarriers
        avg_amplitude = np.abs(csi[:frame+1, :]).mean(axis=0)
        avg_phase = np.angle(csi[:frame+1, :]).mean(axis=0)

        lineL.set_data(range(num_subcarriers), avg_amplitude)
        lineR.set_data(range(num_subcarriers), avg_phase)

        axs[0].relim()
        axs[0].autoscale_view(True, True) 

        axs[1].relim()
        axs[1].autoscale_view(True, True) 

        plt.pause(0.001)  # Increase the pause duration to 0.1 seconds

    plt.show()

if __name__ == '__main__':
    csifile = r'D:\Projects\WIFI Research\April\my-experiment-file2.csv' 
    animate_plot(csifile)
