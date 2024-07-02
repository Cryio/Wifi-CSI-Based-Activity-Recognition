'''
import os
import matplotlib.pyplot as plt
import numpy as np
import collections
import serial
import csv
import time
import threading
import pyaudio
import wave

# Configuration parameters
custom_serial_port = 'COM5'
custom_baud_rate = 1000000
subcarriers = [44, 45, 46, 47]

# Deques to store amplitude and phase data
perm_amp = collections.deque(maxlen=5000)
perm_phase = collections.deque(maxlen=5000)

# Setting up the plots
plt.ion()
fig, axs = plt.subplots(3, 1, figsize=(15, 12))

# Amplitude plot
lines_amp = [axs[0].plot([], [], label=f'Subcarrier {sub}', linestyle='-', linewidth=1.5)[0] for sub in subcarriers]
axs[0].set_xlim(0, 100)
axs[0].set_ylim(0, 50)
axs[0].set_xlabel('Time', fontsize=12)
axs[0].set_ylabel('Amplitude', fontsize=12)
axs[0].set_title('Amplitude Plot', fontsize=14)
axs[0].legend(loc='upper right', fontsize=10)
axs[0].grid(True, linestyle='--', alpha=0.7)

# Phase plot
lines_phase = [axs[1].plot([], [], label=f'Subcarrier {sub}', linestyle='-', linewidth=1.5)[0] for sub in subcarriers]
axs[1].set_xlim(0, 100)
axs[1].set_ylim(-np.pi, np.pi)
axs[1].set_xlabel('Time', fontsize=12)
axs[1].set_ylabel('Phase (radians)', fontsize=12)
axs[1].set_title('Phase Plot', fontsize=14)
axs[1].legend(loc='upper right', fontsize=10)
axs[1].grid(True, linestyle='--', alpha=0.7)

# RSSI plot (Placeholder, modify as needed)
lines_rssi = [axs[2].plot([], [], label=f'RSSI Antenna {ant}', linestyle='-', linewidth=1.5)[0] for ant in ['A', 'B', 'C']]
axs[2].set_xlim(0, 100)
axs[2].set_ylim(-100, 0)
axs[2].set_xlabel('Time', fontsize=12)
axs[2].set_ylabel('RSSI', fontsize=12)
axs[2].set_title('RSSI Plot', fontsize=14)
axs[2].legend(loc='upper right', fontsize=10)
axs[2].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout(pad=3)

# Serial port setup
ser = serial.Serial(custom_serial_port, custom_baud_rate)

# CSV file setup
csv_filename = f"csi_data_{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) + f'.{int(time.time() * 1000) % 1000}'}.csv"
csv_path = os.path.join(os.getcwd(), csv_filename)

with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['CSI_DATA', 'PA', 'MAC', 'RSSI', 'Rate', 'Sig_Mode', 'MCS', 'Bandwidth', 'Smoothing',
                     'Not_Sounding', 'Aggregation', 'STBC', 'FEC_Coding', 'SGI', 'Noise_Floor', 'AMPDU_CNT',
                     'Channel', 'Secondary_Channel', 'Local_Timestamp', 'Ant', 'Sig_Len', 'RX_State',
                     'Real_Time_Set', 'Real_Timestamp', 'Len', 'CSI_Data', 'Timestamp'])

    def update_plot(amp, phase, rssi):
        x = range(100 - len(amp), 100)
        amp = np.asarray(amp)
        phase = np.asarray(phase)

        if amp.ndim == 1:
            amp = amp[:, np.newaxis]
        if phase.ndim == 1:
            phase = phase[:, np.newaxis]

        for i, sub in enumerate(subcarriers):
            y_amp = amp[:, sub] if amp.shape[1] > sub else np.zeros(100)
            lines_amp[i].set_data(x, y_amp)
            axs[0].relim()
            axs[0].autoscale_view()
            y_phase = phase[:, sub] if phase.shape[1] > sub else np.zeros(100)
            lines_phase[i].set_data(x, y_phase)
            axs[1].relim()
            axs[1].autoscale_view()

        for i in range(3):  # Assuming 3 RSSI antennas
            y_rssi = np.asarray(rssi)[:, i] if len(rssi) > 0 else np.zeros(100)
            lines_rssi[i].set_data(x, y_rssi)
        axs[2].relim()
        axs[2].autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()

    def record_audio(duration, filename):
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100

        audio = pyaudio.PyAudio()

        stream = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)

        frames = []

        print("Recording...")

        while record_audio.trigger:
            data = stream.read(CHUNK)
            frames.append(data)

        print("Recording finished.")

        stream.stop_stream()
        stream.close()
        audio.terminate()

        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

    try:
        record_audio.trigger = True

        audio_folder = "Audio_logs"
        os.makedirs(audio_folder, exist_ok=True)
        audio_filename = os.path.join(audio_folder, f"{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) + f'.{int(time.time() * 1000) % 1000}'}.wav")

        action_thread = threading.Thread(target=record_audio, args=(10, audio_filename))
        action_thread.start()

        while True:
            serial_line = ser.readline().decode().strip()
            if "CSI_DATA" in serial_line:
                all_data = serial_line.split(',')
                csi_data = all_data[25].split(" ")
                amp = []
                phase = []
                rssi = []
                if len(csi_data) > 0:
                    for c in csi_data:
                        try:
                            val = int(c)
                            amp.append(val)
                            phase.append(val)
                        except ValueError:
                            pass
                    if amp and phase:
                        rssi = [int(all_data[3]), int(all_data[4]), int(all_data[5])]  # Extract RSSI values
                        perm_amp.append(amp)
                        perm_phase.append(phase)
                        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) + f'.{int(time.time() * 1000) % 1000}'
                        timestamp = timestamp.replace(':', '-')
                        timestamp = timestamp.replace('.', '-')
                        writer.writerow(all_data + [timestamp])
                        update_plot(perm_amp, perm_phase, rssi)
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Exiting gracefully.")
    except Exception as e:
        print("An error occurred:", e)
    finally:
        ser.close()
        record_audio.trigger = False
        action_thread.join()
        print("CSV file saved at:", csv_path)
'''

import os
import matplotlib.pyplot as plt
import numpy as np
import collections
import serial
import csv
import time 
import threading
import pyaudio
import wave

custom_serial_port = 'COM5'
custom_baud_rate = 1000000 

subcarriers = [44, 45, 46]
perm_amp = collections.deque(maxlen=5000)
perm_phase = collections.deque(maxlen=5000)

plt.ion()
fig, axs = plt.subplots(2, 1, figsize=(12, 8))

lines_amp = [axs[0].plot([], [], label=f'Subcarrier {sub}', linestyle='-', linewidth=1.5)[0] for sub in subcarriers]
axs[0].set_xlim(0, 100)
axs[0].set_ylim(0, 50)
axs[0].set_xlabel('Time', fontsize=12)
axs[0].set_ylabel('Amplitude', fontsize=12)
axs[0].set_title('Amplitude Plot', fontsize=14)
axs[0].legend(loc='upper right', fontsize=10)
axs[0].grid(True, linestyle='--', alpha=0.7)

lines_phase = [axs[1].plot([], [], label=f'Subcarrier {sub}', linestyle='-', linewidth=1.5)[0] for sub in subcarriers]
axs[1].set_xlim(0, 100)
axs[1].set_ylim(-np.pi, np.pi)
axs[1].set_xlabel('Time', fontsize=12)
axs[1].set_ylabel('Phase (radians)', fontsize=12)
axs[1].set_title('Phase Plot', fontsize=14)
axs[1].legend(loc='upper right', fontsize=10)
axs[1].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout(pad=3)

ser = serial.Serial(custom_serial_port, custom_baud_rate)

csv_filename = f"csi_data_{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) + f'.{int(time.time() * 1000) % 1000}'}.csv"
csv_path = os.path.join(os.getcwd(), csv_filename)

with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['CSI_DATA', 'PA', 'MAC', 'RSSI', 'Rate', 'Sig_Mode', 'MCS', 'Bandwidth', 'Smoothing',
                     'Not_Sounding', 'Aggregation', 'STBC', 'FEC_Coding', 'SGI', 'Noise_Floor', 'AMPDU_CNT',
                     'Channel', 'Secondary_Channel', 'Local_Timestamp', 'Ant', 'Sig_Len', 'RX_State',
                     'Real_Time_Set', 'Real_Timestamp', 'Len', 'CSI_Data', 'Timestamp'])

    def update_plot(amp, phase):
        x = range(100 - len(amp), 100)
        for i, sub in enumerate(subcarriers):
            y_amp = np.asarray(amp)[:, sub]
            if len(y_amp) == 0:
                y_amp = np.zeros(100)
            lines_amp[i].set_data(x, y_amp)
            axs[0].relim()
            axs[0].autoscale_view()
            y_phase = np.asarray(phase)[:, sub]
            if len(y_phase) == 0:
                y_phase = np.zeros(100)
            lines_phase[i].set_data(x, y_phase)
            axs[1].relim()
            axs[1].autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()

    def record_audio(duration, filename):
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100

        audio = pyaudio.PyAudio()

        stream = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)

        frames = []

        print("Recording...")

        while record_audio.trigger:
            data = stream.read(CHUNK)
            frames.append(data)

        print("Recording finished.")

        stream.stop_stream()
        stream.close()
        audio.terminate()

        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

    try:
        record_audio.trigger = True

        audio_folder = "Audio_logs"
        os.makedirs(audio_folder, exist_ok=True)
        audio_filename = os.path.join(audio_folder, f"{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) + f'.{int(time.time() * 1000) % 1000}'}.wav")

        action_thread = threading.Thread(target=record_audio, args=(10, audio_filename))
        action_thread.start()

        while True:
            serial_line = ser.readline().decode().strip()
            if "CSI_DATA" in serial_line:
                all_data = serial_line.split(',')
                csi_data = all_data[25].split(" ")
                amp = []
                phase = []
                if len(csi_data) > 0:
                    for c in csi_data:
                        try:
                            val = int(c)
                            amp.append(val)
                            phase.append(val)
                        except ValueError:
                            pass
                    if amp and phase:
                        global timestamp
                        perm_amp.append(amp)
                        perm_phase.append(phase)
                        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) + f'.{int(time.time() * 1000) % 1000}'
                        timestamp = timestamp.replace(':', '-') 
                        timestamp = timestamp.replace('.', '-') 
                        writer.writerow(all_data + [timestamp]) 
                        update_plot(perm_amp, perm_phase)
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Exiting gracefully.")
    except Exception as e:
        print("An error occurred:", e)
    finally:
        ser.close()
        record_audio.trigger = False
        action_thread.join()
        print("CSV file saved at:", csv_path)