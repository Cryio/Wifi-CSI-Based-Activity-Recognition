import os
import serial
import csv
import time
import threading
import pyaudio
import wave

# Configuration parameters for CSI capture
custom_serial_port = '/dev/ttyUSB0'
custom_baud_rate = 1000000
target_directory_csi = "data_capture/csi"
record_duration_csi = 10

# Configuration parameters for Audio capture
audio_duration = 10
audio_directory = "data_capture/audio"

# Total duration for both captures combined
total_duration = 10  # in seconds

# Ensure directories exist
os.makedirs(target_directory_csi, exist_ok=True)
os.makedirs(audio_directory, exist_ok=True)

# Function to generate a timestamp string
def generate_timestamp():
    return time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) + f'.{int(time.time() * 1000) % 1000}'

# CSI Capture class
class CSICapture:
    def __init__(self, port, baud_rate, directory, duration, stop_event, timestamp):
        self.port = port
        self.baud_rate = baud_rate
        self.directory = directory
        self.duration = duration
        self.ser = serial.Serial(self.port, self.baud_rate)
        self.csv_filename = f"csi_data_{timestamp.replace(':', '-')}.csv"
        self.csv_path = os.path.join(self.directory, self.csv_filename)
        self.stop_event = stop_event

    def start(self):
        with open(self.csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['CSI_DATA', 'PA', 'MAC', 'RSSI', 'Rate', 'Sig_Mode', 'MCS', 'Bandwidth', 'Smoothing',
                             'Not_Sounding', 'Aggregation', 'STBC', 'FEC_Coding', 'SGI', 'Noise_Floor', 'AMPDU_CNT',
                             'Channel', 'Secondary_Channel', 'Local_Timestamp', 'Ant', 'Sig_Len', 'RX_State',
                             'Real_Time_Set', 'Real_Timestamp', 'Len', 'CSI_Data', 'Timestamp'])
            start_time = time.time()
            try:
                while not self.stop_event.is_set():
                    if time.time() - start_time > self.duration:
                        break
                    if self.ser.in_waiting > 0:
                        serial_line = self.ser.readline().decode().strip()
                        if "CSI_DATA" in serial_line:
                            all_data = serial_line.split(',')
                            timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) + f'.{int(time.time() * 1000) % 1000}'
                            timestamp = timestamp.replace(':', '-').replace('.', '-')
                            writer.writerow(all_data + [timestamp])
            except KeyboardInterrupt:
                print("CSI capture interrupted. Exiting gracefully.")
            except Exception as e:
                print("An error occurred during CSI capture:", e)
            finally:
                self.ser.close()
                print("CSI CSV file saved at:", self.csv_path)

# Audio Capture class
class AudioCapture:
    def __init__(self, duration, directory, stop_event, timestamp):
        self.duration = duration
        self.directory = directory
        os.makedirs(self.directory, exist_ok=True)
        self.stop_event = stop_event
        self.audio_filename = os.path.join(self.directory, f"{timestamp.replace(':', '-')}.wav")

    def record_audio(self):
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100

        audio = pyaudio.PyAudio()
        device_index = None 

        try:
            stream = audio.open(format=FORMAT,
                                channels=CHANNELS,
                                rate=RATE,
                                input=True,
                                input_device_index=device_index,
                                frames_per_buffer=CHUNK)
        except IOError as e:
            print(f"Error opening audio stream: {e}")
            return

        frames = []
        start_time = time.time()
        while not self.stop_event.is_set():
            if time.time() - start_time > self.duration:
                break
            try:
                data = stream.read(CHUNK)
                frames.append(data)
            except IOError as e:
                print(f"Error reading audio stream: {e}")
                break

        stream.stop_stream()
        stream.close()
        audio.terminate()

        if frames:
            wf = wave.open(self.audio_filename, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            print(f"Audio file saved at: {self.audio_filename}")
        else:
            print("No audio data captured.")

    def start(self):
        self.record_audio()

# Function to start both captures
def run_captures():
    stop_event = threading.Event()
    timestamp = generate_timestamp()

    csi_capture = CSICapture(custom_serial_port, custom_baud_rate, target_directory_csi, record_duration_csi, stop_event, timestamp)
    audio_capture = AudioCapture(audio_duration, audio_directory, stop_event, timestamp)

    try:
        # Start CSI capture in a separate thread
        csi_thread = threading.Thread(target=csi_capture.start)
        csi_thread.start()

        # Start audio capture in the same thread to ensure sync
        audio_capture.start()

        # Wait for the total duration
        time.sleep(total_duration)
        stop_event.set()

        # Wait for the CSI thread to finish
        csi_thread.join()

    except KeyboardInterrupt:
        print("Process interrupted. Exiting...")
        stop_event.set()  # Ensure both threads stop
        csi_thread.join()  # Wait for the CSI thread to finish
    finally:
        # Ensure resources are cleaned up if needed
        print("Cleanup complete.")

if __name__ == "__main__":
    run_captures()
