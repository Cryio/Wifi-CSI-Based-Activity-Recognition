import os
import pyaudio
import wave
import csv
import time
import threading

class AudioRecorder:
    def __init__(self, duration, audio_folder="Audio_logs"):
        self.duration = duration
        self.audio_folder = audio_folder
        self.trigger = threading.Event()
        os.makedirs(audio_folder, exist_ok=True)  # Create Audio_logs folder if it doesn't exist
        self.csv_filename = os.path.join(audio_folder, "timestamps.csv")

    def record_audio(self, filename):
        CHUNK = 1024  # Number of frames per buffer
        FORMAT = pyaudio.paInt16  # Audio format (16 bits per sample)
        CHANNELS = 1  # Single channel for mono recording
        RATE = 44100  # Sample rate

        audio = pyaudio.PyAudio()

        # List available audio devices
        print("Available audio devices:")
        for i in range(audio.get_device_count()):
            dev = audio.get_device_info_by_index(i)
            print(f"  {i}: {dev['name']}")

        # Use default input device or specify device index
        device_index = None  # Set to None to use default device

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

        print("Recording...")

        for i in range(0, int(RATE / CHUNK * self.duration)):
            try:
                data = stream.read(CHUNK)
                frames.append(data)
                print(f"Captured frame {i+1}/{int(RATE / CHUNK * self.duration)}")
            except IOError as e:
                print(f"Error reading audio stream: {e}")
                break

        print("Recording finished.")

        stream.stop_stream()
        stream.close()
        audio.terminate()

        # Save audio to file
        if frames:
            wf = wave.open(filename, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
        else:
            print("No audio data captured.")

    def write_to_csv(self, timestamp, audio_filename):
        with open(self.csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, audio_filename])

    def perform_action(self):
        print("Action started...")
        # Simulate action by sleeping for duration of the audio recording
        time.sleep(self.duration)
        print("Action completed.")
        self.trigger.clear()  # Clear trigger to stop recording

    def recording_thread(self):
        while self.trigger.is_set():
            # Get current timestamp
            timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) + f'.{int(time.time() * 1000) % 1000}'
            timestamp = timestamp.replace(':', '-')  # Replace colons with dashes
            timestamp = timestamp.replace('.', '-')  # Replace periods with dashes
            audio_filename = os.path.join(self.audio_folder, f"{timestamp}.wav")

            # Record audio
            self.record_audio(audio_filename)

            # Write timestamp and file location to CSV file
            self.write_to_csv(timestamp, audio_filename)

def main():
    audio_duration = 10  # Duration of audio recording in seconds
    recorder = AudioRecorder(audio_duration)
    
    # Set trigger to True to start recording
    recorder.trigger.set()
    
    # Start the action in a separate thread
    action_thread = threading.Thread(target=recorder.perform_action)
    action_thread.start()

    # Start recording in the background
    recording_thread = threading.Thread(target=recorder.recording_thread)
    recording_thread.start()

    # Wait for action thread to finish
    action_thread.join()
    recording_thread.join()

if __name__ == "__main__":
    main()
