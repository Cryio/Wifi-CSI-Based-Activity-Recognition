import os
import pandas as pd

csi_directory = "data_capture/csi_amplitude_44th"
audio_directory = "data_capture/audio_matrix"

csi_files = sorted([f for f in os.listdir(csi_directory) if f.endswith('.csv')])
audio_files = sorted([f for f in os.listdir(audio_directory) if f.endswith('.csv')])

if len(csi_files) != len(audio_files):
    print("Mismatch in the number of CSI and audio files.")
    exit()

data = []
for csi_file, audio_file in zip(csi_files, audio_files):
    data.append({
        "csi_path": os.path.join(csi_directory, csi_file),
        "audio_path": os.path.join(audio_directory, audio_file)
    })

df = pd.DataFrame(data)

output_path = "SVM and CNN models/data_paths.csv"
df.to_csv(output_path, index=False)
print(f"Dataset CSV created at {output_path}")
