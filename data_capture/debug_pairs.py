import os
import glob
import re

csi_dir = 'csi_data/data/csi'
audio_dir = 'csi_data/data/audio'

csi_files = glob.glob(os.path.join(csi_dir, "csi_data_*.csv"))
audio_files = os.listdir(audio_dir)

print(f"CSI files: {len(csi_files)}")
print(f"Audio files: {len(audio_files)}")

# Check what timestamps look like
print("\nCSI timestamps (first 3):")
for f in csi_files[:3]:
    m = re.search(r'csi_data_(.+)\.csv', os.path.basename(f))
    if m:
        print(f"  {m.group(1)}")

print("\nAudio timestamps (first 3):")
for f in audio_files[:3]:
    m = re.search(r'(.+)\.wav', f)
    if m:
        print(f"  {m.group(1)}")
