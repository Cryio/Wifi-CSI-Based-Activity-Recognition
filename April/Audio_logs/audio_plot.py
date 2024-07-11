from pyAudioProcessing import plot
from pyAudioProcessing.extract_features import get_features

# Spectrogram plot
plot.spectrogram(
    r"D:\Projects\WIFI Research - Git\April\Audio_logs\2024-05-29_19-21-28.805.wav",    
    show=True,
    save_to_disk=True,
    output_file=r'D:\Projects\WIFI Research\April\Audio_logs\plots'
)

# Time-series plot
plot.time(
    r"D:\Projects\WIFI Research - Git\April\Audio_logs\2024-05-29_19-21-28.805.wav",
    show=True,
    save_to_disk=True,
    output_file=r'D:\Projects\WIFI Research - Git\April\Audio_logs\plots'
)

# Feature extraction of a single file

features = get_features(
  file=r"D:\Projects\WIFI Research - Git\April\Audio_logs\2024-05-29_19-21-28.805.wav",
  feature_names=["gfcc", "mfcc"]
)

# Feature extraction of a multiple files

'''
features = get_features(
  file_names={
    "music": [<path to audio>, <path to audio>, ..],
    "speech": [<path to audio>, <path to audio>, ..]
  },
  feature_names=["gfcc", "mfcc"]
)
'''

# or if you have a dir with  sub-folders and audios
# features = get_features(folder_path="data_samples/testing", feature_names=["gfcc", "mfcc"])

# features is a dictionary that will hold data of the following format
"""
{
  music: {file1_path: {"features": <list>, "feature_names": <list>}, ...},
  speech: {file1_path: {"features": <list>, "feature_names": <list>}, ...},
  ...
}
"""