import torch
import torch.nn as nn
import torchaudio
import pandas as pd
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import Resample

# Hyperparameters
LEARNING_RATE = 0.0002
BATCH_SIZE = 16
EPOCHS = 50
AUDIO_SAMPLE_RATE = 44100
DILATION_CYCLES = 8
RESIDUAL_LAYERS = 10

data_df = pd.read_csv("SVM and CNN models/data_paths.csv")

class CSIAudioDataset(Dataset):
    def __init__(self, data_df, audio_sample_rate=AUDIO_SAMPLE_RATE):
        self.data_df = data_df
        self.audio_sample_rate = audio_sample_rate

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):





        csi_path = self.data_df.iloc[idx]['csi_path']
        csi_data = pd.read_csv(csi_path).values.T
        csi_tensor = torch.tensor(csi_data, dtype=torch.float32)







        audio_path = self.data_df.iloc[idx]['audio_path']
        audio, sr = torchaudio.load(audio_path)
        if sr != self.audio_sample_rate:
            resample = Resample(sr, self.audio_sample_rate)
            audio = resample(audio)
        audio = torch.flatten(audio)




        min_len = min(csi_tensor.shape[1], audio.shape[0])
        csi_tensor = csi_tensor[:, :min_len]
        audio = audio[:min_len]

        return csi_tensor, audio



csi_audio_dataset = CSIAudioDataset(data_df)
train_loader = DataLoader(csi_audio_dataset, batch_size=BATCH_SIZE, shuffle=True)


class CW(nn.Module):
    def __init__(self, in_channels, residual_channels, skip_channels, dilation_cycles=DILATION_CYCLES, residual_layers=RESIDUAL_LAYERS):
        super(CW, self).__init__()
        self.initial_conv = nn.Conv1d(1, residual_channels, kernel_size=1)

        self.residual_blocks = nn.ModuleList()
        self.skip_connections = nn.ModuleList()




        for i in range(residual_layers):
            dilation = 2 ** (i % dilation_cycles)
            self.residual_blocks.append(ResidualBlock(residual_channels, dilation, skip_channels))
            self.skip_connections.append(nn.Conv1d(residual_channels, skip_channels, kernel_size=1))

        self.output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(skip_channels, skip_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(skip_channels, 1, kernel_size=1)
        )


        self.csi_conv = nn.Sequential(
            nn.Conv1d(in_channels, residual_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
        )




    def forward(self, csi_data, audio_data):

        csi_conditioned = self.csi_conv(csi_data.unsqueeze(1))
        csi_conditioned = nn.functional.interpolate(csi_conditioned, size=audio_data.size(1), mode='linear')
        x = self.initial_conv(audio_data.unsqueeze(1))
        skip_outs = []


        for i, residual_block in enumerate(self.residual_blocks):
            x, skip = residual_block(x, csi_conditioned)
            skip_outs.append(self.skip_connections[i](skip))


        x = torch.sum(torch.stack(skip_outs), dim=0)
        return self.output_layer(x).squeeze(1)

class ResidualBlock(nn.Module):
    def __init__(self, residual_channels, dilation, skip_channels):
        super(ResidualBlock, self).__init__()
        self.dilated_conv = nn.Conv1d(residual_channels, residual_channels, kernel_size=2, dilation=dilation, padding=dilation)
        self.condition_conv = nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
        self.output_conv = nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
        self.skip_conv = nn.Conv1d(residual_channels, skip_channels, kernel_size=1)

    def forward(self, x, condition):
        gated = torch.tanh(self.dilated_conv(x) + self.condition_conv(condition))
        residual_out = self.output_conv(gated) + x
        skip_out = self.skip_conv(gated)
        return residual_out, skip_out





def train(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for csi_data, audio_data in data_loader:
        csi_data, audio_data = csi_data.to(device), audio_data.to(device)
        optimizer.zero_grad()
        output = model(csi_data, audio_data)
        loss = criterion(output, audio_data)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)




def test(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for csi_data, audio_data in data_loader:
            csi_data, audio_data = csi_data.to(device), audio_data.to(device)
            output = model(csi_data, audio_data)
            loss = criterion(output, audio_data)
            total_loss += loss.item()
    return total_loss / len(data_loader)




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CW(in_channels=1, residual_channels=64, skip_channels=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

######################################################## Training ############################################
for epoch in tqdm(range(EPOCHS), desc="Training Epochs"):
    train_loss = train(model, train_loader, optimizer, criterion, device)
    tqdm.write(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {train_loss:.4f}")


torch.save(model.state_dict(), "wavenet1.pth")




def generate_audio(model, csi_data_path):
    model.eval()
    with torch.no_grad():
        csi_data = pd.read_csv(csi_data_path).values.T
        csi_tensor = torch.tensor(csi_data, dtype=torch.float32).unsqueeze(0).to(device)
        generated_audio = model(csi_tensor, torch.zeros(1, 1, csi_tensor.size(-1)).to(device))
        return generated_audio.cpu().numpy()

generated_audio = generate_audio(model, 'data_capture\csi_amplitude_44th\csi_data_2024-10-27_20-28-20.897.csv')
torchaudio.save('generated_audio.wav', torch.tensor(generated_audio), sample_rate=AUDIO_SAMPLE_RATE)