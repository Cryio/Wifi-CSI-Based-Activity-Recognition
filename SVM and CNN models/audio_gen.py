import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Hyperparameters
latent_dim = 100
csi_channels = 1
audio_channels = 1
epochs = 50
batch_size = 32
learning_rate = 0.0002
beta1 = 0.5
beta2 = 0.999
sample_rate = 16000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Dataset class
class CSIAudioDataset(Dataset):
    def __init__(self, data_df):
        self.data_df = data_df

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        logging.debug(f"Processing index: {idx}")
        csi_path = self.data_df.iloc[idx]['csi_path']
        csi_data = pd.read_csv(csi_path).values.T

        # Interpolation to 400 data points for CSI data
        target_len = 400
        csi_data_resampled = np.array([np.interp(np.linspace(0, len(channel) - 1, target_len), 
                                                 np.arange(len(channel)), channel) for channel in csi_data])
        
        csi_tensor = torch.tensor(csi_data_resampled, dtype=torch.float32)

        # Replace torchaudio with CSV loading for audio
        audio_path = self.data_df.iloc[idx]['audio_path']
        audio_data = pd.read_csv(audio_path, header=None).values.flatten()  # Read audio from CSV
        
        # Interpolation to 400 data points for audio data
        audio_data_resampled = np.interp(np.linspace(0, len(audio_data) - 1, target_len), 
                                         np.arange(len(audio_data)), audio_data)
        
        audio_tensor = torch.tensor(audio_data_resampled, dtype=torch.float32)

        # Ensure both tensors have the same length
        min_len = min(csi_tensor.shape[1], len(audio_tensor))
        if min_len == 0:
            logging.warning(f"No valid audio length for index: {idx}")
            return None

        csi_tensor_trimmed = csi_tensor[:, :min_len]
        audio_trimmed = audio_tensor[:min_len]

        logging.debug(f"csi_tensor_trimmed shape: {csi_tensor_trimmed.shape}")
        logging.debug(f"audio_trimmed shape: {audio_trimmed.shape}")
        return csi_tensor_trimmed, audio_trimmed


train_df = pd.read_csv(r"SVM and CNN models\data_paths.csv")
train_dataset = CSIAudioDataset(train_df)
train_loader = DataLoader([data for data in train_dataset if data is not None], batch_size=batch_size, shuffle=True)


# Generator Model
class Generator(nn.Module):
    def __init__(self, latent_dim, csi_channels, audio_channels=1, csi_length=400):
        super(Generator, self).__init__()
        input_dim = latent_dim + csi_channels * csi_length  # Total input dimension after concatenation
        self.fc = nn.Linear(input_dim, 256 * 100)  # Flatten input to match ConvTranspose1D input size

        self.conv1 = nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose1d(64, audio_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, z, csi_data):
        # Flatten CSI data
        csi_data_flattened = csi_data.view(csi_data.size(0), -1)

        # Concatenate latent vector and flattened CSI data
        x = torch.cat([z, csi_data_flattened], dim=1)

        # Fully connected layer to project to the correct shape
        x = self.fc(x)
        x = x.view(x.size(0), 256, 100)  # Reshape for ConvTranspose1D

        # Apply transposed convolutions
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x).view(x.size(0), -1)  # Final output
        return torch.sigmoid(x)

# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self, csi_channels, audio_channels=1):
        super(Discriminator, self).__init__()
        input_dim = csi_channels + audio_channels
        self.fc = nn.Linear(input_dim, 128)
        self.conv1 = nn.Conv1d(1, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv1d(128, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, audio_data, csi_data):
        logging.debug(f"audio_data shape: {audio_data.shape}")
        logging.debug(f"csi_data shape: {csi_data.shape}")
        csi_data_flattened = csi_data.view(csi_data.size(0), -1)
        try:
            x = torch.cat([audio_data, csi_data_flattened], dim=1).unsqueeze(1)
        except RuntimeError as e:
            logging.error(f"Error during concatenation: {e}")
            raise
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x).view(x.size(0), -1)
        return torch.sigmoid(x)


# Loss functions
def generator_loss(d_output):
    return F.binary_cross_entropy(d_output, torch.ones_like(d_output))


def discriminator_loss(d_output_real, d_output_fake):
    real_loss = F.binary_cross_entropy(d_output_real, torch.ones_like(d_output_real))
    fake_loss = F.binary_cross_entropy(d_output_fake, torch.zeros_like(d_output_fake))
    return real_loss + fake_loss


# Initialize models and optimizers
generator = Generator(latent_dim, csi_channels).to(device)
discriminator = Discriminator(csi_channels).to(device)
optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, beta2))
optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, beta2))

# Training loop
for epoch in range(epochs):
    logging.info(f"Epoch {epoch + 1}/{epochs}")
    running_d_loss, running_g_loss = 0.0, 0.0
    for csi_data, audio_data in tqdm(train_loader):
        csi_data, audio_data = csi_data.to(device), audio_data.to(device)

        # Train discriminator
        optimizer_d.zero_grad()
        z = torch.randn(audio_data.size(0), latent_dim).to(device)
        generated_audio = generator(z, csi_data)
        d_output_real = discriminator(audio_data, csi_data)
        d_output_fake = discriminator(generated_audio.detach(), csi_data)
        d_loss = discriminator_loss(d_output_real, d_output_fake)
        d_loss.backward()
        optimizer_d.step()
        running_d_loss += d_loss.item()

        # Train generator
        optimizer_g.zero_grad()
        d_output_fake = discriminator(generated_audio, csi_data)
        g_loss = generator_loss(d_output_fake)
        g_loss.backward()
        optimizer_g.step()
        running_g_loss += g_loss.item()

    logging.info(f"Epoch {epoch + 1} | D Loss: {running_d_loss / len(train_loader):.4f} | G Loss: {running_g_loss / len(train_loader):.4f}")

# Save models
torch.save(generator.state_dict(), "generator.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")
