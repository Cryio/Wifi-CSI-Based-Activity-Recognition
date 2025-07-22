import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class CSIAudioDataset(Dataset):
    def __init__(self, data_df):
        self.data_df = data_df
        logging.info(f"Initialized dataset with {len(data_df)} samples")

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        try:
            logging.debug(f"Processing index: {idx}")
            csi_path = self.data_df.iloc[idx]['csi_path']
            
            if not os.path.exists(csi_path):
                logging.error(f"CSI file not found: {csi_path}")
                return None
                
            csi_data = pd.read_csv(csi_path)
            subcarrier_cols = [col for col in csi_data.columns if col.startswith('subcarrier_')]
            
            if not subcarrier_cols:
                logging.error(f"No subcarrier columns found in {csi_path}")
                return None
                
            csi_data = csi_data[subcarrier_cols].values.T  # shape: (8, N)
            
            # Interpolate to 400 data points for each subcarrier
            target_len = 400
            csi_data_resampled = np.array([
                np.interp(
                    np.linspace(0, len(channel) - 1, target_len),
                    np.arange(len(channel)),
                    channel
                ) for channel in csi_data
            ])
            csi_tensor = torch.tensor(csi_data_resampled, dtype=torch.float32)

            # Load audio data
            audio_path = self.data_df.iloc[idx]['audio_path']
            
            if not os.path.exists(audio_path):
                logging.error(f"Audio file not found: {audio_path}")
                return None
                
            audio_data = pd.read_csv(audio_path, header=None).values.flatten()
            audio_data_resampled = np.interp(
                np.linspace(0, len(audio_data) - 1, target_len),
                np.arange(len(audio_data)),
                audio_data
            )
            audio_tensor = torch.tensor(audio_data_resampled, dtype=torch.float32)

            # Normalize the data
            csi_tensor = (csi_tensor - csi_tensor.mean()) / (csi_tensor.std() + 1e-8)
            audio_tensor = (audio_tensor - audio_tensor.mean()) / (audio_tensor.std() + 1e-8)

            return csi_tensor, audio_tensor
            
        except Exception as e:
            logging.error(f"Error processing index {idx}: {str(e)}")
            return None

class WindowedTrainer:
    def __init__(self, generator, discriminator, dataset, hyperparameters):
        """
        Initialize the windowed trainer.
        
        Args:
            generator: Generator model
            discriminator: Discriminator model
            dataset: CSIAudioDataset instance
            hyperparameters: Dictionary containing training hyperparameters
        """
        self.generator = generator.to(hyperparameters['device'])
        self.discriminator = discriminator.to(hyperparameters['device'])
        self.dataset = dataset
        self.hyperparameters = hyperparameters
        
        logging.info(f"Dataset size: {len(dataset)}")
        
        # Initialize optimizers
        self.optimizer_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=hyperparameters['learning_rate'],
            betas=(hyperparameters['beta1'], hyperparameters['beta2'])
        )
        self.optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=hyperparameters['learning_rate'],
            betas=(hyperparameters['beta1'], hyperparameters['beta2'])
        )
        
        # Initialize data loader
        self.train_loader = DataLoader(
            dataset,
            batch_size=hyperparameters['batch_size'],
            shuffle=True,
            drop_last=True,
            collate_fn=self.collate_fn,
            num_workers=0  # Set to 0 for debugging
        )
        
        logging.info(f"Number of batches: {len(self.train_loader)}")
        
        # Initialize lists for tracking losses
        self.d_losses = []
        self.g_losses = []
        
    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to handle None values and create proper batches.
        """
        # Filter out None values
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            logging.warning("Empty batch after filtering None values")
            return None
        
        # Separate CSI and audio data
        csi_data = [item[0] for item in batch]
        audio_data = [item[1] for item in batch]
        
        try:
            # Stack the tensors
            csi_batch = torch.stack(csi_data)
            audio_batch = torch.stack(audio_data)
            
            logging.debug(f"Batch shapes - CSI: {csi_batch.shape}, Audio: {audio_batch.shape}")
            return csi_batch, audio_batch
            
        except Exception as e:
            logging.error(f"Error in collate_fn: {str(e)}")
            return None
    
    def create_windows(self, data, window_size, stride):
        """
        Create windows from a batch of data.
        
        Args:
            data: Tensor of shape (batch_size, channels/features, sequence_length)
            window_size: Number of time steps in each window
            stride: Number of time steps to move between windows
            
        Returns:
            List of windows, each of shape (batch_size, channels/features, window_size)
        """
        if data is None:
            return []
            
        windows = []
        # Handle the case where data is 2D (batch_size, sequence_length)
        if len(data.shape) == 2:
            data = data.unsqueeze(1)  # Add channel dimension
            
        try:
            for i in range(0, data.shape[2] - window_size + 1, stride):
                window = data[:, :, i:i + window_size]
                windows.append(window)
            return windows
        except Exception as e:
            logging.error(f"Error creating windows: {str(e)}")
            return []
    
    def train_epoch(self, epoch):
        """
        Train for one epoch using windowed data.
        """
        running_d_loss = 0.0
        running_g_loss = 0.0
        total_windows = 0
        
        # Window parameters
        window_size = 64  # Adjust based on your needs
        stride = 32      # 50% overlap
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")
        for batch_idx, batch_data in enumerate(progress_bar):
            if batch_data is None:
                logging.warning(f"Skipping empty batch {batch_idx}")
                continue
                
            csi_data, audio_data = batch_data
            
            # Log shapes for debugging
            logging.debug(f"Batch {batch_idx} shapes - CSI: {csi_data.shape}, Audio: {audio_data.shape}")
            
            # Move data to device
            csi_data = csi_data.to(self.hyperparameters['device'])
            audio_data = audio_data.to(self.hyperparameters['device'])
            
            # Create windows
            csi_windows = self.create_windows(csi_data, window_size, stride)
            audio_windows = self.create_windows(audio_data, window_size, stride)
            
            if not csi_windows or not audio_windows:
                logging.warning(f"No windows created for batch {batch_idx}")
                continue
            
            # Update total windows count
            total_windows += len(csi_windows)
            
            # Train on each window
            for window_idx, (csi_window, audio_window) in enumerate(zip(csi_windows, audio_windows)):
                try:
                    # Ensure proper shapes
                    if len(audio_window.shape) == 2:
                        audio_window = audio_window.unsqueeze(1)
                    
                    # Train discriminator
                    self.optimizer_d.zero_grad()
                    
                    z = torch.randn(
                        csi_window.size(0),
                        self.hyperparameters['latent_dim'],
                        device=self.hyperparameters['device']
                    )
                    
                    fake_audio = self.generator(z, csi_window)
                    
                    d_real = self.discriminator(audio_window, csi_window)
                    d_fake = self.discriminator(fake_audio.detach(), csi_window)
                    
                    d_loss_real = -torch.mean(d_real)
                    d_loss_fake = torch.mean(d_fake)
                    d_loss = d_loss_real + d_loss_fake
                    
                    gp = self.gradient_penalty(audio_window, fake_audio.detach(), csi_window)
                    d_loss += 10.0 * gp
                    
                    d_loss.backward()
                    self.optimizer_d.step()
                    
                    # Train generator
                    self.optimizer_g.zero_grad()
                    
                    fake_audio = self.generator(z, csi_window)
                    d_fake = self.discriminator(fake_audio, csi_window)
                    
                    g_loss = -torch.mean(d_fake)
                    
                    g_loss.backward()
                    self.optimizer_g.step()
                    
                    # Update running losses
                    running_d_loss += d_loss.item()
                    running_g_loss += g_loss.item()
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'd_loss': f"{d_loss.item():.4f}",
                        'g_loss': f"{g_loss.item():.4f}"
                    })
                    
                except Exception as e:
                    logging.error(f"Error in window {window_idx} of batch {batch_idx}: {str(e)}")
                    continue
        
        if total_windows == 0:
            logging.error("No windows were processed in this epoch")
            return float('inf'), float('inf')
        
        avg_d_loss = running_d_loss / total_windows
        avg_g_loss = running_g_loss / total_windows
        
        self.d_losses.append(avg_d_loss)
        self.g_losses.append(avg_g_loss)
        
        return avg_d_loss, avg_g_loss
    
    def gradient_penalty(self, real_data, fake_data, condition):
        """
        Calculate gradient penalty for WGAN-GP.
        """
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, 1).to(self.hyperparameters['device'])
        alpha = alpha.expand_as(real_data)
        
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)
        
        d_interpolated = self.discriminator(interpolated, condition)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        
        return gradient_penalty
    
    def train(self):
        """
        Complete training loop for all epochs.
        """
        logging.info("Starting training...")
        for epoch in range(self.hyperparameters['epochs']):
            d_loss, g_loss = self.train_epoch(epoch)
            
            logging.info(
                f"Epoch [{epoch + 1}/{self.hyperparameters['epochs']}] "
                f"D_loss: {d_loss:.4f}, G_loss: {g_loss:.4f}"
            )
            
            # Save models periodically
            if (epoch + 1) % 10 == 0:
                torch.save(self.generator.state_dict(), f'generator_epoch_{epoch + 1}.pth')
                torch.save(self.discriminator.state_dict(), f'discriminator_epoch_{epoch + 1}.pth')
        
        self.plot_training_progress()
    
    def plot_training_progress(self):
        """
        Plot the training losses.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.d_losses, label='Discriminator Loss', color='red')
        plt.plot(self.g_losses, label='Generator Loss', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_progress.png')
        plt.close()

def main():
    # Create the data DataFrame
    data_dir = "data_capture"
    csi_dir = os.path.join(data_dir, "csi")
    audio_dir = os.path.join(data_dir, "audio_matrix")
    
    # Get all CSI files
    csi_files = [f for f in os.listdir(csi_dir) if f.endswith('.csv')]
    
    # Create DataFrame with matching audio files
    data = []
    for csi_file in csi_files:
        audio_file = csi_file.replace('csi_data_', '')
        audio_path = os.path.join(audio_dir, audio_file)
        if os.path.exists(audio_path):
            data.append({
                'csi_path': os.path.join(csi_dir, csi_file),
                'audio_path': audio_path
            })
    
    train_df = pd.DataFrame(data)
    logging.info(f"Found {len(train_df)} matching CSI-Audio pairs")
    
    if len(train_df) == 0:
        logging.error("No matching CSI-Audio pairs found!")
        return
    
    # Your hyperparameters
    hyperparameters = {
        "latent_dim": latent_dim,
        "csi_channels": csi_channels,
        "audio_channels": audio_channels,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "beta1": beta1,
        "beta2": beta2,
        "sample_rate": sample_rate,
        "device": device
    }
    
    # Create dataset
    dataset = CSIAudioDataset(train_df)
    
    # Initialize models
    generator = Generator(latent_dim, csi_channels).to(device)
    discriminator = Discriminator(csi_channels).to(device)
    
    # Initialize trainer
    trainer = WindowedTrainer(generator, discriminator, dataset, hyperparameters)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main() 