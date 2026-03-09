import os
import re
import gc
import math
import glob
import time
import random
import logging
import warnings
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast

import torchaudio
import torchaudio.transforms as T

import matplotlib
matplotlib.use("Agg")  # Safe for headless servers/scripts
import matplotlib.pyplot as plt

# =========================
# ===== HYPERPARAMS =======
# =========================

DEBUG_MODE = False  # Set to True for quick code testing

# Model Config
D_MODEL = 192
NHEAD = 6
NUM_ENC_LAYERS = 6
NUM_DEC_LAYERS = 6
DIM_FF = 768
DROPOUT = 0.1

# Audio Config
TARGET_SAMPLE_RATE = 16000
CHUNK_SIZE = 4096
MAX_LEN = 4096

# --- TRAINING CONFIG ---
BATCH_SIZE = 6
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
EPOCHS = 30

DEBUG_FILE_LIMIT = 10 if DEBUG_MODE else None
DEBUG_BATCH_LIMIT = 10 if DEBUG_MODE else None

# =========================
# ===== UTILITIES =========
# =========================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class MultiResolutionSTFTLoss(nn.Module):
    """
    Computes spectral convergence and log magnitude loss over multiple STFT resolutions.
    This forces the model to learn frequency content, preventing 'static noise' artifacts.
    """
    def __init__(self, fft_sizes=[1024, 2048, 512], hop_sizes=[120, 240, 50], win_lengths=[600, 1200, 240]):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

    def forward(self, x, y):
        # x: predicted, y: target. Shape: [Batch, Time] or [Batch, 1, Time]
        if x.dim() == 3: x = x.squeeze(1)
        if y.dim() == 3: y = y.squeeze(1)

        loss = 0.0
        for n_fft, hop_length, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            window = torch.hann_window(win_length).to(x.device)

            # return_complex=True is required for PyTorch > 1.7
            x_stft = torch.stft(x, n_fft, hop_length, win_length, window, return_complex=True)
            y_stft = torch.stft(y, n_fft, hop_length, win_length, window, return_complex=True)

            x_mag = torch.abs(x_stft) + 1e-7
            y_mag = torch.abs(y_stft) + 1e-7

            # Spectral Convergence Loss
            sc_loss = torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")
            # Log Magnitude Loss
            mag_loss = F.l1_loss(torch.log(x_mag), torch.log(y_mag))

            loss += sc_loss + mag_loss

        return loss / len(self.fft_sizes)

# =========================
# ===== DATASET ===========
# =========================

class CsiAudioDataset(Dataset):
    def __init__(self, csi_dir, audio_dir, target_sample_rate=16000, chunk_size=1024, debug_limit_files=None):
        self.csi_dir = csi_dir
        self.audio_dir = audio_dir
        self.target_sample_rate = target_sample_rate
        self.chunk_size = chunk_size
        self.num_csi_features = 0
        self.chunks = []

        file_pairs = self._find_pairs()
        if debug_limit_files is not None:
            file_pairs = file_pairs[:debug_limit_files]
            logging.warning(f"[DEBUG] Limiting to {len(file_pairs)} file pairs.")

        if not file_pairs:
            logging.warning(f"No matching CSI/Audio pairs found in {csi_dir}")
            return

        logging.info(f"Found {len(file_pairs)} matched pairs. Pre-processing & chunking...")

        valid_files_count = 0

        for csi_path, audio_path in tqdm(file_pairs, desc="Pre-processing"):
            try:
                csi_tensor, audio_tensor = self._load_data(csi_path, audio_path)

                if (csi_tensor is None) or (audio_tensor is None):
                    continue
                if (csi_tensor.shape[0] < 10) or (audio_tensor.shape[0] < 10):
                    continue

                if self.num_csi_features == 0:
                    self.num_csi_features = csi_tensor.shape[1]

                if csi_tensor.shape[1] != self.num_csi_features:
                    logging.debug(f"Skipping {os.path.basename(csi_path)}: Feature dim {csi_tensor.shape[1]} != expected {self.num_csi_features}")
                    continue

                # Align lengths
                csi_len, audio_len = csi_tensor.shape[0], audio_tensor.shape[0]
                if csi_len != audio_len:
                    csi_tensor = csi_tensor.permute(1, 0).unsqueeze(0)
                    csi_tensor = F.interpolate(csi_tensor, size=audio_len, mode='linear', align_corners=False)
                    csi_tensor = csi_tensor.squeeze(0).permute(1, 0)

                # Standardize
                csi_tensor = (csi_tensor - csi_tensor.mean(dim=0)) / (csi_tensor.std(dim=0) + 1e-6)
                audio_tensor = (audio_tensor - audio_tensor.mean()) / (audio_tensor.std() + 1e-6)

                csi_tensor = torch.nan_to_num(csi_tensor)
                audio_tensor = torch.nan_to_num(audio_tensor)

                # Create Chunks
                total_len = csi_tensor.shape[0]
                for start_idx in range(0, total_len, self.chunk_size):
                    end_idx = min(start_idx + self.chunk_size, total_len)

                    if end_idx - start_idx < self.chunk_size // 2:
                        continue

                    self.chunks.append({
                        'csi': csi_tensor[start_idx:end_idx].clone(),
                        'audio': audio_tensor[start_idx:end_idx].clone(),
                        'csi_path': csi_path,
                        'audio_path': audio_path
                    })

                valid_files_count += 1
                del csi_tensor, audio_tensor
                if valid_files_count % 20 == 0:
                     gc.collect()

            except Exception as e:
                logging.warning(f"Skipping file due to error: {csi_path} - {e}")
                continue

        if self.num_csi_features == 0:
            logging.fatal("FATAL: Could not determine CSI feature count. All data may be corrupt.")
        else:
            logging.info(f"Dataset created: {len(self.chunks)} chunks from {valid_files_count} valid files.")
            gc.collect()

    def _find_pairs(self):
        pairs = []
        csi_files = glob.glob(os.path.join(self.csi_dir, "csi_data_*.csv"))
        for csi_path in csi_files:
            filename = os.path.basename(csi_path)
            m = re.search(r'csi_data_(.*)\.csv', filename)
            if not m: continue
            ts = m.group(1)
            possible_audio_names = [f"{ts.replace('.', '-')}.wav", f"{ts}.wav"]
            found = False
            for aud_name in possible_audio_names:
                aud_path = os.path.join(self.audio_dir, aud_name)
                if os.path.exists(aud_path):
                    pairs.append((csi_path, aud_path))
                    found = True
                    break
        return pairs

    def _load_data(self, csi_path, audio_path):
            try:
                csi_df = pd.read_csv(csi_path, on_bad_lines='skip')
                if csi_df.empty: return None, None
                csi_num = csi_df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
                csi_tensor = torch.from_numpy(csi_num.values.astype(np.float32))
                del csi_df, csi_num
            except Exception:
                return None, None

            try:
                data, sr = sf.read(audio_path)
                waveform = torch.from_numpy(data).float()
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
                else:
                    waveform = waveform.t()
                if sr != self.target_sample_rate:
                    resampler = T.Resample(orig_freq=sr, new_freq=self.target_sample_rate)
                    waveform = resampler(waveform)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                audio_tensor = waveform.transpose(0, 1).float()
                return csi_tensor, audio_tensor
            except Exception:
                return None, None

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    valid = [b for b in batch if b['csi'].numel() > 0 and b['audio'].numel() > 0]
    if not valid: return None, None, None, None

    csi_batch = [b['csi'] for b in valid]
    audio_batch = [b['audio'] for b in valid]

    csi_pad = rnn_utils.pad_sequence(csi_batch, batch_first=False, padding_value=0.0)
    audio_pad = rnn_utils.pad_sequence(audio_batch, batch_first=False, padding_value=0.0)

    batch_size = len(valid)
    max_len_csi = csi_pad.size(0)
    max_len_audio = audio_pad.size(0)

    src_pad_mask = torch.zeros(batch_size, max_len_csi, dtype=torch.bool)
    tgt_pad_mask = torch.zeros(batch_size, max_len_audio, dtype=torch.bool)

    for i in range(batch_size):
        src_pad_mask[i, len(csi_batch[i]):] = True
        tgt_pad_mask[i, len(audio_batch[i]):] = True

    return csi_pad, audio_pad, src_pad_mask, tgt_pad_mask

# =========================
# ===== MODEL =============
# =========================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(0)
        if seq_len > self.pe.size(0):
            x = x[:self.pe.size(0)]
            seq_len = x.size(0)
        x = x + self.pe[:seq_len]
        return self.dropout(x)

class CSIToAudioTransformer(nn.Module):
    def __init__(self, num_csi_features, num_audio_features=1, d_model=192, nhead=6,
                 num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=768,
                 dropout=0.1, max_len=4096):
        super().__init__()
        self.d_model = d_model

        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)

        # UPDATED: Convolutional Encoder for CSI
        # Converts [Batch, CSI_Feat, Time] -> [Batch, D_Model, Time]
        self.csi_embedder = nn.Sequential(
            nn.Conv1d(num_csi_features, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        )

        self.audio_embedder = nn.Linear(num_audio_features, d_model)

        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.output_layer = nn.Linear(d_model, num_audio_features)

    @staticmethod
    def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None):
        # src: [S, B, C] -> Permute for Conv1d -> [B, C, S]
        src = src.permute(1, 2, 0)
        src_emb = self.csi_embedder(src)
        # Permute back for Transformer -> [S, B, D_Model]
        src_emb = src_emb.permute(2, 0, 1)

        src_emb = self.pos_encoder(src_emb * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.audio_embedder(tgt) * math.sqrt(self.d_model))

        tgt_mask = self.generate_square_subsequent_mask(tgt.size(0), src.device)

        out = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        return self.output_layer(out)

# =========================
# ===== TRAIN / EVAL ======
# =========================

def train_epoch(model, dataloader, optimizer, criterion, device, epoch_num, scaler, amp_enabled, debug_limit_batches=None, history=None):
    model.train()
    total_loss, batches = 0.0, 0
    loop = tqdm(dataloader, desc=f"Epoch {epoch_num:02} Train", leave=False)

    for i, batch in enumerate(loop):
        if debug_limit_batches is not None and i >= debug_limit_batches: break

        src, tgt, src_mask, tgt_mask = batch
        if src is None or tgt is None: continue

        src, tgt = src.to(device), tgt.to(device)
        src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)

        tgt_in = tgt[:-1]
        tgt_lab = tgt[1:]
        tgt_mask_in = tgt_mask[:, :-1]

        if tgt_in.size(0) == 0: continue

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=amp_enabled):
            pred = model(src, tgt_in, src_key_padding_mask=src_mask, tgt_key_padding_mask=tgt_mask_in)

            # Reshape for STFT Loss: [Time, Batch, 1] -> [Batch, Time]
            pred_flat = pred.squeeze(-1).transpose(0, 1)
            tgt_flat = tgt_lab.squeeze(-1).transpose(0, 1)

            loss = criterion(pred_flat, tgt_flat)

        if torch.isnan(loss):
            logging.warning("NaN loss detected. Skipping batch.")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        loss_val = loss.item()
        total_loss += loss_val
        batches += 1
        loop.set_postfix(loss=loss_val)

        if history is not None:
            history['batch_losses'].append(loss_val)

    return total_loss / batches if batches > 0 else 0.0

@torch.no_grad()
def validate_epoch(model, dataloader, criterion, device, debug_limit_batches=None):
    model.eval()
    total_loss, batches = 0.0, 0
    loop = tqdm(dataloader, desc="Validation", leave=False)

    for i, batch in enumerate(loop):
        if debug_limit_batches is not None and i >= debug_limit_batches: break

        src, tgt, src_mask, tgt_mask = batch
        if src is None or tgt is None: continue

        src, tgt = src.to(device), tgt.to(device)
        src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)

        tgt_in = tgt[:-1]
        tgt_lab = tgt[1:]
        tgt_mask_in = tgt_mask[:, :-1]

        if tgt_in.size(0) == 0: continue

        pred = model(src, tgt_in, src_key_padding_mask=src_mask, tgt_key_padding_mask=tgt_mask_in)

        pred_flat = pred.squeeze(-1).transpose(0, 1)
        tgt_flat = tgt_lab.squeeze(-1).transpose(0, 1)

        loss = criterion(pred_flat, tgt_flat)

        if not torch.isnan(loss):
            total_loss += loss.item()
            batches += 1
            loop.set_postfix(loss=loss.item())

    return total_loss / batches if batches > 0 else 0.0

@torch.no_grad()
def generate_audio_greedy(model, csi_tensor, device, max_len=None, start_token=0.0):
    model.eval()
    src = csi_tensor.to(device)
    if src.dim() == 2:
        src = src.unsqueeze(1)  # [T, F] -> [T, 1, F]

    batch_size = src.size(1)
    src_pad = torch.zeros(batch_size, src.size(0), dtype=torch.bool, device=device)
    tgt = torch.tensor([[[start_token]]], dtype=torch.float32, device=device)
    generated_samples = []
    target_len = src.size(0) if max_len is None else max_len
    window_size = 512

    pbar = tqdm(range(target_len), desc="Generating samples")
    for _ in pbar:
        if tgt.size(0) > window_size:
            tgt_window = tgt[-window_size:]
        else:
            tgt_window = tgt

        tgt_mask = model.generate_square_subsequent_mask(tgt_window.size(0), device)
        tgt_pad = torch.zeros(tgt_window.size(0), dtype=torch.bool, device=device).unsqueeze(0).expand(batch_size, -1)
        out = model(src, tgt_window, src_key_padding_mask=src_pad, tgt_key_padding_mask=tgt_pad)
        next_val = out[-1:]
        generated_samples.append(next_val.item())
        tgt = torch.cat([tgt, next_val], dim=0)

    return np.array(generated_samples, dtype=np.float32)

# =========================
# ====== MAIN =============
# =========================

if __name__ == "__main__":
    log_file = 'model_training.log'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s',
                        handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()])
    warnings.filterwarnings("ignore")

    logging.info("=== Script start ===")
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = torch.cuda.is_available()
    logging.info(f"Using device: {device} | AMP Enabled: {amp_enabled}")

    try:
        # 1. Dataset
        logging.info("Loading dataset...")
        dataset = CsiAudioDataset(CSI_DIR, AUDIO_DIR, target_sample_rate=TARGET_SAMPLE_RATE,
                                  chunk_size=CHUNK_SIZE, debug_limit_files=DEBUG_FILE_LIMIT)
        if len(dataset) == 0:
            logging.fatal("Dataset is empty. Exiting.")
            raise SystemExit(1)

        val_count = max(1, int(0.1 * len(dataset)))
        train_count = len(dataset) - val_count
        generator = torch.Generator().manual_seed(42)
        train_ds, val_ds = torch.utils.data.random_split(dataset, [train_count, val_count], generator=generator)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=collate_fn, pin_memory=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                collate_fn=collate_fn, pin_memory=True, num_workers=NUM_WORKERS)

        # 2. Model
        num_csi_features = dataset.num_csi_features
        logging.info(f"CSI Features detected: {num_csi_features}")

        model = CSIToAudioTransformer(
            num_csi_features=num_csi_features,
            num_audio_features=1, d_model=D_MODEL, nhead=NHEAD,
            num_encoder_layers=NUM_ENC_LAYERS, num_decoder_layers=NUM_DEC_LAYERS,
            dim_feedforward=DIM_FF, dropout=DROPOUT, max_len=MAX_LEN
        ).to(device)

        logging.info(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        # Optimization on Windows is often buggy, skip if on Windows
        if os.name != 'nt':
            try:
                model = torch.compile(model)
                logging.info("Model compiled with torch.compile().")
            except Exception as e:
                logging.warning(f"Compilation failed: {e}. Running eager mode.")
        else:
             logging.info("Windows detected: Skipping torch.compile().")

        # UPDATED: Use MultiResolutionSTFTLoss instead of MSELoss
        criterion = MultiResolutionSTFTLoss().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        scaler = GradScaler(enabled=amp_enabled)

        # 3. Training
        history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'batch_losses': []}
        best_val_loss = float('inf')

        logging.info("=== Starting Training ===")
        for epoch in range(1, EPOCHS + 1):
            start_time = time.time()
            t_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, scaler, amp_enabled, DEBUG_BATCH_LIMIT, history)
            v_loss = validate_epoch(model, val_loader, criterion, device, DEBUG_BATCH_LIMIT)
            scheduler.step(v_loss)

            elapsed = time.time() - start_time
            logging.info(f"Epoch {epoch:02d} | Time {elapsed:.1f}s | Train Loss {t_loss:.5f} | Val Loss {v_loss:.5f}")

            history['epoch'].append(epoch)
            history['train_loss'].append(t_loss)
            history['val_loss'].append(v_loss)

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                torch.save(model.state_dict(), "best_csi2audio.pth")
                logging.info(" -> New best model saved.")

            # --- UPDATED PLOTTING LOGIC ---
            plt.figure(figsize=(12, 5))

            # Plot 1: Epoch Averaged Loss
            plt.subplot(1, 2, 1)
            plt.plot(history['epoch'], history['train_loss'], label='Train Loss', marker='o')
            plt.plot(history['epoch'], history['val_loss'], label='Val Loss', marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('STFT Loss')
            plt.title('Training vs Validation Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Plot 2: Detailed Batch Loss
            plt.subplot(1, 2, 2)
            plt.plot(history['batch_losses'], color='gray', alpha=0.3, linewidth=0.5)
            if len(history['batch_losses']) > 100:
                trend = pd.Series(history['batch_losses']).rolling(50).mean()
                plt.plot(trend, color='blue', linewidth=1.5, label='Trend')
            plt.xlabel('Total Batches')
            plt.ylabel('Batch Loss')
            plt.title('Detailed Training Stability')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('training_loss_plot.png')
            plt.close()

        # 5. Inference Demo
        logging.info("=== Running Inference ===")
        if os.path.exists("best_csi2audio.pth"):
             try: model.load_state_dict(torch.load("best_csi2audio.pth", map_location=device))
             except: logging.warning("Strict load failed, using current weights.")

        sample_idx = random.randint(0, len(val_ds)-1)
        sample = val_ds[sample_idx]
        csi_chunk = sample['csi']
        original_audio = sample['audio']

        logging.info(f"Generating from chunk: {os.path.basename(sample['csi_path'])}")
        gen_len = min(csi_chunk.size(0), 1000) # Short generation for speed
        gen_audio = generate_audio_greedy(model, csi_chunk[:gen_len], device, max_len=gen_len)

        sf.write("generated_demo.wav", gen_audio, TARGET_SAMPLE_RATE)
        sf.write("original_demo.wav", original_audio[:gen_len].squeeze().numpy(), TARGET_SAMPLE_RATE)
        logging.info("Saved generated_demo.wav")

    except KeyboardInterrupt:
        logging.info("Stopped by user.")
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
    finally:
        logging.info("Done.")
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()