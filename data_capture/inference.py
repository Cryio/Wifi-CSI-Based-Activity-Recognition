import os
import re
import gc
import math
import glob
import logging
import warnings
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchaudio.transforms as T

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSI_DIR = 'csi_data/data/csi'
AUDIO_DIR = 'csi_data/data/audio'
MODEL_PATH = 'best_csi2audio.pth'
TARGET_SAMPLE_RATE = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        src = src.permute(1, 2, 0)
        src_emb = self.csi_embedder(src)
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

def load_csi_file(csi_path, num_csi_features):
    try:
        csi_df = pd.read_csv(csi_path, on_bad_lines='skip')
        if csi_df.empty: return None
        csi_num = csi_df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
        csi_tensor = torch.from_numpy(csi_num.values.astype(np.float32))
        del csi_df, csi_num
    except Exception as e:
        logging.warning(f"Failed to load CSI: {csi_path} - {e}")
        return None

    if csi_tensor.shape[1] != num_csi_features:
        logging.warning(f"Feature mismatch: {csi_tensor.shape[1]} != {num_csi_features}")
        return None

    csi_tensor = (csi_tensor - csi_tensor.mean(dim=0)) / (csi_tensor.std(dim=0) + 1e-6)
    csi_tensor = torch.nan_to_num(csi_tensor)
    return csi_tensor

@torch.no_grad()
def generate_audio(model, csi_tensor, device, max_len=1000, start_token=0.0):
    model.eval()
    src = csi_tensor.to(device)
    
    if src.dim() == 2:
        src = src.unsqueeze(1)  # [T, F] -> [T, 1, F]
    
    batch_size = src.size(1)
    src_pad = torch.zeros(batch_size, src.size(0), dtype=torch.bool, device=device)
    tgt = torch.tensor([[[start_token]]], dtype=torch.float32, device=device)
    generated_samples = []
    target_len = src.size(0) if max_len is None else max_len
    window_size = min(512, target_len)

    print(f"Generating {target_len} samples...")
    for i in tqdm(range(target_len)):
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

def find_closest_audio(csi_timestamp, audio_dir, tolerance_seconds=60):
    from datetime import datetime
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    
    best_match = None
    best_diff = float('inf')
    
    for aud_file in audio_files:
        m = re.search(r'(.+)\.wav', aud_file)
        if not m:
            continue
        try:
            aud_ts = datetime.strptime(m.group(1), '%Y-%m-%d_%H-%M-%S.%f')
            diff = abs((csi_timestamp - aud_ts).total_seconds())
            if diff < best_diff and diff <= tolerance_seconds:
                best_diff = diff
                best_match = aud_file
        except:
            continue
    
    return best_match, best_diff

def get_csi_timestamp(csi_filename):
    m = re.search(r'csi_data_(.+)\.csv', csi_filename)
    if not m:
        return None
    from datetime import datetime
    try:
        return datetime.strptime(m.group(1), '%Y-%m-%d_%H-%M-%S.%f')
    except:
        try:
            return datetime.strptime(m.group(1), '%Y-%m-%d_%H-%M-%S')
        except:
            return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
    warnings.filterwarnings("ignore")
    
    print(f"Using device: {DEVICE}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        exit(1)
    
    print("Loading model...")
    num_csi_features = 27
    
    model = CSIToAudioTransformer(
        num_csi_features=num_csi_features,
        num_audio_features=1, d_model=192, nhead=6,
        num_encoder_layers=6, num_decoder_layers=6,
        dim_feedforward=768, dropout=0.1, max_len=4096
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Model loaded.")
    
    csi_files = sorted(glob.glob(os.path.join(CSI_DIR, "csi_data_*.csv")))
    print(f"Found {len(csi_files)} CSI files")
    
    output_dir = "generated_audio"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, csi_path in enumerate(csi_files[:5]):
        csi_filename = os.path.basename(csi_path)
        print(f"\n[{i+1}/5] Processing: {csi_filename}")
        
        csi_tensor = load_csi_file(csi_path, num_csi_features)
        if csi_tensor is None:
            continue
        
        print(f"  CSI shape: {csi_tensor.shape}")
        
        gen_len = min(csi_tensor.shape[0], 2000)
        gen_audio = generate_audio(model, csi_tensor[:gen_len], DEVICE, max_len=gen_len)
        
        output_path = os.path.join(output_dir, f"gen_{csi_filename.replace('.csv', '.wav')}")
        sf.write(output_path, gen_audio, TARGET_SAMPLE_RATE)
        print(f"  Saved: {output_path}")
        
        del csi_tensor, gen_audio
        gc.collect()
        torch.cuda.empty_cache()
    
    print("\nDone!")
