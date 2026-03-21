"""
ESC-50 dataset loader with mel spectrogram extraction and augmentation.

Supports:
- Automatic resampling and mono conversion
- Mel spectrogram feature extraction
- SpecAugment (time/frequency masking)
- Waveform-level augmentation (Gaussian noise, random gain)
"""

import os
import random

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset


class ESC50Dataset(Dataset):
    """ESC-50 Environmental Sound Classification dataset.

    Args:
        root: Path to ESC-50 root directory (containing audio/ and meta/).
        folds: List of fold indices to include (1-5).
        sample_rate: Target sample rate for resampling.
        duration: Clip duration in seconds (pads or truncates).
        n_mels: Number of mel frequency bands.
        augment: Whether to apply data augmentation.
    """

    def __init__(
        self,
        root: str,
        folds: list,
        sample_rate: int = 22050,
        duration: float = 5.0,
        n_mels: int = 128,
        augment: bool = False,
    ):
        self.root = root
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.augment = augment
        self.target_length = int(sample_rate * duration)

        # Load and filter metadata by fold
        meta_path = os.path.join(root, "meta", "esc50.csv")
        meta = pd.read_csv(meta_path)
        self.data = meta[meta["fold"].isin(folds)].reset_index(drop=True)

        # Feature extraction transforms
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=n_mels,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

        # SpecAugment transforms
        if augment:
            self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=30)
            self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=10)

    def __len__(self) -> int:
        return len(self.data)

    def _pad_or_truncate(self, waveform: torch.Tensor) -> torch.Tensor:
        """Ensure waveform has exactly target_length samples."""
        length = waveform.shape[1]
        if length > self.target_length:
            # Random crop during training, center crop otherwise
            if self.augment:
                start = random.randint(0, length - self.target_length)
            else:
                start = (length - self.target_length) // 2
            waveform = waveform[:, start : start + self.target_length]
        elif length < self.target_length:
            padding = self.target_length - length
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        return waveform

    def _augment_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply waveform-level augmentations."""
        # Additive Gaussian noise
        if random.random() < 0.5:
            noise_std = random.uniform(0.001, 0.01)
            waveform = waveform + noise_std * torch.randn_like(waveform)
        # Random gain
        if random.random() < 0.3:
            gain = random.uniform(0.7, 1.3)
            waveform = waveform * gain
        return waveform

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        filepath = os.path.join(self.root, "audio", row["filename"])

        # Load audio
        waveform, sr = torchaudio.load(filepath)

        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Pad or truncate to fixed length
        waveform = self._pad_or_truncate(waveform)

        # Waveform augmentation
        if self.augment:
            waveform = self._augment_waveform(waveform)

        # Extract mel spectrogram
        mel_spec = self.mel_transform(waveform)
        mel_spec = self.amplitude_to_db(mel_spec)

        # SpecAugment (spectrogram-level augmentation)
        if self.augment:
            mel_spec = self.time_mask(mel_spec)
            mel_spec = self.freq_mask(mel_spec)

        label = row["target"]
        return mel_spec.squeeze(0), label