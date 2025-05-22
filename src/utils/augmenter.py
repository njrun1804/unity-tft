import torch
from torch.utils.data import Dataset
import random

class Augmenter(Dataset):
    """
    Wrap any TimeSeriesDataset and apply on-the-fly transforms:
      • Random window shift (±3 steps)
      • Gaussian noise on 'target' column
    """
    def __init__(self, base_dataset, shift: int = 3, noise_std: float = 0.01):
        self.base = base_dataset
        self.shift = shift
        self.noise_std = noise_std

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        # ----- shift window -----
        s = random.randint(-self.shift, self.shift)
        if s != 0:
            x["decoder_cont"].roll_(shifts=s, dims=-2)   # shift time dimension
            y = torch.roll(y, shifts=s, dims=-2)
        # ----- Gaussian noise on target -----
        y = y + torch.randn_like(y) * self.noise_std
        return x, y
