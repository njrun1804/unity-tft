from torch.utils.data import DataLoader, Dataset
import pandas as pd, torch
from pathlib import Path
import numpy as np

class PriceDataset(Dataset):
    def __init__(self, parquet_path, seq_len=48, target_col=None):
        if not Path(parquet_path).exists():
            raise FileNotFoundError(
                f"Dataset not found: {parquet_path}. "
                "Did you forget to put the parquet file in the repo or pass the correct path?"
            )
        df = pd.read_parquet(parquet_path)
        # Determine target column
        if target_col is None:
            if "close" in df.columns:
                target_col = "close"
            elif "up_next_day" in df.columns:
                target_col = "up_next_day"
            else:
                # Fallback: use last column (warn user)
                target_col = df.columns[-1]
                print(f"[WARN] No 'close' or 'up_next_day' column found. Using last column '{target_col}' as target.")
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame columns: {df.columns.tolist()}")
        self.target_col = target_col
        self.X, self.y = self.build_sliding_windows(df, seq_len, target_col)

    def build_sliding_windows(self, df, seq_len, target_col):
        X, y = [], []
        for i in range(len(df) - seq_len):
            X_window = df.iloc[i:i+seq_len].to_numpy(dtype=np.float32)
            y_value = df.iloc[i+seq_len][target_col]
            X.append(X_window)
            y.append(float(y_value))
        return X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return torch.tensor(self.X[i], dtype=torch.float32), torch.tensor(self.y[i], dtype=torch.float32)

def build_dataloaders(train_path, val_path, batch=64, num_workers=4):
    train = PriceDataset(Path(train_path).resolve())
    val   = PriceDataset(Path(val_path).resolve())
    train_loader = DataLoader(train, batch_size=batch, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val,   batch_size=batch, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
