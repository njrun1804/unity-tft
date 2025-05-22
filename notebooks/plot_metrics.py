import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

# Find the latest metrics.csv file
glob_path = "../logs/lstm/version_*/metrics.csv"
files = glob(glob_path)
if not files:
    raise FileNotFoundError(f"No metrics.csv found at {glob_path}")
metrics_path = sorted(files)[-1]

# Load metrics
df = pd.read_csv(metrics_path)

# Plot train and val loss if available
plt.figure(figsize=(8, 5))
if 'train_loss' in df.columns:
    plt.plot(df['step'], df['train_loss'], label='Train Loss')
if 'val_loss' in df.columns:
    plt.plot(df['step'], df['val_loss'], label='Val Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.tight_layout()
plt.show()
