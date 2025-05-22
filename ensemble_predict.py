import os
import torch
import pandas as pd
from pathlib import Path
from pytorch_forecasting import TemporalFusionTransformer
from src.utils.data_utils import make_dataset
from train_tft import preprocess_df

MODEL_DIR = Path("models")
DATA_PATH = "data/U_5min.parquet"  # or your test/val file
OUT_PATH = "ensemble_preds.csv"
TOP_K = 5

# Find top-k checkpoints (sorted by modification time, newest first)
ckpts = sorted(MODEL_DIR.glob("*.ckpt"), key=os.path.getmtime, reverse=True)[:TOP_K]
if not ckpts:
    raise FileNotFoundError("No checkpoints found in models/ directory.")
print(f"Using checkpoints: {[str(c) for c in ckpts]}")

# Load data and preprocess
params = {}  # TODO: Load your config or fill as needed
raw_df = pd.read_parquet(DATA_PATH)
df = preprocess_df(raw_df, params)
ds = make_dataset(df, params)
dl = ds.to_dataloader(batch_size=512, num_workers=2)

# Ensemble predictions
all_preds = []
for ckpt in ckpts:
    model = TemporalFusionTransformer.load_from_checkpoint(ckpt)
    model.eval()
    preds = []
    for batch in dl:
        x, _ = batch[0], batch[1]
        with torch.no_grad():
            out = model(x)
            preds.append(out.cpu())
    all_preds.append(torch.cat(preds, dim=0))

ensemble_preds = torch.stack(all_preds).mean(dim=0).numpy()
pd.DataFrame(ensemble_preds).to_csv(OUT_PATH, index=False)
print(f"Ensemble predictions saved to {OUT_PATH}")
