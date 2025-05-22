import os, torch, pytorch_lightning as L
from pathlib import Path
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from lightning.pytorch.loggers import CSVLogger
from unity_gpu.data import fetch_intraday
from unity_gpu.model import SimpleLSTM

TICKER   = "U"
AV_KEY   = os.environ["AV_KEY"]
CACHE    = Path("data")
SEQ_LEN  = 48          # 48 × 5-min ≈ one trading day
EPOCHS   = 30

def main():
    df = fetch_intraday(TICKER, AV_KEY, CACHE)
    from unity_gpu.data import make_tensor as make_tensor_new
    X, y = make_tensor_new(df)

    # walk-forward split: last 20% as val
    split = int(0.8 * len(X))
    train_ds = torch.utils.data.TensorDataset(X[:split], y[:split])
    val_ds   = torch.utils.data.TensorDataset(X[split:], y[split:])
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True)
    val_dl   = torch.utils.data.DataLoader(val_ds,   batch_size=256)

    # Baseline: zero-return MSE
    baseline = (y[split:] ** 2).mean().item()
    print(f"Zero-return baseline MSE: {baseline:.6g}")

    n_features = X.shape[-1]
    model = SimpleLSTM(n_features=n_features, hidden=128, num_layers=2)
    logger = CSVLogger("logs", name="lstm")
    trainer = L.Trainer(
        max_epochs=EPOCHS,
        logger=logger,
        accelerator="auto",   # => mps on Apple Silicon
    )
    trainer.fit(model, train_dl, val_dl)

    # Save checkpoint
    Path("models").mkdir(exist_ok=True)
    trainer.save_checkpoint(f"models/lstm-v0.ckpt")

if __name__ == "__main__":
    main()
