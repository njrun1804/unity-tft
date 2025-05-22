import os
import torch
import pandas as pd
from pathlib import Path
from pytorch_forecasting import TemporalFusionTransformer
from train_tft import (
    build_dataset, SEQ_LEN, HORIZON, DEVICE, DATA_DIR, MODEL_DIR
)

def latest_prob_up():
    df = build_dataset(os.getenv("AV_KEY")).tail(SEQ_LEN + HORIZON)
    dataset = torch.load(DATA_DIR / "ts_dataset.pt")  # reuse encoder settings
    x, _ = dataset.to_dataloader(df.tail(SEQ_LEN)).dataset[0]
    model = TemporalFusionTransformer.load_from_checkpoint(
        MODEL_DIR / "tft-latest.ckpt"
    ).to(DEVICE)
    with torch.no_grad():
        logits = model(x.unsqueeze(0).to(DEVICE))
        prob   = logits.sigmoid()[0, 0, -1].item()
    return prob

if __name__ == "__main__":
    p = latest_prob_up()
    decision = "🔥  RUN the wheel" if p >= 0.55 else "⏸  Sit out"
    print(f"Prob up next day: {p:.2%}  →  {decision}")
