import os, tempfile, torch
from ray import tune
from ray.tune import Checkpoint
from training.checkpoint_utils import save_ckpt
from models.lstm import PriceLSTM
from training.datamodule import build_dataloaders
from training.train_loop import train_one_epoch, evaluate
import numpy as np
from pathlib import Path
import random

def lstm_objective(cfg):
    # 1) Reproducibility
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) Data & model
    train_path = Path(__file__).parent.parent / "data" / "train.parquet"
    val_path = Path(__file__).parent.parent / "data" / "val.parquet"
    train_loader, val_loader = build_dataloaders(
        train_path=train_path,
        val_path=val_path,
        batch=64,
    )
    sample_X, _ = next(iter(train_loader))
    input_dim = sample_X.shape[-1]

    model = PriceLSTM(
        input_dim=input_dim,
        hidden_size=cfg["hidden_size"],
        lstm_layers=cfg["lstm_layers"],
        dropout=cfg["dropout"],
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])

    best_val = float("inf")
    for epoch in range(1, cfg["epochs"] + 1):
        train_one_epoch(model, train_loader, opt, device)
        val_loss = evaluate(model, val_loader, device)

        # save best checkpoint using Ray Tune
        if val_loss < best_val:
            best_val = val_loss
            tune.report({"loss": val_loss, "epoch": epoch}, checkpoint=save_ckpt(model))
        else:
            tune.report({"loss": val_loss, "epoch": epoch})
