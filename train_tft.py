import hydra
from omegaconf import DictConfig
import pandas as pd
from pathlib import Path
from src.utils.data_utils import make_dataset
import torch
import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from lightning.pytorch.loggers import MLFlowLogger
import os

def set_seed(seed: int):
    pl.seed_everything(seed, workers=True)


# Helper: preprocess dataframe for required columns
def preprocess_df(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Ensure the raw price frame has the columns required by make_dataset():
      • time_idx – dense integer index ordered by date/time
      • grp      – single‑series group id (0)
      • up_next_day – binary direction label computed at horizon = predict_len
    The function is idempotent: if columns already exist they are left untouched.
    """
    df = df.copy()

    # Sort chronologically so the np.arange index is stable
    if "time_idx" not in df.columns:
        # Pick a reasonable datetime column
        ts_col = "date" if "date" in df.columns else df.columns[0]
        df = df.sort_values(ts_col).reset_index(drop=True)
        df["time_idx"] = np.arange(len(df))

    if "grp" not in df.columns:
        df["grp"] = 0  # univariate series ⇒ single group id

    if "up_next_day" not in df.columns and "close" in df.columns:
        horizon = int(params.get("predict_len", 39))
        df["up_next_day"] = (df["close"].shift(-horizon) > df["close"]).astype(int)

    # Drop final rows with NaNs introduced by shift
    return df.dropna().reset_index(drop=True)

def train_once(train_df, val_df, params: dict, model_dir: Path) -> float:
    required = {"hidden_size", "lstm_layers", "dropout", "lr", "batch_size", "min_encoder_len", "max_encoder_len", "predict_len", "early_stop_patience", "num_workers", "max_epochs", "seed"}
    missing = required - set(params.keys())
    if missing:
        raise ValueError(f"Missing required params: {missing}")
    # Ensure required engineered columns exist
    train_df = preprocess_df(train_df, params)
    val_df   = preprocess_df(val_df,   params)
    set_seed(params.get("seed", 42))
    train_ds = make_dataset(train_df, params)
    val_ds   = make_dataset(val_df,   params)
    num_workers = params.get("num_workers", max(1, os.cpu_count() // 2))
    train_dl = train_ds.to_dataloader(batch_size=params["batch_size"], num_workers=num_workers)
    val_dl   = val_ds.to_dataloader(batch_size=params["batch_size"], num_workers=num_workers)
    model = TemporalFusionTransformer.from_dataset(
        train_ds,
        hidden_size = params["hidden_size"],
        lstm_layers = params["lstm_layers"],
        attention_head_size = params.get("attention_head_size", 4),
        dropout     = params["dropout"],
        loss       = QuantileLoss(),
        learning_rate = params["lr"],
        output_size=1,
        log_interval=50,
    )
    model.save_hyperparameters(ignore=["loss", "logging_metrics"])
    logger = MLFlowLogger(experiment_name="unity_tft", tracking_uri="file:mlruns")
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath=model_dir,
        filename="best",
        save_top_k=1,
        mode="min"
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    trainer = pl.Trainer(
        max_epochs = params["max_epochs"],
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=params["early_stop_patience"]
            ),
            checkpoint_cb,
            lr_monitor
        ],
        enable_progress_bar=False,
        default_root_dir=model_dir,
        accelerator="auto",
        devices="auto",
        logger=logger,
    )
    trainer.fit(model, train_dl, val_dl)
    model_dir.mkdir(exist_ok=True)
    ckpt_path = str(model_dir / "best.ckpt")
    trainer.save_checkpoint(ckpt_path)
    rmse = trainer.callback_metrics["val_loss"].item() ** 0.5
    return rmse

def main(cfg: DictConfig):
    params = dict(cfg)
    # Remove non-param keys
    params.pop('data', None)
    params.pop('model_dir', None)
    train_df = pd.read_parquet(cfg.data.train_csv)
    val_df = pd.read_parquet(cfg.data.val_csv)
    train_df = preprocess_df(train_df, params)
    val_df   = preprocess_df(val_df,   params)
    score = train_once(train_df, val_df, params, Path(cfg.model_dir))
    print(f"Validation RMSE: {score:.4f}")

if __name__ == "__main__":
    import sys
    if '--params_json' in sys.argv:
        # Legacy CLI mode for backward compatibility
        import json, argparse
        p = argparse.ArgumentParser()
        p.add_argument("--params_json", required=True)
        p.add_argument("--train_csv",   required=True)
        p.add_argument("--val_csv",     required=True)
        p.add_argument("--model_dir",   default="models/tmp")
        args = p.parse_args()
        params = json.loads(Path(args.params_json).read_text())
        train_df = pd.read_parquet(args.train_csv)
        val_df   = pd.read_parquet(args.val_csv)
        train_df = preprocess_df(train_df, params)
        val_df   = preprocess_df(val_df,   params)
        score = train_once(train_df, val_df, params, Path(args.model_dir))
        print(f"Validation RMSE: {score:.4f}")
    else:
        # Hydra config mode
        main = hydra.main(config_path="src/config", config_name="default", version_base=None)(main)
        main()
