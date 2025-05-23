import ray.tune as tune
from ray.tune import Checkpoint
import tempfile, os, torch
from training.checkpoint_utils import save_ckpt
from ray.tune.search.optuna import OptunaSearch
from pathlib import Path
import pandas as pd, hydra
from train_tft import train_once, preprocess_df

def objective(config):
    # Load fixed data once per worker
    train_df = pd.read_parquet(config["train_csv"])
    val_df   = pd.read_parquet(config["val_csv"])
    score = train_once(train_df, val_df, config, Path(config["model_dir"]))
    tune.report({"val_loss": score}, checkpoint=save_ckpt(model=None))  # model can be None if not used

search_space = {
    "hidden_size": tune.choice([64, 96, 128]),
    "lstm_layers": tune.choice([1, 2, 3]),
    "dropout":     tune.uniform(0.1, 0.4),
    "lr":          tune.loguniform(1e-4, 1e-2),
    "batch_size":  256,
    "min_encoder_len": 30,
    "max_encoder_len": 39,
    "predict_len":     1,
    "early_stop_patience": 8,
    "num_workers": 6,
    "max_epochs": 30,
    "seed": tune.randint(1, 10_000),
    "train_csv": "data/train.parquet",
    "val_csv":   "data/val.parquet",
    "model_dir": "models/ray",
}

algo = OptunaSearch(metric="val_loss", mode="min")
tune.run(
    objective,
    name="ray_sweep",
    config=search_space,
    num_samples=40,
    search_alg=algo,
    resources_per_trial={"cpu": 6, "gpu": 1},   # gpu=1 works for MPS
    local_dir="ray_results",
)
