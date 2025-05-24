import hydra
from omegaconf import DictConfig
import pandas as pd
from pathlib import Path
from src.utils.data_utils import make_dataset
import torch
# Enable FlashAttention/SDPA if CUDA is available
if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
from torch.utils.data import DataLoader
import numpy as np
import lightning.pytorch as pl
from pytorch_forecasting import TemporalFusionTransformer
from lightning.pytorch.loggers import MLFlowLogger
import os
import math
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import torch.nn as nn
import optuna
from pytorch_forecasting.metrics import QuantileLoss
from torchmetrics.regression import MeanSquaredError

# Device selection for Apple Silicon (MPS) with FP16 fallback
device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.set_float32_matmul_precision("high")  # Apple-Silicon speed-up

# Set OMP/MKL threads for max CPU fan-out
num_threads = os.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["MKL_NUM_THREADS"] = str(num_threads)

def set_seed(seed: int):
    pl.seed_everything(seed, workers=True)


# --- Feature engineering: add lag and rolling-stat features ---
def make_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = df.copy()
    lags = [3, 7, 14, 28]
    for lag in lags:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
    windows = [7, 14, 28]
    for win in windows:
        df[f'close_rollmean_{win}'] = df['close'].rolling(win).mean()
        df[f'close_rollstd_{win}'] = df['close'].rolling(win).std()
    return df

# Helper: preprocess dataframe for required columns
def preprocess_df(df: pd.DataFrame, params: dict = None, *, symbol: str | None = None) -> pd.DataFrame:
    """
    Ensure the raw price frame has the columns required by make_dataset():
      • time_idx – dense integer index ordered by date/time
      • ticker   – group id for univariate/multivariate
      • up_next_day – binary direction label computed at horizon = predict_len
    The function is idempotent: if columns already exist they are left untouched.
    """
    df = df.copy()
    # --- Guarantee ticker exists everywhere ---
    if "ticker" not in df.columns:
        inferred = symbol or Path(getattr(df, "_source", "UNKNOWN.csv")).stem.split("_")[0]
        df = df.assign(ticker=inferred)
    # Sort chronologically so the np.arange index is stable
    if "time_idx" not in df.columns:
        ts_col = "date" if "date" in df.columns else df.columns[0]
        df = df.sort_values(ts_col).reset_index(drop=True)
        df["time_idx"] = np.arange(len(df))
    if "grp" not in df.columns:
        df["grp"] = 0  # univariate series ⇒ single group id
    if "up_next_day" not in df.columns and "close" in df.columns:
        horizon = int(params.get("predict_len", 39)) if params else 39
        df["up_next_day"] = (df["close"].shift(-horizon) > df["close"]).astype(int)
    if params:
        df = make_features(df, params)
    # Drop final rows with NaNs introduced by shift
    return df.dropna().reset_index(drop=True)

class QuantileMSELoss(nn.Module):
    def __init__(self, quantiles=[0.1, 0.5, 0.9], alpha=0.9):
        super().__init__()
        self.quantile_loss = QuantileLoss(quantiles=quantiles)
        self.mse = nn.MSELoss()
        self.alpha = alpha

    def forward(self, y_pred, target):
        return self.alpha * self.quantile_loss(y_pred, target) + (1 - self.alpha) * self.mse(y_pred, target)

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
    # --- DataLoader construction with optional augmentation ---
    num_workers = params.get("num_workers", max(2, math.floor(num_threads / 2)))
    use_aug = params.get("augment", False)
    from src.utils.augmenter import Augmenter
    from torch.utils.data import DataLoader

    def build_dataloader(ds):
        """
        Wrap dataset with Augmenter when requested and return an appropriate
        DataLoader.  If the wrapped object is still a TimeSeriesDataSet we can
        use its built‑in .to_dataloader(); otherwise fall back to the vanilla
        torch DataLoader interface.
        """
        wrapped = Augmenter(ds) if use_aug else ds
        if hasattr(wrapped, "to_dataloader"):
            return wrapped.to_dataloader(
                batch_size=params["batch_size"],
                num_workers=num_workers,
                persistent_workers=num_workers > 0,
            )
        else:
            return DataLoader(
                wrapped,
                batch_size=params["batch_size"],
                num_workers=num_workers,
                persistent_workers=num_workers > 0,
            )

    train_dl = build_dataloader(train_ds)
    val_dl   = build_dataloader(val_ds)
    # Add RMSE metric
    rmse_metric = MeanSquaredError(squared=False).to(device)
    class TFTWithRMSE(TemporalFusionTransformer):
        def validation_step(self, batch, batch_idx):
            out = super().validation_step(batch, batch_idx)
            y, y_hat = batch[1], out["prediction"]
            rmse = rmse_metric(y_hat, y)
            self.log("val_rmse", rmse, prog_bar=True)
            return out
    model = TemporalFusionTransformer(
        train_ds,
        hidden_size = params["hidden_size"],
        lstm_layers = params["lstm_layers"],
        attention_head_size = params.get("attention_head_size", 4),
        dropout     = params["dropout"],
        # --- PATCH: enforce quantile output head ---
        loss       = QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
        output_size=3,  # 3 quantiles for probabilistic output
        quantiles=[0.1, 0.5, 0.9],
        learning_rate = params["lr"],
        log_interval=50,
    )
    # Assert output size matches quantiles
    expected_output_size = len([0.1, 0.5, 0.9])
    assert model.output_size == expected_output_size, (
        f"output_size ({model.output_size}) does not match loss quantiles ({expected_output_size})")
    # Disable checkpoint reload if output head shape mismatches
    # (Remove/ignore checkpoints with output_size != 3)
    checkpoint_dir = Path('models/checkpoints')
    if checkpoint_dir.exists():
        for ckpt in checkpoint_dir.glob('*.ckpt'):
            try:
                state = torch.load(ckpt, map_location='cpu')
                head_shape = state['state_dict']['output_layer.weight'].shape[0]
                if head_shape != 3:
                    ckpt.unlink()
            except Exception:
                continue
    logger = MLFlowLogger(experiment_name="unity_tft", tracking_uri="file:mlruns")
    # Tag MLflow run with git SHA for traceability
    try:
        import mlflow
        git_sha = os.getenv("GIT_SHA", "unknown")
        mlflow.set_tag("git_sha", git_sha)
    except ImportError:
        print("[WARN] mlflow not installed, skipping git_sha tag.")
    early_stop = EarlyStopping(monitor="val_loss", patience=8, mode="min")
    checkpoint = ModelCheckpoint(monitor="val_loss", save_top_k=5, mode="min")
    trainer = pl.Trainer(
        max_epochs = params["max_epochs"],
        callbacks  = [early_stop, checkpoint],
        enable_progress_bar=False,
        default_root_dir=model_dir,
        accelerator = "mps" if device == "mps" else "auto",
        devices=1,
        precision="16-mixed",
        gradient_clip_val=0.5,
        logger=logger,
        accumulate_grad_batches=params.get("accumulate_grad_batches", 1),
    )
    trainer.fit(
        model, train_dl, val_dl, ckpt_path="last",  # Enable auto-resume from last checkpoint
        optimizers=optimizer,
        lr_scheduler_configs={"scheduler": scheduler, "interval": "epoch"},
    )
    model_dir.mkdir(exist_ok=True)
    val_loss = trainer.callback_metrics["val_loss"].item()
    val_rmse = trainer.callback_metrics.get("val_rmse")
    if val_rmse is not None:
        print(f"Validation RMSE: {val_rmse.item():.4f}")
    # --- Ensemble prediction after training ---
    from src.utils.ensemble import ensemble_predict, save_to_csv
    test_dl = val_ds.to_dataloader(batch_size=params["batch_size"])
    ensemble_out = ensemble_predict(Path(trainer.checkpoint_callback.dirpath), val_ds, test_dl)
    save_to_csv(ensemble_out, model_dir / "ensemble_pred.csv")
    # 7. Generate and save position recommendations based on TFT certainty
    from src.utils.position_recommender import recommend_positions, save_recommendations
    # Example: Use ensemble output to compute certainty (e.g., 1 - quantile spread)
    import numpy as np
    if hasattr(ensemble_out, 'columns') and 'q0.9' in ensemble_out.columns and 'q0.1' in ensemble_out.columns:
        certainty = 1 - (ensemble_out['q0.9'] - ensemble_out['q0.1']) / (np.abs(ensemble_out['q0.5']) + 1e-6)
    else:
        certainty = 0.8  # fallback default
    recommendations = recommend_positions(ensemble_out, certainty)
    save_recommendations(recommendations, Path(model_dir)/"position_recommendations.json")
    print(f"[INFO] Position recommendations saved to position_recommendations.json.")
    # Patch TFT attention to use FlashMHA (FlashAttention/SDPA)
    try:
        from src.unity_gpu.flash_attention import FlashMHA
        if hasattr(model, 'network') and hasattr(model.network, 'self_attention'):  # pytorch_forecasting >=0.10
            model.network.self_attention = FlashMHA(
                d_model=model.network.self_attention.d_model,
                n_head=model.network.self_attention.n_head,
                dropout=model.network.self_attention.dropout,
            )
    except Exception as e:
        print(f"[WARN] Could not patch TFT attention for FlashAttention: {e}")
    return val_loss

# --- Optuna sweep utilities -------------------------------------------------
def sample_params(trial):
    """Define hyper‑parameter search space for Temporal Fusion Transformer."""
    return {
        "hidden_size":  trial.suggest_categorical("hidden_size",  [64, 96, 128, 192]),
        "lstm_layers":  trial.suggest_int("lstm_layers", 1, 4),
        "dropout":      trial.suggest_float("dropout", 0.05, 0.5),
        "lr":           trial.suggest_loguniform("lr", 1e-4, 1e-2),
        "batch_size":   trial.suggest_categorical("batch_size", [64, 128, 256]),
        # Fixed/default values inherited from original config
        "min_encoder_len": 30,
        "max_encoder_len": 120,
        "predict_len":     39,
        "early_stop_patience": 8,
        "num_workers":  max(2, math.floor(num_threads / 2)),
        "max_epochs":   50,
        "seed":         42,
    }

def objective(trial, train_df, val_df, base_model_dir):
    """Optuna objective: train once with sampled params and return val_loss."""
    params = sample_params(trial)
    trial_dir = Path(base_model_dir) / f"trial_{trial.number}"
    loss = train_once(train_df, val_df, params, trial_dir)
    return loss

def run_optuna(train_csv: str, val_csv: str, model_dir: str, n_trials: int = 30):
    """Entry‑point callable for hyper‑parameter sweep."""
    train_df = pd.read_parquet(train_csv)
    val_df   = pd.read_parquet(val_csv)
    study = optuna.create_study(direction="minimize",
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5,
                                                                    n_warmup_steps=10))
    study.optimize(lambda t: objective(t, train_df, val_df, model_dir),
                   n_trials=n_trials,
                   show_progress_bar=True)
    print("Best trial:", study.best_trial.number, "loss", study.best_trial.value)
    print("Best params:", study.best_trial.params)
    # save all trials for later inspection
    trial_df = study.trials_dataframe()
    trial_df.to_csv(Path(model_dir) / "optuna_trials.csv", index=False)
    return study

def main(cfg: DictConfig):
    params = dict(cfg)
    # Remove non-param keys
    params.pop('data', None)
    params.pop('model_dir', None)
    train_df = pd.read_parquet(cfg.data.train_csv)
    train_df._source = cfg.data.train_csv  # for ticker inference
    val_df = pd.read_parquet(cfg.data.val_csv)
    val_df._source = cfg.data.val_csv
    train_df = preprocess_df(train_df, params)
    val_df   = preprocess_df(val_df,   params)
    score = train_once(train_df, val_df, params, Path(cfg.model_dir))
    print(f"Validation loss: {score:.4f}")

if __name__ == "__main__":
    import sys
    if '--params_json' in sys.argv:
        # Direct CLI mode for programmatic training
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
        print(f"Validation loss: {score:.4f}")
    elif '--optuna_trials' in sys.argv:
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--optuna_trials", type=int, default=30)
        p.add_argument("--train_csv",   required=True)
        p.add_argument("--val_csv",     required=True)
        p.add_argument("--model_dir",   default="models/optuna")
        args = p.parse_args()
        run_optuna(args.train_csv, args.val_csv, args.model_dir, args.optuna_trials)
    elif '--ensemble_only' in sys.argv:
        # Ensemble-only mode: skip TFT training, just blend predictions
        # 1. Import ensemble utilities and parse CLI args
        from src.utils.ensemble import ensemble_predict, save_to_csv
        import argparse, json
        p = argparse.ArgumentParser()
        p.add_argument("--model_dir", default="models/tft")
        p.add_argument("--val_csv", required=True)
        p.add_argument("--quantiles", nargs="*", type=float, default=[0.1,0.5,0.9])
        p.add_argument("--blend", default="mean")
        args = p.parse_args()
        val_df = pd.read_parquet(args.val_csv)
        # 2. Try to load Darts models and covariates if available
        darts_models = []
        darts_series = None
        darts_covariates = None
        blend_metadata = {"included_models": [], "blend": args.blend, "quantiles": args.quantiles}
        try:
            from darts import TimeSeries
            from darts.models import NBEATSxModel, CatBoostModel
            nbx_path = Path(args.model_dir) / "nbeatsx.pt"
            if nbx_path.exists():
                model_nbx = NBEATSxModel.load(nbx_path)
                darts_models.append(model_nbx)
                blend_metadata["included_models"].append("nbeatsx")
            cat_path = Path(args.model_dir) / "catboost.ts"
            if cat_path.exists():
                model_cat = CatBoostModel.load(cat_path)
                darts_models.append(model_cat)
                blend_metadata["included_models"].append("catboost")
            if len(val_df) > 0 and "date" in val_df.columns:
                darts_series = TimeSeries.from_dataframe(val_df, time_col="date", value_cols=["close"])
                cov_cols = [c for c in val_df.columns if "lag" in c or "roll" in c]
                if cov_cols:
                    darts_covariates = TimeSeries.from_dataframe(val_df, time_col="date", value_cols=cov_cols)
        except ImportError:
            print("[WARN] Darts not installed. Skipping Darts models in ensemble.")
        # 3. Build TFT dataset and dataloader
        from src.utils.data_utils import make_dataset
        params = {"batch_size": 256, "quantiles": args.quantiles, "predict_len": 39}
        val_ds = make_dataset(val_df, params)
        test_dl = val_ds.to_dataloader(batch_size=params["batch_size"])
        # 4. Check for TFT checkpoints
        checkpoints_dir = Path(args.model_dir)/"checkpoints"
        if checkpoints_dir.exists() and any(checkpoints_dir.glob("*.ckpt")):
            blend_metadata["included_models"].append("tft")
        # 5. If no models found, warn and exit gracefully
        if not blend_metadata["included_models"]:
            print("[WARN] No models found for ensemble. Exiting.")
            with open(Path(args.model_dir)/"ensemble_blend_config.json", "w") as f:
                json.dump(blend_metadata, f, indent=2)
            exit(0)
        # 6. Run ensemble prediction and save outputs
        out = ensemble_predict(
            checkpoints_dir=checkpoints_dir,
            dataloader=test_dl,
            params=params,
            blend=args.blend,
            darts_models=darts_models if darts_models else None,
            darts_series=darts_series,
            darts_covariates=darts_covariates,
        )
        save_to_csv(out, Path(args.model_dir)/"ensemble_blend.csv")
        with open(Path(args.model_dir)/"ensemble_blend_config.json", "w") as f:
            json.dump(blend_metadata, f, indent=2)
        print(f"[INFO] Ensemble blend complete. Models included: {blend_metadata['included_models']}. Blend config saved to ensemble_blend_config.json.")
        # 7. After blending, compute certainty and direction
        # Example assumes out DataFrame with P10, P50, P90 columns
        interval_width = out["q0.9"] - out["q0.1"]
        certainty = 1 - (interval_width / interval_width.max())
        certainty = certainty.clip(0, 1)
        # Add direction (for horizon=1, use diff; for multi-step, use expected move)
        out["direction"] = out["q0.5"].diff().fillna(0).apply(np.sign)
        # Pass both pred_df and certainty to the recommender
        from src.utils.position_recommender import recommend_positions, save_recommendations
        recommendations = recommend_positions(out, certainty)
        save_recommendations(recommendations, Path(args.model_dir)/"position_recommendations.json")
        print(f"[INFO] Position recommendations saved to position_recommendations.json.")
    else:
        # Hydra config mode
        main = hydra.main(config_path="src/config", config_name="default", version_base=None)(main)
        main()
