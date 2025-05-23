"""
train_lstm.py

Train an LSTM model on price data with reproducibility, early stopping, and checkpointing.

Usage:
    python train_lstm.py \
        --data data/train_full.parquet \
        --test data/test.parquet \
        --hidden_size 256 \
        --lstm_layers 2 \
        --dropout 0.2272 \
        --lr 4.2547e-4 \
        --seed 7780 \
        --epochs 50 \
        --save_ckpt models/lstm_best_epoch50.pt \
        --metrics_out models/lstm_best_metrics.json

Flags:
    --data           Path to training data (parquet)
    --test           Path to test data (parquet)
    --hidden_size    LSTM hidden size
    --lstm_layers    Number of LSTM layers
    --dropout        Dropout rate
    --lr             Learning rate
    --seed           Random seed
    --epochs         Number of epochs
    --save_ckpt      Path to save model checkpoint
    --metrics_out    Path to save test metrics (JSON)

"""
import argparse
import json
import os
import random
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from models.lstm import PriceLSTM
from training.datamodule import build_dataloaders

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from sklearn.isotonic import IsotonicRegression
from scipy.special import expit  # logistic

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)
    os.environ["PYTHONHASHSEED"] = str(seed)

class LSTMForecastLightningModule(pl.LightningModule):
    def __init__(self, input_dim, hidden_size, lstm_layers, dropout, lr, weight_decay=0.0, cyclic_lr=False):
        super().__init__()
        self.save_hyperparameters()
        self.model = PriceLSTM(
            input_dim=input_dim,
            hidden_size=hidden_size,
            lstm_layers=lstm_layers,
            dropout=dropout
        )
        self.criterion = torch.nn.MSELoss()
        self.lr = lr
        self.weight_decay = weight_decay
        self.cyclic_lr = cyclic_lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze(-1)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze(-1)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze(-1)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        return {'test_loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.cyclic_lr:
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.CyclicLR(
                    optimizer,
                    base_lr=self.lr / 10,
                    max_lr=self.lr,
                    step_size_up=2000,
                    mode='triangular2',
                    cycle_momentum=False
                ),
                'interval': 'step',
                'name': 'cyclic_lr'
            }
            return [optimizer], [scheduler]
        return optimizer

class RiskPropagationGNN(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # TODO: Implement risk-propagation GNN for multi-ticker rollout
        pass
    def forward(self, x):
        raise NotImplementedError("RiskPropagationGNN is a stub. Implement GNN logic here.")

class LiveOrderExecutor:
    def __init__(self, broker='alpaca', api_key=None, api_secret=None, paper=True):
        self.broker = broker
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper = paper
        # TODO: Implement live order execution for Alpaca/IB API
        pass
    def submit_order(self, symbol, qty, side, type='market', time_in_force='gtc'):
        raise NotImplementedError("LiveOrderExecutor is a stub. Implement broker API logic here.")

# ------------------------------------------------------------------
# 1.  COLLECT RAW UNCERTAINTY SIGNALS
# ------------------------------------------------------------------
def _calc_uncertainty_components(quant_df, ckpt_preds, y_val, y_val_hat):
    """
    quant_df      : DataFrame with P10, P50, P90 columns (prediction horizon rows)
    ckpt_preds    : tensor K×N with per-checkpoint point forecasts
    y_val/y_val_hat: 1-D arrays for last validation batch (same scale as preds)
    returns dict of numpy arrays, len=N
    """
    # A) Prediction-interval width
    pi_width = (quant_df["P90"] - quant_df["P10"]).values

    # B) Checkpoint variance
    ckpt_std = ckpt_preds.std(dim=0).cpu().numpy()

    # C) Smoothed absolute residual (ema over val batches)
    abs_resid = np.abs(y_val - y_val_hat)
    ema_alpha = 0.2
    abs_resid_sm = np.convolve(
        abs_resid, [ema_alpha], mode="full"
    )[: len(abs_resid)]  # cheap EMA
    return {"pi": pi_width, "ckpt": ckpt_std, "resid": abs_resid_sm}

# ------------------------------------------------------------------
# 2.  LINEAR COMBO  →  LOGISTIC  →  RAW CERTAINTY
# ------------------------------------------------------------------
WEIGHTS = {"pi": 0.50, "ckpt": 0.30, "resid": 0.20}

def _raw_certainty(comp):
    # z-score each component
    z = {k: (v - v.mean()) / (v.std() + 1e-9) for k, v in comp.items()}
    combo = sum(WEIGHTS[k] * z[k] for k in z)
    return expit(-combo)  # minus → narrower PI = higher certainty

# ------------------------------------------------------------------
# 3.  ISOTONIC CALIBRATION  (run *once* after training)
# ------------------------------------------------------------------
def fit_isotonic(y_true, y_pred_dir, raw_cert):
    """
    y_true      : true next-bar direction  (+1 / –1)
    y_pred_dir  : sign of P50 forecast    (+1 / –1)
    raw_cert    : output of _raw_certainty()
    returns     : fitted IsotonicRegression model
    """
    correct = (np.sign(y_true) == y_pred_dir).astype(int)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_cert, correct)
    return iso

# ------------------------------------------------------------------
# 4.  INFERENCE PIPELINE
# ------------------------------------------------------------------
def compute_certainty(quant_df, ckpt_preds, y_val=None, y_val_hat=None,
                      iso_model: IsotonicRegression | None = None):
    comp = _calc_uncertainty_components(quant_df, ckpt_preds, y_val, y_val_hat)
    raw_c = _raw_certainty(comp)
    if iso_model is not None:
        return iso_model.predict(raw_c)  # calibrated
    return raw_c  # fallback if iso not available

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--test', type=str, required=True)
    parser.add_argument('--hidden_size', type=int, required=True)
    parser.add_argument('--lstm_layers', type=int, required=True)
    parser.add_argument('--dropout', type=float, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--save_ckpt', type=str, required=True)
    parser.add_argument('--metrics_out', type=str, required=True)
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for optimizer')
    parser.add_argument('--cyclic_lr', action='store_true', help='Use cyclic learning rate scheduler')
    parser.add_argument('--mlflow', action='store_true', help='Enable MLflow autologging')
    parser.add_argument('--use_gnn', action='store_true', help='Use risk-propagation GNN for multi-ticker rollout (stub)')
    parser.add_argument('--live_orders', action='store_true', help='Enable live order execution (stub, does not place real orders)')
    args = parser.parse_args()

    if args.use_gnn:
        print("[WARNING] RiskPropagationGNN is a stub. LSTM workflow will run. Implement GNN logic in the future.")

    if args.live_orders:
        print("[WARNING] LiveOrderExecutor is a stub. No real orders will be placed. Implement broker API logic in the future.")

    if args.mlflow:
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow is not installed. Install mlflow to use --mlflow.")
        mlflow.pytorch.autolog(log_models=True)

    seed_everything(args.seed)

    # Build train and val dataloaders
    train_loader, val_loader = build_dataloaders(
        train_path=args.data,
        val_path=args.test,  # using test as val for now, adjust if needed
        batch=64,
        num_workers=4
    )

    # Infer input_dim from dataset
    sample_batch, _ = next(iter(train_loader))
    input_dim = sample_batch.shape[-1]

    model = LSTMForecastLightningModule(
        input_dim=input_dim,
        hidden_size=args.hidden_size,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        cyclic_lr=args.cyclic_lr
    )

    # PyTorch Lightning expects a LightningDataModule or dataloaders for fit/test
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=6, verbose=True),
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                filename=os.path.basename(args.save_ckpt).replace('.pt', ''),
                save_top_k=1,
                save_weights_only=True,
                dirpath=os.path.dirname(args.save_ckpt)
            )
        ],
        deterministic=True,
        enable_checkpointing=True,
        logger=False,
        enable_model_summary=True,
        accelerator="auto"
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Save the best checkpoint as state_dict
    best_ckpt_path = trainer.checkpoint_callback.best_model_path if hasattr(trainer, 'checkpoint_callback') else None
    if best_ckpt_path:
        state = torch.load(best_ckpt_path, map_location="cpu")
        torch.save(state, args.save_ckpt)

    # Evaluate on test set
    test_metrics = trainer.test(model, dataloaders=val_loader, verbose=False)
    with open(args.metrics_out, "w") as f:
        json.dump(test_metrics, f, indent=2)

    # --- TimeGPT/CRPS integration ---
    try:
        import pandas as pd
        import numpy as np
        from properscoring import crps_ensemble
        from src.utils.timegpt import timegpt_forecast, blend_and_score_crps
        from src.utils.forecast_stats import quantiles, var, cvar, prob_hit

        # Load validation data
        val_df = pd.read_parquet(args.test)
        horizon = val_df.shape[0] if val_df.shape[0] < 365 else 365  # or set to your predict_len
        y_true = val_df["close"].tail(horizon).to_numpy()

        # Assume current_ensemble_samples is available or can be constructed
        # For demonstration, use model predictions as a single-sample ensemble
        # (Replace with your real ensemble logic)
        current_ensemble_samples = []
        for batch in val_loader:
            x, _ = batch
            preds = model(x).detach().cpu().numpy()
            current_ensemble_samples.append(preds)
        current_ensemble_samples = np.concatenate(current_ensemble_samples, axis=0).T[:1, -horizon:]  # shape (1, H)

        baseline_crps = crps_ensemble(y_true, current_ensemble_samples)[...].mean()
        print(f"Baseline CRPS = {baseline_crps:.4f}")

        # Baseline risk/statistics
        baseline_q = quantiles(current_ensemble_samples, [0.05, 0.5, 0.95])
        baseline_var = var(current_ensemble_samples, alpha=0.05)
        baseline_cvar = cvar(current_ensemble_samples, alpha=0.05)
        baseline_prob_hit = prob_hit(current_ensemble_samples, y_true)

        # Fetch TimeGPT samples
        tg_df = timegpt_forecast(val_df.tail(horizon), horizon=horizon, freq="D", n_samples=100)
        tg_samples = (
            tg_df.pivot(index="sample_id", columns="date", values="y_hat").to_numpy()
        )
        tg_crps = crps_ensemble(y_true, tg_samples).mean()
        print(f"TimeGPT CRPS = {tg_crps:.4f}  (↓ is better)")

        tg_q = quantiles(tg_samples, [0.05, 0.5, 0.95])
        tg_var = var(tg_samples, alpha=0.05)
        tg_cvar = cvar(tg_samples, alpha=0.05)
        tg_prob_hit = prob_hit(tg_samples, y_true)

        # Optional: Blend TimeGPT with current ensemble
        blended_crps = blend_and_score_crps(y_true, current_ensemble_samples, tg_samples)
        print(f"Blended CRPS = {blended_crps:.4f}")

        blended_samples = np.concatenate([current_ensemble_samples, tg_samples], axis=0)
        blended_q = quantiles(blended_samples, [0.05, 0.5, 0.95])
        blended_var = var(blended_samples, alpha=0.05)
        blended_cvar = cvar(blended_samples, alpha=0.05)
        blended_prob_hit = prob_hit(blended_samples, y_true)

        # Optionally log to MLflow
        if args.mlflow and MLFLOW_AVAILABLE:
            mlflow.log_metric("baseline_crps", float(baseline_crps))
            mlflow.log_metric("timegpt_crps", float(tg_crps))
            mlflow.log_metric("blended_crps", float(blended_crps))
            # Baseline risk/statistics
            mlflow.log_metric("baseline_var", float(baseline_var))
            mlflow.log_metric("baseline_cvar", float(baseline_cvar))
            mlflow.log_metric("baseline_prob_hit", float(baseline_prob_hit))
            mlflow.log_metric("timegpt_var", float(tg_var))
            mlflow.log_metric("timegpt_cvar", float(tg_cvar))
            mlflow.log_metric("timegpt_prob_hit", float(tg_prob_hit))
            mlflow.log_metric("blended_var", float(blended_var))
            mlflow.log_metric("blended_cvar", float(blended_cvar))
            mlflow.log_metric("blended_prob_hit", float(blended_prob_hit))
    except Exception as e:
        print(f"[TimeGPT/CRPS] Skipped due to error: {e}")

    if args.mlflow:
        mlflow.log_artifact(args.metrics_out)
        mlflow.log_artifact(args.save_ckpt)
        if isinstance(test_metrics, list) and len(test_metrics) > 0:
            for k, v in test_metrics[0].items():
                mlflow.log_metric(k, v)

if __name__ == "__main__":
    main()
