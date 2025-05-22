"""
Train & save Darts N-BEATSx and CatBoost models.
Usage
-----
python train_darts.py \
    --train_csv data/train.parquet \
    --val_csv   data/val.parquet  \
    --model_dir models/darts      \
    --n_epochs  300
"""
import argparse, os, mlflow
import pandas as pd
from pathlib import Path
from darts import TimeSeries
from darts.models import NBEATSxModel, CatBoostModel
from darts.metrics import mae

def to_series(df, cov_cols=None):
    ts = TimeSeries.from_dataframe(df, time_col="date", value_cols=["close"])
    cov = TimeSeries.from_dataframe(df, time_col="date", value_cols=cov_cols) if cov_cols else None
    return ts, cov

def main(args):
    train_df = pd.read_parquet(args.train_csv)
    val_df   = pd.read_parquet(args.val_csv)

    cov_cols = [c for c in train_df.columns if "lag" in c or "roll" in c]
    train_ts, train_cov = to_series(train_df, cov_cols)
    val_ts,   val_cov   = to_series(val_df,   cov_cols)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri("file:mlruns")
    with mlflow.start_run(run_name="darts_nbeatsx") as run:
        nbx = NBEATSxModel(
            input_chunk_length = 120,
            output_chunk_length = 39,
            n_epochs = args.n_epochs,
            random_state = 42,
            loss = "QuantileLoss",
            quantiles=[0.1,0.5,0.9],
        )
        nbx.fit(train_ts, past_covariates=train_cov, verbose=False)
        mlflow.log_metric("val_mae_nbx", mae(val_ts, nbx.predict(39, past_covariates=val_cov)))
        nbx.save(model_dir / "nbeatsx.pt")

    with mlflow.start_run(run_name="darts_catboost") as run:
        cat = CatBoostModel(
            lags = 120,
            output_chunk_length = 39,
            quantiles=[0.1,0.5,0.9],
            random_state=42,
        )
        cat.fit(train_ts, past_covariates=train_cov, verbose=False)
        mlflow.log_metric("val_mae_cat", mae(val_ts, cat.predict(39, past_covariates=val_cov)))
        cat.save(model_dir / "catboost.ts")

    print(f"✔ Models saved to {model_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", required=True)
    p.add_argument("--val_csv",   required=True)
    p.add_argument("--model_dir", default="models/darts")
    p.add_argument("--n_epochs",  type=int, default=300)
    args = p.parse_args()
    main(args)
