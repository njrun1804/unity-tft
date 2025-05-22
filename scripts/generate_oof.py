import yaml, argparse, pandas as pd, mlflow
from pathlib import Path
from src.utils.cv_utils import time_series_folds
from src.trainers import train_tft, train_gnn

def main(cfg):
    oof = []  # collect dfs: [date, symbol, tft_pred, gnn_pred, y]
    for fold, (train_dates, val_dates) in enumerate(time_series_folds(cfg['dates'], cfg['n_folds'])):
        # --- train base models -----------------
        tft_run = train_tft(train_dates, val_dates, cfg)   # returns mlflow run_id
        gnn_run = train_gnn(train_dates, val_dates, cfg)
        # --- grab predictions ------------------
        def dl(run_id):
            return pd.read_parquet(mlflow.artifacts.download_artifacts(run_id, "val_preds.parquet"))
        df = dl(tft_run).merge(dl(gnn_run), on=["date","symbol"], suffixes=("_tft","_gnn"))
        oof.append(df)
    oof_df = pd.concat(oof).reset_index(drop=True)
    oof_df.to_parquet("oof_preds.parquet")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/oof.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    main(cfg)
