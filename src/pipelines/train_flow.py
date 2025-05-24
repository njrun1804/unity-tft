from prefect import flow, task, get_run_logger
import subprocess
import datetime
import os
from scipy.stats import ks_2samp
import pandas as pd
from src.utils.preprocess import preprocess_df
from pathlib import Path

RAW_DIR = Path("data/raw")
FEATURE_ROOT = Path("data/feature_store")

@task(retries=2, retry_delay_seconds=30)
def ingest_raw():
    cmd = ["python", "scripts/ingest.py", "--source-dir", str(RAW_DIR), "--dest-root", str(FEATURE_ROOT)]
    subprocess.run(cmd, check=True)

@task
def preprocess_data(train_path: str, val_path: str, out_train: str, out_val: str):
    logger = get_run_logger()
    logger.info("[Prefect] Preprocessing data for training and drift checks...")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    # Use centralized preprocessing with ticker guarantee
    train_df = preprocess_df(pd.read_parquet(train_path), source_path=train_path)
    val_df = preprocess_df(pd.read_parquet(val_path), source_path=val_path)
    # Log columns and abort if ticker missing
    for name, df in [("train", train_df), ("val", val_df)]:
        logger.info(f"{name} columns: {list(df.columns)}")
        if "ticker" not in df.columns:
            logger.error(f"{name} set missing 'ticker' column!")
            raise ValueError(f"{name} set missing 'ticker' column!")
    train_df.to_parquet(out_train)
    val_df.to_parquet(out_val)
    logger.info(f"Saved preprocessed train to {out_train}, val to {out_val}")
    return out_train, out_val

@task
def run_training():
    logger = get_run_logger()
    logger.info("[Prefect] Launching training script with Hydra config...")
    result = subprocess.run([
        "python", "train_tft.py"
    ], capture_output=True, text=True)
    logger.info(result.stdout)
    logger.error(result.stderr)
    if result.returncode != 0:
        raise RuntimeError("Training failed!")
    return True

@task
def log_artifacts():
    logger = get_run_logger()
    logger.info("[Prefect] Logging training artifacts...")
    
    # Copy model artifacts to a standard location for deployment
    from pathlib import Path
    import shutil
    
    artifacts_dir = Path("outputs/training_artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy TFT model checkpoints if they exist
    model_dir = Path("models/tft")
    if model_dir.exists():
        for ckpt_file in model_dir.glob("*.ckpt"):
            shutil.copy2(ckpt_file, artifacts_dir / ckpt_file.name)
            logger.info(f"Copied {ckpt_file.name} to artifacts directory")
    
    # Copy training logs
    logs_dir = Path("lightning_logs")
    if logs_dir.exists():
        latest_version = max(logs_dir.glob("version_*"), default=None, key=lambda x: x.stat().st_mtime)
        if latest_version:
            shutil.copytree(latest_version, artifacts_dir / "latest_logs", dirs_exist_ok=True)
            logger.info("Copied latest training logs to artifacts directory")
    
    return True

@task
def notify_success():
    logger = get_run_logger()
    logger.info("[Prefect] Training pipeline completed successfully!")
    # Add email, Slack, or webhook notification here
    return True

@task
def notify_failure():
    logger = get_run_logger()
    logger.error("[Prefect] Training pipeline failed!")
    # Add email, Slack, or webhook notification here
    return True

def preprocess_for_drift(df):
    # Apply the same renaming as in build_dataset
    df = df.rename(columns={
        '1. open': 'open',
        '2. high': 'high',
        '3. low': 'low',
        '4. close': 'close',
        '5. volume': 'volume',
    })
    return df

@task
def check_drift(train_path: str, new_path: str, feature: str = "close", alpha: float = 0.01):
    logger = get_run_logger()
    logger.info(f"[Prefect] Running drift check on feature '{feature}'...")
    train_df = pd.read_parquet(train_path)
    new_df = pd.read_parquet(new_path)
    train_df = preprocess_for_drift(train_df)
    new_df = preprocess_for_drift(new_df)
    if feature not in train_df.columns or feature not in new_df.columns:
        logger.error(f"Feature '{feature}' not found in one of the dataframes after preprocessing.")
        return False
    train_vals = train_df[feature].dropna()
    new_vals = new_df[feature].dropna()
    stat, pval = ks_2samp(train_vals, new_vals)
    logger.info(f"KS test statistic: {stat:.4f}, p-value: {pval:.4g}")
    if pval < alpha:
        logger.warning(f"Drift detected in feature '{feature}' (p < {alpha})!")
        return False
    logger.info(f"No significant drift detected in feature '{feature}'.")
    return True

@flow(name="TFT Training Orchestration Pipeline")
def tft_training_flow():
    try:
        # Preprocess data and get new file paths
        pre_train, pre_val = preprocess_data(
            train_path="data/U_5min.parquet",
            val_path="data/U_5min.parquet",
            out_train="data/pre_train.parquet",
            out_val="data/pre_val.parquet"
        )
        # Drift check before training
        drift_ok = check_drift(
            train_path=pre_train,
            new_path=pre_val,
            feature="close",
            alpha=0.01
        )
        if not drift_ok:
            get_run_logger().warning("Drift detected, but continuing with training (customize as needed)")
        # Patch: run_training expects raw file, so temporarily copy preprocessed to expected location
        import shutil
        shutil.copy(pre_train, "data/U_5min.parquet")
        run_training()
        log_artifacts()
        notify_success()
    except Exception as e:
        notify_failure()
        raise e

@flow(name="unity_e2e")
def unity_e2e_flow():
    try:
        ingest_raw()
        pre_train, pre_val = preprocess_data(
            train_path="data/U_5min.parquet",
            val_path="data/U_5min.parquet",
            out_train="data/pre_train.parquet",
            out_val="data/pre_val.parquet"
        )
        drift_ok = check_drift(
            train_path=pre_train,
            new_path=pre_val,
            feature="close",
            alpha=0.01
        )
        if not drift_ok:
            get_run_logger().warning("Drift detected, but continuing with training (customize as needed)")
        import shutil
        shutil.copy(pre_train, "data/U_5min.parquet")
        run_training()
        log_artifacts()
        notify_success()
    except Exception as e:
        notify_failure()
        raise e

if __name__ == "__main__":
    tft_training_flow()
    # Creates a deployment called "unity_e2e_dev" and runs it hourly.
    # Change the cron or interval as you like.
    from datetime import timedelta

    unity_e2e_flow.serve(
        name="unity_e2e_dev",
        interval=timedelta(hours=1),      # ← or cron="0 * * * *"
    )
