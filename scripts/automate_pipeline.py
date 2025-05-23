"""
Automated pipeline: Ingest CSVs → Parquet → Preprocess → Inference → Save predictions.
Usage:
    python scripts/automate_pipeline.py
"""
import sys
import logging
from pathlib import Path
import importlib.util
import pandas as pd
import torch
import json
from training.objectives import PriceLSTM

# --- CONFIG ---
RAW_CSV_DIR = Path("data/raw")
FEATURE_STORE_DIR = Path("data/feature_store")
OPTION_CHAIN_DIR = FEATURE_STORE_DIR / "tos_option_chain"
WATCHLIST_DIR = FEATURE_STORE_DIR / "tos_watchlist"
MODEL_PATH = Path("models/lstm_best_epoch50.pt")
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOG = logging.getLogger("automate_pipeline")

# --- 1. Ingest CSVs to Parquet ---
def run_ingest():
    LOG.info("Step 1: Ingesting CSVs...")
    spec = importlib.util.spec_from_file_location("ingest", "scripts/ingest.py")
    ingest = importlib.util.module_from_spec(spec)
    sys.modules["ingest"] = ingest
    spec.loader.exec_module(ingest)
    ingest.ingest(RAW_CSV_DIR, FEATURE_STORE_DIR)
    LOG.info("Ingest complete.")

# --- 2. Load latest Parquet data ---
def load_latest_parquet(parquet_dir):
    if not parquet_dir.exists():
        return pd.DataFrame()
    all_files = sorted(parquet_dir.glob("*/part_*.parquet"), reverse=True)
    if not all_files:
        return pd.DataFrame()
    # Load most recent file(s) for today
    today_dir = parquet_dir / pd.Timestamp.now().strftime("%Y-%m-%d")
    files = list(today_dir.glob("part_*.parquet"))
    if not files:
        files = all_files[:1]
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# --- 3. Preprocess and join features ---
def preprocess(option_chain_df, watchlist_df):
    # Example: join on ticker, add any feature engineering here
    if option_chain_df.empty or watchlist_df.empty:
        return option_chain_df
    merged = option_chain_df.merge(watchlist_df[["ticker", "price", "timestamp"]], on="ticker", how="left", suffixes=("", "_wl"))
    # Remove duplicates after join
    merged = merged.drop_duplicates(subset=["ticker", "expiry", "strike", "cp"])
    # Add more preprocessing as needed
    return merged

# --- Load model ---
def load_model(model_path, input_dim, hidden_size, num_layers):
    """Rebuild model *then* load weights."""
    model = PriceLSTM(
        input_dim=input_dim,
        hidden_size=hidden_size,
        lstm_layers=num_layers,
        dropout=0.0  # Set to your training value if needed
    )
    state = torch.load(model_path, map_location="cpu")
    # Lightning checkpoints wrap state-dicts differently; handle both.
    if "state_dict" in state:
        state = {k.replace("model.", ""): v for k, v in state["state_dict"].items()}
    model.load_state_dict(state)
    model.eval()
    return model

# --- 4. Model inference ---
def run_inference(df, model_path):
    if df.empty:
        LOG.warning("No data for inference.")
        return pd.DataFrame()
    try:
        # Load persisted feature list
        with open("artifacts/feature_list.json") as f:
            feature_list = json.load(f)
        # Ensure all required features are present
        missing = [f for f in feature_list if f not in df.columns]
        if missing:
            raise ValueError(f"Missing required features for inference: {missing}")
        X = df[feature_list].fillna(0).astype(float).values
        model = load_model(model_path, input_dim=X.shape[1], hidden_size=256, num_layers=2)
        with torch.no_grad():
            y_hat = model(torch.tensor(X, dtype=torch.float32)).squeeze().numpy()
        df_out = df.copy()
        df_out["prediction"] = y_hat
        return df_out
    except Exception as e:
        LOG.exception("Inference failed – aborting.")
        raise

# --- 5. Save predictions ---
def save_predictions(df):
    if df.empty:
        LOG.warning("No predictions to save.")
        return None
    # Coerce known numeric columns to float
    for col in ["strike", "bid", "ask", "iv", "delta", "gamma", "theta", "vega", "oi", "vol", "price", "net_change", "mark_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    out_path = OUTPUTS_DIR / f"predictions_{pd.Timestamp.now().strftime('%Y-%m-%d_%H%M%S')}.parquet"
    df.to_parquet(out_path, index=False)
    LOG.info(f"Predictions saved to {out_path}")
    return out_path

if __name__ == "__main__":
    run_ingest()
    option_chain_df = load_latest_parquet(OPTION_CHAIN_DIR)
    watchlist_df = load_latest_parquet(WATCHLIST_DIR)
    features_df = preprocess(option_chain_df, watchlist_df)
    predictions_df = run_inference(features_df, MODEL_PATH)
    save_predictions(predictions_df)
    LOG.info("Pipeline complete.")
