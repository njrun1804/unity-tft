from pathlib import Path
import torch
import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer
from typing import List, Dict, Any, Optional

# Optionally import Darts and CatBoost if available
try:
    from darts import TimeSeries
    from darts.models import NBEATSxModel, CatBoostModel
    _DARTS_AVAILABLE = True
except ImportError:
    _DARTS_AVAILABLE = False

# --- Helper for Darts/Tabular models ---
def predict_df(model, ts, name, params, covariates=None):
    pr = model.predict(n=params["predict_len"], past_covariates=covariates)
    df = pr.pd_dataframe()
    df.columns = [f"{name}_q{c}" for c in df.columns]
    return df

# --- Main ensemble_predict ---
def ensemble_predict(
    checkpoints_dir: Optional[Path] = None,
    dataset=None,
    dataloader=None,
    top_k: int = 5,
    device: str = "mps" if torch.backends.mps.is_available() else "cpu",
    darts_models: Optional[List[Any]] = None,
    darts_series: Optional[Any] = None,
    darts_covariates: Optional[Any] = None,
    blend: str = "mean",
    params: Optional[Dict] = None,
) -> Dict[str, pd.Series]:
    """
    Run ensemble prediction for TFT (PyTorch Lightning), Darts N-BEATSx, and CatBoost.
    Returns a dict of blended quantile series.
    """
    preds = []
    names = []
    # --- TFT (PyTorch Lightning) ---
    if checkpoints_dir and dataloader is not None:
        ckpts = sorted(checkpoints_dir.glob("epoch*=ckpt"), key=lambda p: p.stat().st_mtime)[:top_k]
        tft_preds = []
        for ckpt in ckpts:
            model = TemporalFusionTransformer.load_from_checkpoint(ckpt, map_location=device)
            model.eval()
            with torch.no_grad():
                batch_pred = model.predict(dataloader, mode="prediction", return_y=False)
            tft_preds.append(batch_pred.cpu())
            del model
        tft_mean = torch.stack(tft_preds).mean(dim=0)
        tft_df = pd.DataFrame(tft_mean.numpy(), columns=[f"tft_q{q}" for q in params.get("quantiles", [0.1,0.5,0.9])])
        preds.append(tft_df)
        names.append("tft")
    # --- Darts models ---
    if _DARTS_AVAILABLE and darts_models and darts_series is not None:
        for model in darts_models:
            name = type(model).__name__.lower().replace("model", "")
            df = predict_df(model, darts_series, name, params, darts_covariates)
            preds.append(df)
            names.append(name)
    # --- Blend ---
    full = pd.concat(preds, axis=1)
    out = {}
    for q in params.get("quantiles", [0.1, 0.5, 0.9]):
        cols = full.filter(regex=f"_q{q}$").columns
        if blend == "mean":
            out[f"P{int(q*100)}"] = full[cols].mean(axis=1)
        # Optionally add weighted blend here
    return out

def save_to_csv(pred_dict: Dict[str, pd.Series], output_path: Path):
    df = pd.DataFrame(pred_dict)
    df.to_csv(output_path, index=False)
