import numpy as np
from pytorch_forecasting import TimeSeriesDataSet
import duckdb, torch
import pandas as pd
from pathlib import Path
import pyarrow.parquet as pq
from typing import Tuple, List

from pytorch_forecasting import GroupNormalizer

FEATURE_ROOT = Path("data/feature_store")

def make_dataset(df, params):
    # Guarantee ticker column
    if "ticker" not in df.columns:
        raise ValueError("Missing 'ticker' column in dataset.")
    # Identify all lag/rolling features
    lag_cols = [c for c in df.columns if c.startswith('close_lag_')]
    rollmean_cols = [c for c in df.columns if c.startswith('close_rollmean_')]
    rollstd_cols = [c for c in df.columns if c.startswith('close_rollstd_')]
    # --- Explicitly set known/unknown reals ---
    unknown_reals = ["close"] + params.get("extra_unknown_reals", [])
    known_reals = params.get("known_reals", [])
    # Only use explicit columns
    all_cols = ["ticker", "time_idx", "close"] + known_reals + unknown_reals + lag_cols + rollmean_cols + rollstd_cols
    all_cols = list(dict.fromkeys(all_cols))  # remove duplicates, preserve order
    df = df[[c for c in all_cols if c in df.columns]]
    return TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="close",  # continuous target for quantile regression
        group_ids=["ticker"],
        min_encoder_length=params["min_encoder_len"],
        max_encoder_length=params["max_encoder_len"],
        min_prediction_length=params["predict_len"],
        max_prediction_length=params["predict_len"],
        static_categoricals=["ticker"],
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=unknown_reals,  # PATCH: 'close' must be unknown
        target_normalizer=GroupNormalizer(groups=["ticker"]),
        categorical_encoders={},
        add_relative_time_idx=True,
        add_target_scales=True,
        allow_missing_timesteps=True,
    )

def _date_path(date: str) -> Path:
    """Parquet path convention: data/prices/YYYY/MM/DD.parquet"""
    y, m = date[:4], date[5:7]
    return Path(f"data/prices/{y}/{m}/{date}.parquet")

def _to_tensor(df: pd.DataFrame, cols: List[str]) -> torch.Tensor:
    return torch.tensor(df[cols].values, dtype=torch.float)

def load_price_features(date: str,
                        universe: List[str],
                        window: int = 30,
                        feature_cols: List[str] = None
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Unity-only: returns (1, T, F) for 'U' and (1, 1) label.
    """
    if feature_cols is None:
        feature_cols = ["close", "volume", "high", "low"]
    y, m = date[:4], date[5:7]
    pq = Path(f"data/prices/{y}/{m}/{date}.parquet")
    df = duckdb.sql(f"SELECT * FROM read_parquet('{pq}') WHERE symbol = 'U' ORDER BY date").to_df()
    tft_x = torch.tensor(df.tail(window)[feature_cols].values, dtype=torch.float).unsqueeze(0)  # (1, T, F)
    y = torch.tensor([[(df.iloc[-1]["close"] / df.iloc[-2]["close"]) - 1.0]])
    return tft_x, y

def load_feature_slice(source: str, start: str, end: str) -> pd.DataFrame:
    """
    Fast read of partitioned Parquet for a single source between start & end (inclusive).
    """
    date_range = pd.date_range(start, end).strftime("%Y-%m-%d")
    paths = [
        FEATURE_ROOT / source / d / f
        for d in date_range
        for f in (FEATURE_ROOT / source / d).glob("*.parquet")
    ]
    if not paths:
        raise FileNotFoundError(f"No feature data for {source} between {start} and {end}")
    dfs = [pq.read_table(p).to_pandas() for p in paths]
    return pd.concat(dfs, ignore_index=True)
