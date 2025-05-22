import numpy as np
from pytorch_forecasting import TimeSeriesDataSet
import duckdb, torch
import pandas as pd
from pathlib import Path
from typing import Tuple, List

def make_dataset(df, params):
    # Identify all lag/rolling features
    lag_cols = [c for c in df.columns if c.startswith('close_lag_')]
    rollmean_cols = [c for c in df.columns if c.startswith('close_rollmean_')]
    rollstd_cols = [c for c in df.columns if c.startswith('close_rollstd_')]
    time_varying_known_reals = ["close"] + lag_cols + rollmean_cols + rollstd_cols
    return TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="up_next_day",
        group_ids=["grp"],
        min_encoder_length=params["min_encoder_len"],
        max_encoder_length=params["max_encoder_len"],
        min_prediction_length=params["predict_len"],
        max_prediction_length=params["predict_len"],
        static_categoricals=[],
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=["close"],
        target_normalizer=None,
        categorical_encoders={},
        add_relative_time_idx=True,
        add_target_scales=False,
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
