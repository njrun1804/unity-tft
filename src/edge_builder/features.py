"""Generate functions that operate on a news_df and **never** look past the cutoff_date variable (to avoid look-ahead leakage). Return a feature_df with only numeric cols ready for model input."""
import numpy as np
import pandas as pd

def add_recency_weights(df: pd.DataFrame, now: pd.Timestamp, half_life_days: int = 7) -> pd.DataFrame:
    """
    Add exponential time-decay weights to news events.
    """
    recency = (now - pd.to_datetime(df["date"])).dt.days.clip(lower=0)
    df["decay_weight"] = np.exp(-recency / half_life_days)
    return df

def aggregate_edges(df: pd.DataFrame, window_days: int = 3, cutoff_date=None) -> pd.DataFrame:
    """
    Aggregate news events into edge features for each ticker pair and window.
    """
    if cutoff_date is not None:
        df = df[pd.to_datetime(df["date"]) <= cutoff_date]
    df["window"] = pd.to_datetime(df["date"]).dt.floor(f"{window_days}D")
    # One-hot encode event_type for graph input
    event_types = pd.get_dummies(df["event_type"], prefix="event")
    df = pd.concat([df, event_types], axis=1)
    agg_dict = {
        "sentiment": ["mean", "std"],
        "decay_weight": "sum"
    }
    for col in event_types.columns:
        agg_dict[col] = "sum"
    agg = (
        df.groupby(["window", "ticker_a", "ticker_b"])
          .agg(agg_dict)
    )
    agg.columns = ["_".join(x) for x in agg.columns]
    agg = agg.reset_index()
    return agg
