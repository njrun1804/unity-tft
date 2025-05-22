"""Pipeline to orchestrate ETL → edge DataFrame for news-graph risk features."""
import pandas as pd
from pathlib import Path
from .features import add_recency_weights, aggregate_edges

def run_news_edge_pipeline(news_path, now, half_life_days=7, window_days=3, cutoff_date=None):
    """
    Load news data, clean, engineer features, and aggregate for graph construction.
    Args:
        news_path: Path to raw news CSV or parquet
        now: pd.Timestamp, current time for recency calculation
        half_life_days: int, decay half-life
        window_days: int, rolling window for aggregation
        cutoff_date: pd.Timestamp, do not use data after this date
    Returns:
        edge_df: DataFrame with aggregated edge features
    """
    if str(news_path).endswith(".parquet"):
        news_df = pd.read_parquet(news_path)
    else:
        news_df = pd.read_csv(news_path)
    # Basic cleaning: dropna, enforce dtypes
    news_df = news_df.dropna(subset=["date", "ticker_a", "ticker_b", "sentiment"])
    news_df["date"] = pd.to_datetime(news_df["date"])
    news_df["sentiment"] = news_df["sentiment"].astype(float)
    # Feature engineering
    news_df = add_recency_weights(news_df, now, half_life_days=half_life_days)
    # Aggregate
    edge_df = aggregate_edges(news_df, window_days=window_days, cutoff_date=cutoff_date)
    return edge_df
