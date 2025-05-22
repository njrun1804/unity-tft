"""Write pytest that generates a toy news_df with two tickers, injects a -10 day and +1 day article, and asserts decay_weight monotonically decreases with recency."""
import pandas as pd
import numpy as np
from src.edge_builder.features import add_recency_weights
import pytest

def test_decay_weight_monotonic():
    now = pd.Timestamp("2025-05-22")
    df = pd.DataFrame({
        "date": [now - pd.Timedelta(days=10), now + pd.Timedelta(days=1)],
        "ticker_a": ["A", "A"],
        "ticker_b": ["B", "B"],
        "sentiment": [0.5, -0.2],
        "event_type": ["earnings", "product"]
    })
    df = add_recency_weights(df, now)
    assert df.loc[0, "decay_weight"] > df.loc[1, "decay_weight"]
