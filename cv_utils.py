import pandas as pd
from typing import Iterator, Tuple

def rolling_cv(df: pd.DataFrame,
               n_folds: int = 4,
               min_hist: int = 500,
               val_len: int = 39) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Yield (train_df, val_df) pairs.
    • Each successive fold grows the training window.
    • Each validation slice is exactly `val_len` rows (match HORIZON).
    """
    step = (len(df) - min_hist - val_len) // n_folds
    for k in range(n_folds):
        split = min_hist + k * step
        train_df = df.iloc[:split].copy()
        val_df   = df.iloc[split : split + val_len].copy()
        yield train_df, val_df
