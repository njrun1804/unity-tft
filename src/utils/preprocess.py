import logging
import os
import re

import pandas as pd

LOGGER = logging.getLogger("preprocess")

_TICKER_RE = re.compile(r"([A-Z]{1,5})(?:_|\.|$)")  # crude but works for U_5min.parquet, AAPL.csv, etc.


def _infer_ticker_from_path(path: str | os.PathLike | None) -> str | None:
    if path is None:
        return None
    m = _TICKER_RE.search(os.path.basename(str(path)))
    return m.group(1) if m else None


def preprocess_df(df: pd.DataFrame, *, source_path: str | os.PathLike | None = None) -> pd.DataFrame:
    """
    Centralized preprocessing used by ingest + Prefect flow.

    Parameters
    ----------
    df : pd.DataFrame
    source_path : optional, original file path the DF came from.
                  Only used to guess the ticker if it's missing.
    """
    # 1) Guarantee ticker column -------------------------------------------------
    if "ticker" not in df.columns:
        ticker = _infer_ticker_from_path(source_path) or "UNKNOWN"
        LOGGER.warning(
            "⛑  Added missing `ticker` column with value '%s' (source=%s)",
            ticker,
            source_path or "<in-memory>",
        )
        df["ticker"] = ticker

    # 2) Minimal hygiene ---------------------------------------------------------
    df = df.dropna(how="all")  # no all-NaN rows
    df = df.reset_index(drop=True)

    # cast dtypes you rely on
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype("category")

    return df
