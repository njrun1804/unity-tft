from pathlib import Path
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import numpy as np
import torch

def fetch_intraday(
    ticker: str,
    key: str,
    cache_dir: Path = Path("data"),
    interval: str = "5min",
) -> pd.DataFrame:
    """Download intraday OHLCV from Alpha Vantage and cache as Parquet."""
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{ticker}_{interval}.parquet"

    if cache_file.exists():
        return pd.read_parquet(cache_file)

    ts = TimeSeries(key=key, output_format="pandas")
    df, _ = ts.get_intraday(symbol=ticker, interval=interval, outputsize="full")
    df.to_parquet(cache_file)
    return df

# ---------- NEW ----------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    price  = df["4. close"].astype(float)
    volume = df["5. volume"].astype(float)

    feat = pd.DataFrame(index=df.index)
    feat["ret_pp"]    = price.pct_change() * 1000  # scale up to ×1000
    feat["vol_5"]     = feat["ret_pp"].rolling(5).std()
    feat["volpct_20"] = (volume / volume.rolling(20).median()) - 1

    # 14-bar RSI
    delta = price.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    roll_up  = up.ewm(alpha=1/14, adjust=False).mean()
    roll_dn  = down.ewm(alpha=1/14, adjust=False).mean()
    rs       = roll_up / roll_dn.replace(0, np.nan)
    feat["rsi_14"] = 100 - (100 / (1 + rs))

    # feat["ivr_chg"] = iv_rank.diff()          # <-- leave until Tradier is in
    return feat.dropna()
# -------------------------

def make_tensor(df: pd.DataFrame):
    feat = engineer_features(df)

    X = torch.tensor(feat.values, dtype=torch.float32)
    y = torch.tensor(feat["ret_pp"].shift(-1).dropna().values, dtype=torch.float32)

    X = X[:-1]  # align after shift
    SEQ_LEN = 48
    # Vectorized sliding window using unfold, then permute to (windows, seq_len, features)
    X_seq = X.unfold(0, SEQ_LEN, 1).permute(0, 2, 1)  # shape: (num_windows, SEQ_LEN, num_features)
    y_seq = y[SEQ_LEN:]
    return X_seq, y_seq
