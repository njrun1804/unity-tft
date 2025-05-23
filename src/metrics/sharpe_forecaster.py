from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class AlphaInputs:
    mu_tft: float      # % expected return over H days
    mu_skew: float
    mu_news: float
    ewma_vol: float    # annualised %
    tft_p10: float | None = None
    tft_p90: float | None = None

_ANNUAL = np.sqrt(252)     # daily ↔ annual scale

# baseline & earnings-window dynamic weights
def weight_fn(ts) -> dict[str, float]:
    import pandas as pd, datetime as dt
    earnings_dates = [pd.Timestamp("2025-08-07")]  # drop future dates here
    if any(abs((ts - d).days) <= 10 for d in earnings_dates):
        return dict(tft=0.55, skew=0.30, news=0.15)
    return dict(tft=0.65, skew=0.15, news=0.20)

def ex_ante_sharpe(x: AlphaInputs, ts, w_fn=weight_fn) -> float:
    w = w_fn(ts)
    mu_hat = w["tft"]*x.mu_tft + w["skew"]*x.mu_skew + w["news"]*x.mu_news

    # pick conservative vol
    if x.tft_p10 is not None and x.tft_p90 is not None:
        sigma_q = (x.tft_p90 - x.tft_p10)/2 / 0.8416
        sigma = max(x.ewma_vol, sigma_q)
    else:
        sigma = x.ewma_vol

    return (mu_hat * _ANNUAL) / (sigma * np.sqrt(_ANNUAL))
