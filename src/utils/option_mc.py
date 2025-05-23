import numpy as np
import pandas as pd
from scipy.stats import norm

def bs_price(S, K, T, r, sigma, call=True):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if call:
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def simulate_portfolio_pnl(samples: np.ndarray,
                           positions: pd.DataFrame,
                           r: float,
                           sigma: float,
                           dt_years: float):
    """
    samples: (N, H) price matrix → we mark to market at horizon H
    positions: DataFrame with columns {symbol, side, strike, qty, call_bool}
    r, sigma: risk-free + assumed vol for Black-Scholes re-pricing
    dt_years: time-to-expiry in years (same for all legs here)
    Returns an array of length N = sample paths of portfolio P/L.
    """
    S_T = samples[:, -1]                         # terminal prices
    payoffs = np.zeros(len(S_T))
    for _, p in positions.iterrows():
        # market value now → theoretical BS
        price_now = bs_price(S_T, p.strike, dt_years, r, sigma, p.call_bool)
        direction  = 1 if (p.side == "long") else -1
        payoffs   += direction * p.qty * price_now
    return payoffs
