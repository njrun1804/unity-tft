import numpy as np
import pandas as pd

def quantiles(samples: np.ndarray, q=(0.1, 0.5, 0.9)):
    """
    Return a DataFrame of quantile paths: index = horizon step, columns = q% levels.
    samples: shape (n_samples, H)
    """
    qs = np.quantile(samples, q, axis=0)          # shape (len(q), H)
    return pd.DataFrame(qs.T, columns=[f"q{int(x*100)}" for x in q])

def var(samples: np.ndarray, alpha=0.05, current_price: float = None):
    """
    α-level Value-at-Risk on *returns* over the horizon.
    If current_price is passed, convert sample prices → log-returns first.
    """
    if current_price is not None:
        # use terminal price vs today
        rets = np.log(samples[:, -1] / current_price)
    else:
        # already returns
        rets = samples[:, -1]
    return np.quantile(rets, alpha)

def cvar(samples: np.ndarray, alpha=0.05, current_price: float = None):
    """
    Conditional Value-at-Risk (Expected Shortfall).
    """
    if current_price is not None:
        rets = np.log(samples[:, -1] / current_price)
    else:
        rets = samples[:, -1]
    tail = rets[rets <= np.quantile(rets, alpha)]
    return tail.mean()

def prob_hit(samples: np.ndarray, thresh: float, direction="above"):
    """
    Scenario probability: P(price_H ≥ thresh) or ≤ thresh.
    """
    if direction == "above":
        return (samples[:, -1] >= thresh).mean()
    else:
        return (samples[:, -1] <= thresh).mean()
