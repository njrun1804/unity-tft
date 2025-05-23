import numpy as np

def kelly_fraction(returns: np.ndarray):
    """
    returns: array of terminal *multiplicative* returns, e.g. 1.10 for +10 %.
    """
    if np.any(returns <= 0):
        raise ValueError("Kelly formula requires positive returns.")
    log_ret = np.log(returns)
    mean, var = log_ret.mean(), log_ret.var(ddof=1)
    return max(0.0, mean / var)

def f_kelly(returns, f=0.5):
    return f * kelly_fraction(returns)
