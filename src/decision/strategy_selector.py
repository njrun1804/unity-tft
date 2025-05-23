from dataclasses import dataclass
import numpy as np

@dataclass
class EquityMetrics:
    mu_hat: float     # expected % return / day
    sigma_hat: float
    sharpe: float

@dataclass
class WheelMetrics:
    premium_yield: float   # annualised %
    delta: float
    sigma_hat: float
    capital_at_risk: float
    sharpe: float

def pick_strategy(eq: EquityMetrics,
                  wh: WheelMetrics,
                  cfg) -> str:
    # 1. Sharpe hurdle
    if max(eq.sharpe, wh.sharpe) < cfg.min_sharpe:
        return "EXIT"

    # 2. Compare with margin
    if wh.sharpe - eq.sharpe >= cfg.delta_sharpe \
       and wh.capital_at_risk <= cfg.cash_budget:
        return "WHEEL"
    if eq.sharpe - wh.sharpe >= cfg.delta_sharpe:
        return "EQUITY"

    # 3. Tie-break on μ̂ sign
    return "EQUITY" if eq.mu_hat >= 0 else "EXIT"
