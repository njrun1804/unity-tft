from decision.strategy_selector import EquityMetrics, WheelMetrics, pick_strategy
from types import SimpleNamespace as C

def test_selector_wheel():
    cfg = C(min_sharpe=0.7, delta_sharpe=0.15, cash_budget=0.1)
    eq = EquityMetrics(0.03, 0.18, 1.0)
    wh = WheelMetrics(0.25, 0.2, 0.18, 0.05, 1.2)
    assert pick_strategy(eq, wh, cfg) == "WHEEL"

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
