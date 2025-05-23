from src.metrics.sharpe_forecaster import AlphaInputs, ex_ante_sharpe
import pandas as pd

def test_sharpe_monotonic():
    base = AlphaInputs(0.02,0.0,0.0,0.15)
    ts = pd.Timestamp("2025-05-23")
    S1 = ex_ante_sharpe(base, ts=ts)
    base2 = AlphaInputs(0.04,0.0,0.0,0.15)
    S2 = ex_ante_sharpe(base2, ts=ts)
    assert S2 > S1

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
