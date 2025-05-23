import numpy as np

def optimise_sl_tp(samples, fee_bp=10, grid=None):
    if grid is None:
        grid = np.arange(0.95, 1.10, 0.01)  # 95-110 % of spot
    best, best_sharpe = None, -np.inf
    spot = samples[:, 0].mean()
    for sl in grid:
        for tp in grid:
            if sl >= tp: 
                continue
            pnl = np.where(samples[:, -1] <= sl*spot,
                           (sl*spot - spot) - fee_bp/1e4*spot,
                           np.where(samples[:, -1] >= tp*spot,
                                    (tp*spot - spot) - fee_bp/1e4*spot,
                                    samples[:, -1] - spot))
            sharpe = pnl.mean() / (pnl.std() + 1e-9)
            if sharpe > best_sharpe:
                best_sharpe, best = sharpe, (sl, tp)
    return {"stop": best[0], "take": best[1], "sharpe": best_sharpe}
