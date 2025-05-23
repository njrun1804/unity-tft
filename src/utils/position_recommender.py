import json
from pathlib import Path
import pandas as pd
from metrics.sharpe_forecaster import AlphaInputs, ex_ante_sharpe
from decision.strategy_selector import EquityMetrics, WheelMetrics, pick_strategy
import numpy as np


def load_positions_config(config_path=None):
    """
    Load the positions.json config file.
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "positions.json"
    with open(config_path, "r") as f:
        return json.load(f)


def _scale_from_ladder(cert, ladder):
    for rung in ladder:
        lo = rung.get("min", float("-inf"))
        hi = rung.get("max", float("inf"))
        if (lo is None or cert >= lo) and (hi is None or cert < hi):
            return rung["scale"]
    return 1.0


def recommend_positions(predictions, certainty, config=None):
    if config is None:
        config = load_positions_config()
    meta = config.get("meta", {})
    ladder = meta.get("certainty_ladder", [])
    delta_shift_cfg = meta.get("certainty_delta_shift", {"slope": 0.3})
    base_delta_lo, base_delta_hi = meta.get("portfolio_delta_target", [0.6, 0.8])
    out = {}
    for idx, row in predictions.iterrows():
        cert = float(certainty[idx] if hasattr(certainty, "__getitem__") else certainty)
        scale = _scale_from_ladder(cert, ladder)
        base_contracts = row.get("base_contracts", 1)
        size = int(np.floor(base_contracts * scale))
        shift = delta_shift_cfg.get("slope", 0.3) * (cert - 0.5)
        delta_lo = base_delta_lo + shift
        delta_hi = base_delta_hi + shift
        going_long = row.get("direction", 1) > 0
        pos_delta = row.get("current_delta", 0)
        if pos_delta < delta_lo or (going_long and scale > 1):
            action = "INCREASE"
        elif pos_delta > delta_hi or (not going_long and scale > 1):
            action = "DECREASE"
        else:
            action = "HOLD"
        out[row.get("position_id", idx)] = {
            "action": action,
            "size": size,
            "certainty": round(cert, 3),
            "delta_band": [round(delta_lo, 2), round(delta_hi, 2)],
            "predicted_move": round(row.get("q0.5", 0), 3)
        }
    return out


def save_recommendations(recommendations, out_path):
    with open(out_path, "w") as f:
        json.dump(recommendations, f, indent=2)


def recommend_size(row, cfg, ts):
    # Sharpe for equity
    inputs = AlphaInputs(
        mu_tft=row["mu_tft"],
        mu_skew=row["mu_skew"],
        mu_news=row["mu_news"],
        ewma_vol=row["ewma_vol"],
        tft_p10=row.get("tft_p10"),
        tft_p90=row.get("tft_p90"),
    )
    S_eq = ex_ante_sharpe(inputs, ts)
    # --- wheel metrics ---
    S_wh = row["wheel_sharpe"]
    wheel = WheelMetrics(
        premium_yield=row["premium_yield"],
        delta=row["delta"],
        sigma_hat=row["ewma_vol"],
        capital_at_risk=row["capital_at_risk"],
        sharpe=S_wh,
    )
    strategy = pick_strategy(
        EquityMetrics(row["mu_tft"], row["ewma_vol"], S_eq),
        wheel,
        cfg,
    )
    if strategy == "EQUITY":
        tgt = cfg.base_size * S_eq / cfg.sharpe_target
    elif strategy == "WHEEL":
        tgt = 0.0    # equity leg handled via wheel
    else:
        tgt = 0.0
    tgt = float(np.clip(tgt, -cfg.max_size, cfg.max_size))
    return dict(
        ticker=row["ticker"],
        strategy=strategy,
        sharpe_equity=S_eq,
        sharpe_wheel=S_wh,
        target_w=tgt,
    )


cache_path = Path("path_to_your_parquet_file.parquet")
df_cache = pd.read_parquet(cache_path)
