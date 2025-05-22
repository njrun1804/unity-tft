import json
from pathlib import Path


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
    import numpy as np
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
