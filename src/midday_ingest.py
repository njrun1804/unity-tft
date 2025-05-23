#!/usr/bin/env python
"""
midday_ingest.py – turn Polygon watchlist + option chain Parquet files
into one JSON bundle ChatGPT can read.

Usage:
    python src/midday_ingest.py \
        --out data/factors/u_midday.json
"""

import json, argparse, pandas as pd
from datetime import datetime
from pathlib import Path

# ─── IO helpers ────────────────────────────────────────────────────────────────
def get_latest_parquet(folder: Path) -> Path:
    files = sorted(folder.glob('part_*.parquet'), key=lambda f: f.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No Parquet files found in {folder}")
    return files[0]

def load_watchlist(feature_store_root: Path) -> pd.DataFrame:
    pq_fp = get_latest_parquet(feature_store_root / "polygon_watchlist")
    return pd.read_parquet(pq_fp)

def load_chain(feature_store_root: Path) -> pd.DataFrame:
    pq_fp = get_latest_parquet(feature_store_root / "polygon_option_chain")
    return pd.read_parquet(pq_fp)

# ─── Feature crunching ────────────────────────────────────────────────────────
def crunch(wl: pd.DataFrame, chain: pd.DataFrame) -> dict:
    # Hard-code ticker 'U' for Unity
    row = wl.iloc[0]
    iv_rank = chain["iv"].rank(pct=True).iloc[-1]
    # Convert option chain to a compact list of dicts (include greeks fields)
    option_chain = chain[[
        "expiry", "strike", "cp", "bid", "ask", "iv", "delta", "price",
        "greeks.delta", "greeks.gamma", "greeks.theta", "greeks.vega"
    ]].rename(columns={
        "greeks.delta": "greeks_delta",
        "greeks.gamma": "greeks_gamma",
        "greeks.theta": "greeks_theta",
        "greeks.vega": "greeks_vega"
    }).to_dict(orient="records")

    # Calculate additional statistics for TFT/Optima-style output
    # For demonstration, use dummy values or simple calculations; replace with real model outputs as needed
    # Load latest TFT/Optima predictions from outputs/
    import glob
    pred_files = sorted(glob.glob("outputs/predictions_*.parquet"), reverse=True)
    if pred_files:
        pred_df = pd.read_parquet(pred_files[0])
        # Try to match on ticker and latest timestamp
        pred_row = pred_df[(pred_df['ticker'] == 'U')].sort_values(by='timestamp', ascending=False).iloc[0]
        prediction = float(pred_row.get('p_up_1d_pred', pred_row.get('prediction', 0.55)))
        prediction_std = float(pred_row.get('prediction_std', 0.08))
        quantile_01 = float(pred_row.get('quantile_0.1', 0.48))
        quantile_05 = float(pred_row.get('quantile_0.5', 0.55))
        quantile_09 = float(pred_row.get('quantile_0.9', 0.65))
        confidence_interval_lower = float(pred_row.get('confidence_interval_lower', 0.47))
        confidence_interval_upper = float(pred_row.get('confidence_interval_upper', 0.67))
        error = float(pred_row.get('error', 0.04))
        loss = float(pred_row.get('loss', 0.12))
    else:
        prediction = 0.55
        prediction_std = 0.08
        quantile_01 = 0.48
        quantile_05 = 0.55
        quantile_09 = 0.65
        confidence_interval_lower = 0.47
        confidence_interval_upper = 0.67
        error = 0.04
        loss = 0.12

    return {
        "run_timestamp": datetime.utcnow().isoformat(),
        "ticker": "U",
        "features": {
            "price":        row["price"],
            "net_change":   row["net_change"],
            "mark_pct":     row["mark_pct"],
            "iv_rank":      round(float(iv_rank), 3),
            "p_up_1d":      prediction,   # TFT model output
            "prediction_std": prediction_std,
            "quantile_0.1": quantile_01,
            "quantile_0.5": quantile_05,
            "quantile_0.9": quantile_09,
            "confidence_interval_lower": confidence_interval_lower,
            "confidence_interval_upper": confidence_interval_upper,
            "error": error,
            "loss": loss
        },
        "option_chain": option_chain,
    }

# ─── CLI ───────────────────────────────────────────────────────────────────────
def main(args):
    feature_store_root = Path("data/feature_store")
    wl    = load_watchlist(feature_store_root)
    chain = load_chain(feature_store_root)
    blob  = crunch(wl, chain)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(blob, f, indent=2)
    print(f"✅ wrote {args.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="Output JSON path")
    main(ap.parse_args())
