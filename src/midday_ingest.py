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
    return {
        "run_timestamp": datetime.utcnow().isoformat(),
        "ticker": "U",
        "features": {
            "price":        row["price"],
            "net_change":   row["net_change"],
            "mark_pct":     row["mark_pct"],
            "iv_rank":      round(float(iv_rank), 3),
            "p_up_1d":      0.55,   # ← replace with your TFT output any time
        },
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
