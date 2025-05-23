#!/usr/bin/env python
"""
midday_ingest.py – turn TOS watchlist + option chain exports
into one JSON bundle ChatGPT can read.

Usage:
    python src/midday_ingest.py \
        --watchlist ~/Downloads/tos_watchlist.csv \
        --options   ~/Downloads/tos_option_chain.csv \
        --out       data/factors/u_midday.json
"""

import json, argparse, pandas as pd
from datetime import datetime
from pathlib import Path
import subprocess
import sys

# ─── IO helpers ────────────────────────────────────────────────────────────────
def find_parquet_for_csv(csv_fp: Path, feature_store_root: Path) -> Path:
    """Given a CSV filename, find the corresponding Parquet file in the feature store."""
    # Infer type
    fn = csv_fp.name.lower()
    if "watchlist" in fn:
        subdir = "tos_watchlist"
    elif "optionchain" in fn or "stockandoptionquote" in fn:
        subdir = "tos_option_chain"
    else:
        raise ValueError(f"Unknown CSV type for {csv_fp}")
    # Find most recent Parquet in feature store subdir
    fs_dir = feature_store_root / subdir
    candidates = sorted(fs_dir.glob("*/part_*.parquet"), reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No Parquet files found in {fs_dir}")
    return candidates[0]

def load_watchlist(fp: Path, feature_store_root: Path) -> pd.DataFrame:
    # Run robust ingest if needed
    subprocess.run([
        sys.executable, "scripts/ingest.py",
        "--source-dir", str(fp.parent),
        "--dest-root", str(feature_store_root)
    ], check=True)
    pq_fp = find_parquet_for_csv(fp, feature_store_root)
    return pd.read_parquet(pq_fp)

def load_chain(fp: Path, feature_store_root: Path) -> pd.DataFrame:
    subprocess.run([
        sys.executable, "scripts/ingest.py",
        "--source-dir", str(fp.parent),
        "--dest-root", str(feature_store_root)
    ], check=True)
    pq_fp = find_parquet_for_csv(fp, feature_store_root)
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
    wl    = load_watchlist(args.watchlist, feature_store_root)
    chain = load_chain(args.options, feature_store_root)
    blob  = crunch(wl, chain)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(blob, f, indent=2)
    print(f"✅ wrote {args.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--watchlist", type=Path, required=True, help="Path to TOS watchlist CSV")
    ap.add_argument("--options",   type=Path, required=True, help="Path to TOS option chain CSV")
    ap.add_argument("--out",       type=Path, required=True, help="Output JSON path")
    main(ap.parse_args())
