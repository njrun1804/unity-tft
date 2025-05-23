#!/usr/bin/env python
"""
Ingest TOS *watch-list* CSVs into the feature store.

Usage:
    python scripts/ingest_watchlist.py \
        --source-dir data/raw \
        --dest-root data/feature_store \
        --log-level DEBUG
"""
import argparse, logging
from pathlib import Path

from scripts.ingest_common import ingest_one

def main(src_dir: Path, dest_root: Path, log_level: str):
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO),
                        format="%(levelname)s %(message)s")
    csvs = sorted(p for p in src_dir.glob("*watchlist*.csv") if p.is_file())
    if not csvs:
        logging.info("No watch-list CSVs found.")
        return
    for p in csvs:
        ingest_one(p, dest_root, "tos_watchlist")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", default="data/raw")
    ap.add_argument("--dest-root", default="data/feature_store")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    main(Path(args.source_dir), Path(args.dest_root), args.log_level)
