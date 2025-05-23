"""
Polygon-only Parquet ingester for Unity pipeline.
Usage:
    python scripts/ingest.py --source-dir data/raw --dest-root data/feature_store
    # (No longer supports CSV ingestion)
"""
from __future__ import annotations

import argparse, logging, os
from datetime import datetime
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

LOG = logging.getLogger("ingest")

# Remove all TOS/CSV schemas and logic
# Only keep risk schema if still needed elsewhere
SCHEMAS = {
    "tft_risk": {
        "timestamp": "timestamp",
        "ticker": "ticker",
        "risk_score": "risk_score",
        "certainty": "certainty",
    },
}

ARROW_SCHEMAS = {
    "tft_risk": pa.schema([
        ("timestamp", pa.timestamp("ms")),
        ("ticker", pa.string()),
        ("risk_score", pa.float64()),
        ("certainty", pa.float64()),
    ]),
}

def write_partitioned_arrow(df: pd.DataFrame, dest_dir: Path, schema_key: str, partition_cols=None):
    if partition_cols is None:
        partition_cols = []
    table = pa.Table.from_pandas(df, preserve_index=False, schema=ARROW_SCHEMAS[schema_key])
    hashval = str(abs(hash(df.to_string())))[:8]
    out_dir = dest_dir / datetime.utcnow().strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_dir / f"part_{hashval}.parquet", compression="zstd")

# Remove all CSV/legacy ingest logic and CLI
if __name__ == "__main__":
    print("This script no longer ingests CSVs. Use scripts/polygon_ingest.py for all data ingestion.")
