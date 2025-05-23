"""
CSV → Parquet ingester for Unity-only pipeline.
Usage:
    python scripts/ingest.py --source-dir data/raw --dest-root data/feature_store
"""
from __future__ import annotations

import argparse, csv, hashlib, itertools, logging, os, re, shutil, sys
from datetime import datetime
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor

try:
    import pandera as pa_validate
    PANDERA_AVAILABLE = True
except ImportError:
    PANDERA_AVAILABLE = False

LOG = logging.getLogger("ingest")
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(levelname)s %(message)s")

SCHEMAS = {
    "tos_watchlist": {
        "Symbol": "ticker",
        "Last": "price",
        "Net Chg": "net_change",
        "Mark %": "mark_pct",
        "QuoteTime": "timestamp",
    },
    "tos_option_chain": {
        "UnderlyingSymbol": "ticker",
        "QuoteTime": "timestamp",
        "Expiry": "expiry",
        "Strike": "strike",
        "CallPut": "cp",
        "Bid": "bid",
        "Ask": "ask",
        "ImpliedVolatility": "iv",
        "Delta": "delta",
        "Gamma": "gamma",
        "Theta": "theta",
        "Vega": "vega",
        "OpenInterest": "oi",
        "Volume": "vol",
    },
    "tft_risk": {
        "timestamp": "timestamp",
        "ticker": "ticker",
        "risk_score": "risk_score",
        "certainty": "certainty",
    },
}

ARROW_SCHEMAS = {
    "tos_watchlist": pa.schema([
        ("ticker", pa.string()),
        ("price", pa.float64()),
        ("net_change", pa.float64()),
        ("mark_pct", pa.float64()),
        ("timestamp", pa.timestamp("ms")),
    ]),
    "tos_option_chain": pa.schema([
        ("ticker", pa.string()),
        ("timestamp", pa.timestamp("ms")),
        ("expiry", pa.string()),
        ("strike", pa.float64()),
        ("cp", pa.string()),
        ("bid", pa.float64()),
        ("ask", pa.float64()),
        ("iv", pa.float64()),
        ("delta", pa.float64()),
        ("gamma", pa.float64()),
        ("theta", pa.float64()),
        ("vega", pa.float64()),
        ("oi", pa.int64()),
        ("vol", pa.int64()),
    ]),
    "tft_risk": pa.schema([
        ("timestamp", pa.timestamp("ms")),
        ("ticker", pa.string()),
        ("risk_score", pa.float64()),
        ("certainty", pa.float64()),
    ]),
}

ALT_HEADER_MAP = {
    "option_chain_custom": {
        "Symbol": "UnderlyingSymbol",
        "Expiration": "Expiry",
        "Type": "CallPut",
        "Strike": "Strike",
        "Last": "Bid",
        "Bid": "Bid",
        "Ask": "Ask",
        "Volume": "Volume",
        "Open Int": "OpenInterest",
        "IV": "ImpliedVolatility",
        "Delta": "Delta",
        "Gamma": "Gamma",
        "Theta": "Theta",
        "Vega": "Vega",
        "Rho": "Rho",
    },
    "watchlist_custom": {
        "Symbol": "Symbol",
        "Last": "Last",
        "Net Chng": "Net Chg",
        "Bid": "Bid",
        "Ask": "Ask",
    },
}

def detect_header_and_type(row):
    if set(["Symbol", "Type", "Expiration", "Strike", "Last", "Bid", "Ask"]).issubset(set(row)):
        return "option_chain_custom"
    if set(["Symbol", "Last", "Net Chg"]).issubset(set(row)):
        return "watchlist_custom"
    return None

def split_by_expiry(raw_lines):
    out, current, expiry = [], [], None
    exp_re = re.compile(r'^\d{1,2} [A-Z]{3} \d{2}')
    for ln in raw_lines:
        if exp_re.match(ln):
            if current: out.append((expiry, current)); current=[]
            expiry = exp_re.match(ln).group(0)
        elif ln.strip():
            current.append(ln)
    if current: out.append((expiry, current))
    return out

def canonical_columns(schema_key):
    return list(SCHEMAS[schema_key].values())

def robust_read_csv(path: Path, schema_key: str) -> pd.DataFrame:
    """
    1. Cleans BOM, whitespace, <empty> tokens.
    2. Auto-detects header row by intersecting with expected OR alt header sets.
    3. Allows thousand-separators and stray '%' signs in numerics.
    """
    expected = set(SCHEMAS[schema_key])
    alts = set().union(*(ALT_HEADER_MAP.get(h, {}).keys() for h in ALT_HEADER_MAP))
    header = None
    rows = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        lines = [ln.lstrip("\ufeff") for ln in fh]
        # Option-chain: split by expiry section if needed
        if schema_key == "tos_option_chain":
            for expiry, chunk in split_by_expiry(lines):
                # Find header in chunk
                rdr = csv.reader(chunk)
                for row in rdr:
                    if not row or all(c.strip() == '' for c in row):
                        continue
                    cols = set(row)
                    if (cols >= expected) or (cols & alts):
                        header = row
                        break
                if header is None:
                    continue
                # DictReader for section
                rdr = csv.DictReader(chunk, fieldnames=header)
                # Skip to header
                for rec in rdr:
                    if rec[header[0]] == header[0]:
                        break
                for rec in rdr:
                    if len(rec) != len(header):
                        continue
                    clean = {k: (None if v in ('', '<empty>') else v.replace(',', '').rstrip('%')) for k, v in rec.items()}
                    clean['expiry'] = expiry
                    rows.append(clean)
        else:
            rdr = csv.reader(lines)
            for row in rdr:
                if not row or all(c.strip() == '' for c in row):
                    continue
                cols = set(row)
                if (cols >= expected) or (cols & alts):
                    header = row
                    break
            if header is None:
                LOG.warning("%s: header not found", path.name)
                return pd.DataFrame(columns=canonical_columns(schema_key))
            rdr = csv.DictReader(lines, fieldnames=header)
            for rec in rdr:
                if rec[header[0]] == header[0]:
                    break
            for rec in rdr:
                if len(rec) != len(header):
                    continue
                clean = {k: (None if v in ('', '<empty>') else v.replace(',', '').rstrip('%')) for k, v in rec.items()}
                rows.append(clean)
    df = pd.DataFrame(rows)
    alt_type = detect_header_and_type(header) if header else None
    # Fix: Only map 'Last' to 'Bid' if 'Bid' is not present in the header
    if alt_type == "option_chain_custom":
        alt_map = ALT_HEADER_MAP[alt_type].copy()
        if header and "Bid" in header:
            alt_map = {k: v for k, v in alt_map.items() if not (k == "Last" and "Bid" in header)}
        df.rename(columns=alt_map, inplace=True)
    elif alt_type and alt_type in ALT_HEADER_MAP:
        df.rename(columns=ALT_HEADER_MAP[alt_type], inplace=True)
    df.rename(columns=SCHEMAS[schema_key], inplace=True, errors='ignore')
    # Ensure all canonical columns are present
    for col in canonical_columns(schema_key):
        if col not in df.columns:
            df[col] = None
    df = df[canonical_columns(schema_key)]
    # Pandera validation (optional)
    if PANDERA_AVAILABLE:
        try:
            # Example: build a simple schema on the fly
            import pandera as pa_validate
            schema = pa_validate.DataFrameSchema({col: pa_validate.Column(pa.Object) for col in df.columns})
            schema.validate(df, lazy=True)
        except Exception as e:
            LOG.warning("Pandera validation failed: %s", e)
    return df

def write_partitioned_arrow(df: pd.DataFrame, dest_dir: Path, schema_key: str, partition_cols=None):
    if partition_cols is None:
        partition_cols = []
    table = pa.Table.from_pandas(df, preserve_index=False, schema=ARROW_SCHEMAS[schema_key])
    hashval = hashlib.md5(df.to_string().encode()).hexdigest()[:8]
    out_dir = dest_dir / datetime.utcnow().strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_dir / f"part_{hashval}.parquet", compression="zstd")

def process_one(args):
    csv_path, dest_root = args
    src = infer_source(csv_path.name)
    if src is None:
        LOG.warning("Skip %s: unknown pattern", csv_path.name)
        return
    try:
        df = robust_read_csv(csv_path, src)
        if df.empty:
            LOG.warning("%s: no valid data, skipping.", csv_path.name)
            return
        write_partitioned_arrow(df, dest_root / src, src)
        shutil.move(csv_path, csv_path.with_suffix(".done"))
        LOG.info("%s → %s/", csv_path.name, src)
    except Exception as e:
        LOG.error("Failed to ingest %s: %s", csv_path.name, e)

def ingest(source_dir: Path, dest_root: Path, log_level="INFO") -> None:
    LOG.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    csv_files = sorted(source_dir.glob("*.csv"))
    if not csv_files:
        LOG.info("No CSV files found; nothing to ingest.")
        return
    with ProcessPoolExecutor() as executor:
        for _ in executor.map(process_one, [(csv_path, dest_root) for csv_path in csv_files]):
            pass

def infer_source(filename: str) -> str | None:
    fn = filename.lower()
    if "watchlist" in fn:
        return "tos_watchlist"
    if "option_chain" in fn or "stockandoptionquote" in fn:
        return "tos_option_chain"
    if "risk" in fn:
        return "tft_risk"
    return None

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", default="data/raw")
    p.add_argument("--dest-root", default="data/feature_store")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    ingest(Path(args.source_dir), Path(args.dest_root), log_level=args.log_level)
