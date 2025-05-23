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

# --- LOGGING PATCH: must be at the very top ---
parser = argparse.ArgumentParser()
parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
args, unknown = parser.parse_known_args()
logging.basicConfig(
    level=getattr(logging, args.log_level),
    format="%(levelname)s %(message)s"
)

LOG = logging.getLogger("ingest")

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
        "LAST": "Bid",
        "Bid": "Bid",
        "BID": "Bid",
        "Ask": "Ask",
        "ASK": "Ask",
        "Volume": "Volume",
        "Open Int": "OpenInterest",
        "IV": "ImpliedVolatility",
        "Delta": "Delta",
        "Gamma": "Gamma",
        "Theta": "Theta",
        "Vega": "Vega",
        "Rho": "Rho",
        "Net Chng": "Net Chg",
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
                        logging.debug(f"{path.name}: Detected header row: {header}")
                        break
                if header is None:
                    logging.warning(f"{path.name}: No header found in expiry chunk. Required: {expected}")
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
                    # Defensive: handle None, strip whitespace, and clean numerics
                    clean = {k: (None if v is None or v.strip() in ('', '<empty>') else v.strip().replace(',', '').rstrip('%')) for k, v in rec.items()}
                    # Drop subtotal/garbage lines: require Strike, Bid, Ask to be present
                    required_cols = ['Strike', 'Bid', 'Ask']
                    if any((col not in clean or clean[col] is None or clean[col] == '') for col in required_cols):
                        continue
                    # Parse QuoteTime to datetime and store as 'timestamp'
                    if 'QuoteTime' in clean and clean['QuoteTime']:
                        try:
                            clean['timestamp'] = pd.to_datetime(clean['QuoteTime'], errors='coerce', utc=True)
                        except Exception:
                            clean['timestamp'] = None
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
                    logging.debug(f"{path.name}: Detected header row: {header}")
                    break
            if header is None:
                logging.warning(f"{path.name}: header not found. Required: {expected}")
                return pd.DataFrame(columns=canonical_columns(schema_key))
            rdr = csv.DictReader(lines, fieldnames=header)
            for rec in rdr:
                if rec[header[0]] == header[0]:
                    break
            for rec in rdr:
                if len(rec) != len(header):
                    continue
                clean = {k: (None if v is None or v.strip() in ('', '<empty>') else v.strip().replace(',', '').rstrip('%')) for k, v in rec.items()}
                # Only apply required_cols filtering for option chain, not watchlist
                if schema_key == "tos_option_chain":
                    # Drop subtotal/garbage lines: require Strike, Bid, Ask to be present
                    required_cols = ['Strike', 'Bid', 'Ask']
                    if any((col not in clean or clean[col] is None or clean[col] == '') for col in required_cols):
                        continue
                # Parse QuoteTime to datetime and store as 'timestamp'
                if 'QuoteTime' in clean and clean['QuoteTime']:
                    try:
                        clean['timestamp'] = pd.to_datetime(clean['QuoteTime'], errors='coerce', utc=True)
                    except Exception:
                        clean['timestamp'] = None
                # Only set expiry for option chain, not watchlist
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
    # --- CANONICAL RENAME MAP ---
    RENAME = {
        "ImpliedVolatility": "iv",
        "OpenInterest": "oi",
        # Add any other renames as needed
    }
    df.rename(columns=RENAME, inplace=True)
    df.rename(columns=SCHEMAS[schema_key], inplace=True, errors='ignore')
    # Ensure all canonical columns are present
    for col in canonical_columns(schema_key):
        if col not in df.columns:
            df[col] = None
    # Remove duplicate columns (e.g., expiry)
    df = df.loc[:, ~df.columns.duplicated()]
    df = df[canonical_columns(schema_key)]
    # Convert numerics for watchlist and option chain
    if schema_key in ("tos_watchlist", "tos_option_chain"):
        for col in ["price", "net_change", "mark_pct", "bid", "ask", "iv", "delta", "gamma", "theta", "vega", "oi", "vol", "strike"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        # --- PATCH: fill missing Greeks with 0.0 ---
        for greek in ["gamma", "vega"]:
            if greek in df.columns:
                df[greek] = df[greek].fillna(0.0)
    # Ensure every option row has `price`
    if schema_key == "tos_option_chain":
        import re, numpy as np
        raw_text = path.read_text(encoding="utf-8", errors="replace")
        if "price" not in df.columns:
            # 1️⃣ Try to pull the underlying LAST from the header line
            m = re.search(r"quote.*?for\s+\w+\s+on.*?[-\s](\d+\.\d+)", raw_text.splitlines()[0])
            underlying_px = float(m.group(1)) if m else np.nan
            # 2️⃣ Mid-price fallback when Bid/Ask present
            bid = pd.to_numeric(df.get("bid"), errors="coerce")
            ask = pd.to_numeric(df.get("ask"), errors="coerce")
            mid = (bid + ask) / 2
            df["price"] = np.where(mid.notna(), mid, underlying_px)
    # After DataFrame creation, ensure 'timestamp' is datetime64[ns, UTC] if present
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
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
        # Guarantee ticker column exists – infer from filename if absent
        from src.utils.preprocess import preprocess_df
        df = preprocess_df(df, source_path=csv_path)
        # --- Wheel metrics for options chain ---
        if src == "tos_option_chain":
            import numpy as np
            df["DTE"] = (pd.to_datetime(df["expiry"]) - pd.to_datetime(df["timestamp"]))
            df["DTE"] = df["DTE"].dt.days
            df["premium_yield"] = df["bid"] / df["strike"] / df["DTE"] * 365
            df["capital_at_risk"] = df["strike"] * 100
            df["wheel_sharpe"] = df["premium_yield"] / (df["iv"] * np.sqrt(252))
        write_partitioned_arrow(df, dest_root / src, src)
        shutil.move(csv_path, csv_path.with_suffix(".done"))
        LOG.info("%s → %s/", csv_path.name, src)
    except Exception as e:
        LOG.error("Failed to ingest %s: %s", csv_path.name, e)

def find_csv_files(source_dir: Path) -> list[Path]:
    """Return all CSV files in the source directory."""
    return sorted(source_dir.glob("*.csv"))

def cleanup_processed_files(source_dir: Path):
    """Remove all .done files in the source directory."""
    for f in source_dir.glob("*.done"):
        try:
            f.unlink()
            LOG.info("Deleted processed file: %s", f)
        except Exception as e:
            LOG.warning("Could not delete %s: %s", f, e)

def ingest(source_dir: Path, dest_root: Path, log_level="INFO") -> None:
    """Ingest all CSVs in source_dir to Parquet feature store at dest_root."""
    LOG.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    csv_files = find_csv_files(source_dir)
    if not csv_files:
        LOG.info("No CSV files found; nothing to ingest.")
        return
    with ProcessPoolExecutor() as executor:
        for _ in executor.map(process_one, [(csv_path, dest_root) for csv_path in csv_files]):
            pass
    cleanup_processed_files(source_dir)

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
    p = argparse.ArgumentParser(description="Robust TOS CSV → Parquet ingester.")
    p.add_argument("--source-dir", default="data/raw", help="Directory with raw TOS CSVs")
    p.add_argument("--dest-root", default="data/feature_store", help="Output Parquet feature store root")
    p.add_argument("--log-level", default="INFO", help="Logging level")
    args = p.parse_args()
    ingest(Path(args.source_dir), Path(args.dest_root), log_level=args.log_level)

# 2. mu_news in news/risk pipeline (example, to be placed in your news ETL aggregation step):
# agg["mu_news"] = agg["sentiment_score"] * agg["decay_weight"]
# Make sure this is saved to Parquet in your news/risk ETL.
