import os
import argparse
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from polygon import RESTClient
from polygon.exceptions import BadResponse

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
# Temporary debug print removed as entitlement is now confirmed working
assert POLYGON_API_KEY, "Set POLYGON_API_KEY in your environment."

client = RESTClient()  # Uses POLYGON_API_KEY from env
TODAY = datetime.now().strftime("%Y-%m-%d")
TS_UTC = datetime.now(timezone.utc).isoformat()

WATCHLIST_SCHEMA = [
    ("ticker", pa.string()),
    ("price", pa.float64()),
    ("net_change", pa.float64()),
    ("mark_pct", pa.float64()),
    ("timestamp", pa.timestamp("ms")),
]
OPTION_CHAIN_SCHEMA = [
    ("ticker", pa.string()),
    ("timestamp", pa.timestamp("ms")),
    ("expiry", pa.string()),
    ("strike", pa.float64()),
    ("cp", pa.string()),
    ("bid", pa.float64()),
    ("ask", pa.float64()),
    ("iv", pa.float64()),
    ("delta", pa.float64()),
    ("price", pa.float64()),
]

# --- SDK-based watchlist fetch ---
def fetch_watchlist(tickers):
    rows = []
    for ticker in tickers:
        try:
            s = client.get_snapshot_ticker(market_type="stocks", ticker=ticker)
        except BadResponse as e:
            print(f"[warn] skipping stock snapshot for {ticker}: {e}")
            continue
        rows.append({
            "ticker": s.ticker,
            "price": s.last_trade.p if s.last_trade else None,
            "net_change": s.todays_change,
            "mark_pct": s.todays_change_percent,
            # Use current UTC time as pandas.Timestamp
            "timestamp": pd.Timestamp(TS_UTC),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.floor("ms")
    return df

# --- SDK-based option chain fetch ---
def fetch_option_chain(ticker):
    rows = []
    for o in client.list_snapshot_options_chain(ticker):
        rows.append({
            "ticker": ticker,
            # Use current UTC time as pandas.Timestamp
            "timestamp": pd.Timestamp(TS_UTC),
            "expiry": o.details.expiration_date if o.details else None,
            "strike": o.details.strike_price if o.details else None,
            "cp": o.details.contract_type if o.details else None,
            "bid": getattr(o.day, "bid", None),
            "ask": getattr(o.day, "ask", None),
            "iv": o.implied_volatility,
            "delta": getattr(o.greeks, "delta", None),
            "price": getattr(o.last_quote, "p", None),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.floor("ms")
    return df

def write_parquet(df, dest_dir, schema, prefix):
    if df.empty:
        print(f"[warn] {prefix}: no rows, nothing written.")
        return
    table = pa.Table.from_pandas(df, preserve_index=False, schema=pa.schema(schema))
    out_dir = Path(dest_dir) / f"polygon_{prefix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"part_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.parquet"
    pq.write_table(table, fname, compression="zstd")
    print(f"Wrote {fname}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--dest-root", required=True)
    parser.add_argument("--option-chain", action="store_true")
    args = parser.parse_args()

    if args.option_chain:
        for ticker in args.tickers:
            df = fetch_option_chain(ticker)
            write_parquet(df, args.dest_root, OPTION_CHAIN_SCHEMA, "option_chain")
    else:
        df = fetch_watchlist(args.tickers)
        write_parquet(df, args.dest_root, WATCHLIST_SCHEMA, "watchlist")

if __name__ == "__main__":
    main()
