"""
Aggregates news sentiment for each ticker and writes mu_news to Parquet for downstream risk/model use.
Input: expects a CSV or DataFrame with at least columns: ticker, timestamp, sentiment_score, decay_weight
Output: Parquet with columns: ticker, timestamp, mu_news
"""
import pandas as pd
from pathlib import Path
import sys
import subprocess

def find_parquet_for_csv(csv_fp: Path, feature_store_root: Path) -> Path:
    fn = csv_fp.name.lower()
    if "news" in fn:
        subdir = "news"  # adjust as needed for your feature store
    else:
        subdir = "tft_risk"  # fallback or adjust as needed
    fs_dir = feature_store_root / subdir
    candidates = sorted(fs_dir.glob("*/part_*.parquet"), reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No Parquet files found in {fs_dir}")
    return candidates[0]

def main(input_path, output_dir):
    # Use robust ingest for CSVs
    input_path = Path(input_path)
    feature_store_root = Path("data/feature_store")
    if input_path.suffix.lower() == ".csv":
        subprocess.run([
            sys.executable, "scripts/ingest.py",
            "--source-dir", str(input_path.parent),
            "--dest-root", str(feature_store_root)
        ], check=True)
        pq_fp = find_parquet_for_csv(input_path, feature_store_root)
        df = pd.read_parquet(pq_fp)
    else:
        df = pd.read_parquet(input_path)
    # Aggregate mu_news per ticker per timestamp (or per day)
    df['mu_news'] = df['sentiment_score'] * df['decay_weight']
    agg = df.groupby(['ticker', 'timestamp'], as_index=False)['mu_news'].sum()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"news_{pd.Timestamp.now().date()}.parquet"
    agg.to_parquet(out_path, index=False)
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python src/data/news_etl.py <input_csv> <output_dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
