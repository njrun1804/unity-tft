"""
Automated midday pipeline: Ingest TOS CSVs, update feature store, export LLM-ready JSON, and print ChatGPT prompt.
Run: python src/automate_midday.py --watchlist ~/Downloads/tos_watchlist.csv --options ~/Downloads/tos_option_chain.csv --out data/factors/u_midday.json
"""
import subprocess, sys, argparse, json
from pathlib import Path

def run_ingest():
    # Ingest all new CSVs in data/raw/ to feature store
    print("[INFO] Running robust pipeline ingest...")
    subprocess.run([
        sys.executable, "scripts/ingest.py",
        "--source-dir", "data/raw",
        "--dest-root", "data/feature_store"
    ], check=True)

def run_midday_ingest(watchlist_csv, options_csv, out, feature_store_root=Path("data/feature_store")):
    # Find the latest Parquet files for each CSV
    from src.midday_ingest import find_parquet_for_csv
    wl_parquet = find_parquet_for_csv(watchlist_csv, feature_store_root)
    opt_parquet = find_parquet_for_csv(options_csv, feature_store_root)
    print(f"[INFO] Creating LLM-ready JSON from Parquet: {wl_parquet}, {opt_parquet}")
    subprocess.run([
        sys.executable, "src/midday_ingest.py",
        "--watchlist", str(wl_parquet),
        "--options", str(opt_parquet),
        "--out", str(out)
    ], check=True)

def print_chatgpt_prompt(json_path):
    blob = json.loads(Path(json_path).read_text())
    print("\n\n--- COPY BELOW INTO CHATGPT ---\n")
    print("You are my equity-options strategist.\n\nNumerical snapshot:\n``json\n" + json.dumps(blob, indent=2) + "\n```\n")
    print("Tasks (max ≈200 words):\n\t1. Interpret these factors—bullish, bearish, or neutral?\n\t2. Check fresh public news for Unity Software (U) since today’s open.\n\t3. Recommend ONE of:\n• (a) start / continue a Wheel\n• (b) buy shares outright\n• (c) hold (no trade)\n• (d) higher-order spread (name it + strikes)\n\t4. Give the key driver(s) in 1–2 sentences.\n\nReturn exactly this JSON:\n\n{\n  \"action\": \"wheel | long | hold | spread\",\n  \"rationale\": \"…two sentences…\"\n}\n\nKeep the answer deterministic; cite credible sources for any news.\n---\n")

def find_latest_csv(csv_dir: Path, pattern: str) -> Path | None:
    """Find the most recent CSV in csv_dir whose name ends with the given pattern (case-insensitive)."""
    files = sorted(csv_dir.glob("*.csv"), reverse=True)
    for f in files:
        if f.name.lower().endswith(pattern.lower()):
            return f
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", type=Path, default=Path("csv"), help="Directory with TOS CSVs")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    # Find latest watchlist and option chain CSVs by file ending
    wl_csv = find_latest_csv(args.csv_dir, "watchlist.csv")
    opt_csv = find_latest_csv(args.csv_dir, "StockAndOptionQuoteForU.csv")
    if not wl_csv or not opt_csv:
        print(f"[ERROR] Could not find both watchlist and option chain CSVs in {args.csv_dir}")
        sys.exit(1)
    print(f"[INFO] Using watchlist: {wl_csv}\n[INFO] Using option chain: {opt_csv}")

    run_ingest()  # still runs robust ingest for all csv/raw if needed
    run_midday_ingest(wl_csv, opt_csv, args.out)
    print_chatgpt_prompt(args.out)

    # Remove processed CSVs if everything succeeded
    try:
        wl_csv.unlink()
        opt_csv.unlink()
        print(f"[INFO] Deleted processed files: {wl_csv}, {opt_csv}")
    except Exception as e:
        print(f"[WARN] Could not delete one or more files: {e}")

if __name__ == "__main__":
    main()
