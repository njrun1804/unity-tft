"""
Quick script to inspect what's in your parquet files
"""
import pandas as pd
from pathlib import Path
import sys

def inspect_parquet_files():
    """Inspect all parquet files in the feature store."""
    
    base_path = Path("data/feature_store")
    
    print("="*60)
    print("PARQUET FILE INSPECTION")
    print("="*60)
    
    # Check each data type
    for subdir in ["polygon_watchlist", "polygon_option_chain", "polygon_minute_bars"]:
        dir_path = base_path / subdir
        print(f"\n📁 Checking {subdir}:")
        
        if not dir_path.exists():
            print(f"   ❌ Directory does not exist: {dir_path}")
            continue
            
        # Find all parquet files
        parquet_files = list(dir_path.rglob("*.parquet"))
        print(f"   Found {len(parquet_files)} parquet files")
        
        if parquet_files:
            # Read the most recent file
            latest_file = sorted(parquet_files)[-1]
            print(f"   Reading: {latest_file}")
            
            try:
                df = pd.read_parquet(latest_file)
                print(f"   Shape: {df.shape}")
                print(f"   Columns: {list(df.columns)}")
                
                # Show unique tickers if column exists
                if 'ticker' in df.columns:
                    unique_tickers = df['ticker'].unique()
                    print(f"   Tickers: {list(unique_tickers)[:20]}")  # First 20
                elif 'symbol' in df.columns:
                    unique_symbols = df['symbol'].unique()
                    print(f"   Symbols: {list(unique_symbols)[:20]}")
                    
                # Show sample data
                print("\n   Sample data:")
                print(df.head(3).to_string())
                
            except Exception as e:
                print(f"   ❌ Error reading file: {e}")
                
    print("\n" + "="*60)

def check_specific_ticker(ticker="U"):
    """Check if a specific ticker exists in the data."""
    print(f"\n🔍 Searching for ticker '{ticker}'...")
    
    base_path = Path("data/feature_store")
    found = False
    
    for subdir in ["polygon_watchlist", "polygon_option_chain"]:
        dir_path = base_path / subdir
        if not dir_path.exists():
            continue
            
        for parquet_file in dir_path.rglob("*.parquet"):
            try:
                df = pd.read_parquet(parquet_file)
                
                # Check different possible column names
                for col in ['ticker', 'symbol', 'underlying_ticker']:
                    if col in df.columns and ticker in df[col].values:
                        print(f"✅ Found '{ticker}' in {parquet_file} (column: {col})")
                        found = True
                        # Show sample rows for this ticker
                        sample = df[df[col] == ticker].head(2)
                        print(sample.to_string())
                        break
                        
            except Exception as e:
                pass
                
    if not found:
        print(f"❌ Ticker '{ticker}' not found in any parquet files")
        print("\nPossible issues:")
        print("1. Data fetch didn't include this ticker")
        print("2. Ticker might be under a different symbol")
        print("3. API might not have returned data for this ticker")

if __name__ == "__main__":
    # Check all files
    inspect_parquet_files()
    
    # Check for specific ticker
    check_specific_ticker("U")
    
    # Also check what tickers we tried to fetch
    config_path = Path("positions.json")
    if config_path.exists():
        import json
        with open(config_path) as f:
            config = json.load(f)
            print(f"\n📋 Configured tickers: {config.get('tickers', ['U'])}")
