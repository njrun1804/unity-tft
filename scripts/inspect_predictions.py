#!/usr/bin/env python3
"""
Inspect the predictions parquet file to understand the data structure
"""
import pandas as pd
from pathlib import Path
import sys

def inspect_predictions():
    # Find the most recent predictions file
    outputs_dir = Path("outputs")
    prediction_files = list(outputs_dir.glob("predictions_*.parquet"))
    
    if not prediction_files:
        print("No prediction files found in outputs/")
        return
    
    # Get the most recent file
    latest_file = max(prediction_files, key=lambda x: x.stat().st_mtime)
    print(f"Inspecting: {latest_file}")
    
    # Load and inspect
    df = pd.read_parquet(latest_file)
    
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nData types:")
    print(df.dtypes)
    
    print("\nPrediction column stats:")
    if 'prediction' in df.columns:
        print(f"  Min: {df['prediction'].min():.4f}")
        print(f"  Max: {df['prediction'].max():.4f}")
        print(f"  Mean: {df['prediction'].mean():.4f}")
        print(f"  Std: {df['prediction'].std():.4f}")
    
    print("\nConfidence column stats:")
    if 'confidence' in df.columns:
        print(f"  Min: {df['confidence'].min():.4f}")
        print(f"  Max: {df['confidence'].max():.4f}")
        print(f"  Mean: {df['confidence'].mean():.4f}")
    
    print("\nSample rows for position recommendations:")
    sample_cols = ['ticker', 'strike', 'prediction', 'confidence', 'q0.5', 'delta', 'gamma'] 
    available_cols = [col for col in sample_cols if col in df.columns]
    print(df[available_cols].head(10))

if __name__ == "__main__":
    inspect_predictions()
