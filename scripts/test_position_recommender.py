#!/usr/bin/env python3
"""
Test the fixed position recommender with real predictions data
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.utils.position_recommender import recommend_positions, load_positions_config

def test_position_recommender():
    # Load the latest predictions
    outputs_dir = Path("outputs")
    prediction_files = list(outputs_dir.glob("predictions_*.parquet"))
    
    if not prediction_files:
        print("No prediction files found")
        return
    
    latest_file = max(prediction_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading predictions from: {latest_file}")
    
    df = pd.read_parquet(latest_file)
    print(f"Loaded {len(df)} predictions")
    
    # Test with scalar confidence (average)
    avg_confidence = df['confidence'].mean()
    print(f"Average confidence: {avg_confidence:.3f}")
    
    try:
        # Load config
        config = load_positions_config()
        print("Config loaded successfully")
        
        # Test recommendations
        recommendations = recommend_positions(df.head(10), avg_confidence, config)
        print(f"Generated {len(recommendations)} recommendations")
        
        # Print sample recommendations
        for i, (pos_id, rec) in enumerate(list(recommendations.items())[:5]):
            print(f"  {pos_id}: {rec['action']} size={rec['size']}, conf={rec['certainty']:.2f}, delta={rec['delta']:.3f}")
            
        print("✓ Position recommender test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Position recommender test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_position_recommender()
