#!/usr/bin/env python3
"""
Generate realistic test data for Unity options to test position recommender logic.
This mimics what we would get from Polygon API during market hours.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json

# Unity's approximate current price
UNITY_STOCK_PRICE = 20.75
RISK_FREE_RATE = 0.05

# Options parameters
EXPIRATIONS = [
    "2025-05-30",  # 6 days to expiry
    "2025-06-06",  # 13 days to expiry  
    "2025-06-20",  # 27 days to expiry
    "2025-07-18",  # 55 days to expiry
]

# Strike ladder around current price (typical for liquid options)
STRIKES = np.arange(15.0, 26.0, 0.5)  # $15-$26 in $0.50 increments

def calculate_black_scholes_greeks(S, K, T, r, sigma, option_type='C'):
    """Calculate Black-Scholes Greeks."""
    from scipy.stats import norm
    import math
    
    if T <= 0:
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'price': max(S-K, 0) if option_type == 'C' else max(K-S, 0)}
    
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    
    if option_type == 'C':  # Call
        delta = norm.cdf(d1)
        price = S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
    else:  # Put
        delta = -norm.cdf(-d1)
        price = K*math.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    
    gamma = norm.pdf(d1) / (S*sigma*math.sqrt(T))
    theta = (-S*norm.pdf(d1)*sigma/(2*math.sqrt(T)) - 
             r*K*math.exp(-r*T)*norm.cdf(d2 if option_type == 'C' else -d2))
    if option_type == 'P':
        theta += r*K*math.exp(-r*T)
    theta /= 365  # Convert to daily
    
    vega = S*norm.pdf(d1)*math.sqrt(T) / 100  # Convert to percentage
    
    return {
        'delta': delta,
        'gamma': gamma, 
        'theta': theta,
        'vega': vega,
        'price': price
    }

def generate_realistic_options_data():
    """Generate realistic Unity options data with proper Greeks and pricing."""
    options_data = []
    
    for expiry in EXPIRATIONS:
        exp_date = datetime.strptime(expiry, "%Y-%m-%d")
        days_to_expiry = (exp_date - datetime.now()).days
        T = days_to_expiry / 365.0  # Time to expiration in years
        
        for strike in STRIKES:
            # Generate realistic IV based on moneyness
            moneyness = strike / UNITY_STOCK_PRICE
            
            # IV smile: higher IV for OTM options
            if moneyness < 0.95:  # ITM calls / OTM puts
                iv_call = 0.35 + 0.15 * (0.95 - moneyness)
                iv_put = 0.45 + 0.20 * (0.95 - moneyness)
            elif moneyness > 1.05:  # OTM calls / ITM puts  
                iv_call = 0.35 + 0.25 * (moneyness - 1.05)
                iv_put = 0.40 + 0.15 * (moneyness - 1.05)
            else:  # ATM
                iv_call = 0.35 + np.random.normal(0, 0.05)
                iv_put = 0.40 + np.random.normal(0, 0.05)
            
            # Ensure realistic IV bounds
            iv_call = np.clip(iv_call, 0.20, 1.50)
            iv_put = np.clip(iv_put, 0.25, 1.50)
            
            # Calculate Greeks and theoretical prices
            call_data = calculate_black_scholes_greeks(UNITY_STOCK_PRICE, strike, T, RISK_FREE_RATE, iv_call, 'C')
            put_data = calculate_black_scholes_greeks(UNITY_STOCK_PRICE, strike, T, RISK_FREE_RATE, iv_put, 'P')
            
            # Generate realistic bid/ask spreads
            # Liquid ATM options: tighter spreads
            # OTM options: wider spreads
            distance_from_money = abs(strike - UNITY_STOCK_PRICE) / UNITY_STOCK_PRICE
            
            if distance_from_money < 0.05:  # Very close to money
                spread_pct = 0.02
                min_spread = 0.05
            elif distance_from_money < 0.10:  # Near money
                spread_pct = 0.04  
                min_spread = 0.05
            elif distance_from_money < 0.20:  # Moderately OTM
                spread_pct = 0.08
                min_spread = 0.05
            else:  # Far OTM
                spread_pct = 0.15
                min_spread = 0.05
            
            # Call option
            call_mid = max(call_data['price'], 0.05)  # Minimum $0.05
            call_spread = max(call_mid * spread_pct, min_spread)
            call_bid = max(call_mid - call_spread/2, 0.01)
            call_ask = call_mid + call_spread/2
            
            # Simulate some volume/open interest based on liquidity
            if distance_from_money < 0.10:
                call_oi = np.random.randint(50, 500)
                call_volume = np.random.randint(5, 50)
            elif distance_from_money < 0.20:
                call_oi = np.random.randint(10, 150)
                call_volume = np.random.randint(0, 20)
            else:
                call_oi = np.random.randint(0, 50)
                call_volume = np.random.randint(0, 5)
            
            options_data.append({
                'ticker': 'U',
                'timestamp': datetime.now(),
                'expiry': expiry,
                'strike': strike,
                'cp': 'C',  # Call
                'bid': round(call_bid, 2),
                'ask': round(call_ask, 2),
                'iv': round(iv_call, 3),
                'delta': round(call_data['delta'], 4),
                'gamma': round(call_data['gamma'], 6),
                'theta': round(call_data['theta'], 4),
                'vega': round(call_data['vega'], 4),
                'oi': call_oi,
                'price': UNITY_STOCK_PRICE
            })
            
            # Put option
            put_mid = max(put_data['price'], 0.05)
            put_spread = max(put_mid * spread_pct, min_spread)
            put_bid = max(put_mid - put_spread/2, 0.01)
            put_ask = put_mid + put_spread/2
            
            # Put volume/OI (usually lower than calls for Unity)
            put_oi = int(call_oi * 0.7)
            put_volume = int(call_volume * 0.6)
            
            options_data.append({
                'ticker': 'U',
                'timestamp': datetime.now(),
                'expiry': expiry,
                'strike': strike,
                'cp': 'P',  # Put
                'bid': round(put_bid, 2),
                'ask': round(put_ask, 2), 
                'iv': round(iv_put, 3),
                'delta': round(put_data['delta'], 4),
                'gamma': round(put_data['gamma'], 6),
                'theta': round(put_data['theta'], 4),
                'vega': round(put_data['vega'], 4),
                'oi': put_oi,
                'price': UNITY_STOCK_PRICE
            })
    
    return pd.DataFrame(options_data)

def generate_stock_data():
    """Generate Unity stock data."""
    return pd.DataFrame([{
        'ticker': 'U',
        'timestamp': datetime.now(),
        'price': UNITY_STOCK_PRICE,
        'open': 20.50,
        'high': 21.00,
        'low': 20.25,
        'close': UNITY_STOCK_PRICE,
        'volume': 9_609_761,
        'prev_close': 20.50,
        'daily_return': 0.012,  # +1.2%
        'net_change': 0.25,
        'mark_pct': 1.22,
        'day_range': 0.75,
        'gap': 0.0,
        'close_position': 'neutral'
    }])

def save_test_data():
    """Save test data to feature store locations."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Generate data
    options_df = generate_realistic_options_data()
    stock_df = generate_stock_data()
    
    print(f"Generated {len(options_df)} options and {len(stock_df)} stock records")
    print(f"Options strikes: {sorted(options_df['strike'].unique())}")
    print(f"Delta range: {options_df['delta'].min():.3f} to {options_df['delta'].max():.3f}")
    print(f"Non-zero deltas: {(options_df['delta'].abs() > 0.001).sum()}/{len(options_df)}")
    print(f"Bid/Ask ranges: ${options_df['bid'].min():.2f}-${options_df['bid'].max():.2f} / ${options_df['ask'].min():.2f}-${options_df['ask'].max():.2f}")
    
    # Save to feature store directories
    options_dir = Path("data/feature_store/polygon_option_chain")
    stock_dir = Path("data/feature_store/polygon_watchlist")
    
    options_dir.mkdir(parents=True, exist_ok=True)
    stock_dir.mkdir(parents=True, exist_ok=True)
    
    # Save with current timestamp
    options_path = options_dir / f"test_data_{timestamp}.parquet"
    stock_path = stock_dir / f"test_data_{timestamp}.parquet" 
    
    options_df.to_parquet(options_path, index=False)
    stock_df.to_parquet(stock_path, index=False)
    
    print(f"\nSaved test data:")
    print(f"  Options: {options_path}")
    print(f"  Stock: {stock_path}")
    
    # Show sample of meaningful options
    meaningful = options_df[
        (options_df['delta'].abs() > 0.01) & 
        (options_df['bid'] > 0.01)
    ].copy()
    
    print(f"\nMeaningful options ({len(meaningful)} total):")
    print("Top 10 by delta:")
    top_delta = meaningful.nlargest(10, 'delta')[['strike', 'cp', 'delta', 'bid', 'ask', 'iv', 'oi']]
    print(top_delta.to_string(index=False))
    
    return options_path, stock_path

if __name__ == "__main__":
    print("Generating realistic Unity options test data...")
    print(f"Unity stock price: ${UNITY_STOCK_PRICE}")
    print(f"Risk-free rate: {RISK_FREE_RATE:.1%}")
    print("="*60)
    
    try:
        save_test_data()
        print("\n✅ Test data generation completed successfully!")
        print("\nNow run the pipeline with: python scripts/automate_pipeline.py")
    except Exception as e:
        print(f"❌ Error generating test data: {e}")
        import traceback
        traceback.print_exc()
