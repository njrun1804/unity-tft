"""
Black-Scholes Greeks Calculator for Unity Price Trading System
Calculates missing option Greeks when Polygon API doesn't provide them
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GreeksCalculator:
    """
    Calculate Black-Scholes Greeks for options when not provided by data source.
    Handles the 9 required features for LSTM model.
    """
    
    def __init__(self, risk_free_rate: float = 0.05):
        """
        Initialize Greeks calculator.
        
        Args:
            risk_free_rate: Risk-free interest rate (default 5% for current Fed funds)
        """
        self.risk_free_rate = risk_free_rate
        
    def calculate_time_to_expiry(self, expiry_date, current_date=None):
        """Calculate time to expiry in years."""
        if current_date is None:
            current_date = datetime.now()
            
        if isinstance(expiry_date, str):
            expiry_date = pd.to_datetime(expiry_date)
        if isinstance(current_date, str):
            current_date = pd.to_datetime(current_date)
            
        days_to_expiry = (expiry_date - current_date).days
        # Ensure minimum 1 day to avoid division by zero
        days_to_expiry = max(1, days_to_expiry)
        
        return days_to_expiry / 365.0
    
    def calculate_d1_d2(self, S, K, T, r, sigma):
        """Calculate d1 and d2 for Black-Scholes formula."""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2
    
    def calculate_all_greeks(self, S, K, T, r, sigma, option_type='CALL'):
        """
        Calculate all Greeks for a single option.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiry in years
            r: Risk-free rate
            sigma: Implied volatility
            option_type: 'CALL' or 'PUT'
            
        Returns:
            dict with delta, gamma, theta, vega, rho
        """
        # Ensure positive values
        S = max(S, 0.01)
        K = max(K, 0.01)
        T = max(T, 1/365)  # Minimum 1 day
        sigma = max(sigma, 0.01)  # Minimum 1% volatility
        
        d1, d2 = self.calculate_d1_d2(S, K, T, r, sigma)
        
        # Common calculations
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        n_d1 = norm.pdf(d1)
        
        # Gamma (same for calls and puts)
        gamma = n_d1 / (S * sigma * np.sqrt(T))
        
        # Vega (same for calls and puts, divide by 100 for 1% move)
        vega = S * n_d1 * np.sqrt(T) / 100
        
        if option_type.upper() == 'CALL':
            # Delta for call
            delta = N_d1
            
            # Theta for call (negative, per day)
            theta = (-S * n_d1 * sigma / (2 * np.sqrt(T)) 
                    - r * K * np.exp(-r * T) * N_d2) / 365
            
            # Rho for call (per 1% move)
            rho = K * T * np.exp(-r * T) * N_d2 / 100
            
        else:  # PUT
            # Delta for put
            delta = N_d1 - 1
            
            # Theta for put (negative, per day)
            theta = (-S * n_d1 * sigma / (2 * np.sqrt(T)) 
                    + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
            
            # Rho for put (per 1% move)
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    def process_options_dataframe(self, df, stock_price_col='price', 
                                 strike_col='strike', expiry_col='expiry',
                                 iv_col='iv', option_type_col='cp'):
        """
        Process entire options DataFrame and calculate missing Greeks.
        
        Args:
            df: DataFrame with options data
            stock_price_col: Column name for underlying stock price
            strike_col: Column name for strike price
            expiry_col: Column name for expiry date
            iv_col: Column name for implied volatility
            option_type_col: Column name for option type (CALL/PUT)
            
        Returns:
            DataFrame with calculated Greeks
        """
        df = df.copy()
        
        # Initialize Greek columns if they don't exist
        greek_cols = ['delta', 'gamma', 'theta', 'vega']
        for col in greek_cols:
            if col not in df.columns:
                df[col] = np.nan
        
        # Calculate time to expiry
        df['time_to_expiry'] = df[expiry_col].apply(self.calculate_time_to_expiry)
        
        # Process each row
        calculated_count = 0
        for idx, row in df.iterrows():
            # Skip if all Greeks already exist
            if all(pd.notna(row[col]) and row[col] != 0 for col in greek_cols):
                continue
                
            try:
                # Get required values
                S = float(row[stock_price_col])
                K = float(row[strike_col])
                T = float(row['time_to_expiry'])
                sigma = float(row[iv_col]) if pd.notna(row[iv_col]) else 0.3  # Default 30% IV
                option_type = row[option_type_col]
                
                # Calculate Greeks
                greeks = self.calculate_all_greeks(S, K, T, self.risk_free_rate, sigma, option_type)
                
                # Update DataFrame
                for greek, value in greeks.items():
                    if greek in greek_cols:
                        df.at[idx, greek] = value
                        
                calculated_count += 1
                
            except Exception as e:
                logger.debug(f"Failed to calculate Greeks for row {idx}: {e}")
                # Set default values for failed calculations
                for col in greek_cols:
                    if pd.isna(df.at[idx, col]):
                        df.at[idx, col] = 0.0
        
        logger.info(f"Calculated Greeks for {calculated_count} options")
        return df
    
    def prepare_lstm_features(self, df):
        """
        Prepare the exact 9 features required by LSTM model.
        Based on your feature_list.json: 
        ['strike', 'bid', 'ask', 'iv', 'delta', 'gamma', 'theta', 'vega', 'oi']
        """
        required_features = ['strike', 'bid', 'ask', 'iv', 'delta', 'gamma', 'theta', 'vega', 'oi']
        
        # Ensure all columns exist
        for col in required_features:
            if col not in df.columns:
                if col == 'oi':  # Open Interest might be missing
                    df[col] = 0
                else:
                    logger.warning(f"Missing required feature: {col}")
                    df[col] = 0.0
        
        # Clean up values
        for col in required_features:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
        return df[required_features]


# Example usage and testing
if __name__ == "__main__":
    # Test with sample data
    sample_data = pd.DataFrame({
        'ticker': ['U', 'U', 'U', 'U'],
        'price': [25.50, 25.50, 25.50, 25.50],
        'strike': [25.0, 26.0, 24.0, 25.0],
        'expiry': ['2025-06-20', '2025-06-20', '2025-06-20', '2025-07-18'],
        'cp': ['CALL', 'CALL', 'PUT', 'PUT'],
        'bid': [1.20, 0.65, 0.45, 0.85],
        'ask': [1.25, 0.70, 0.50, 0.90],
        'iv': [0.35, 0.38, 0.33, 0.36],
        'oi': [150, 200, 175, 125]
    })
    
    calculator = GreeksCalculator()
    result = calculator.process_options_dataframe(sample_data)
    
    print("Sample Greeks Calculation:")
    print(result[['ticker', 'strike', 'cp', 'delta', 'gamma', 'theta', 'vega']].to_string())
    
    # Test LSTM feature preparation
    lstm_features = calculator.prepare_lstm_features(result)
    print("\nLSTM Features Shape:", lstm_features.shape)
    print("LSTM Features Columns:", list(lstm_features.columns))
