"""
Greeks Calculator for Options Trading
=====================================

This module implements Black-Scholes option pricing and Greeks calculation
to compensate for missing Greeks data in Polygon API responses.

Author: Unity Trading System
Date: 2024-12-28
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, date

logger = logging.getLogger(__name__)


class GreeksCalculator:
    """
    Calculate Black-Scholes option prices and Greeks for options contracts.
    
    This class handles the calculation of:
    - Option theoretical prices (bid/ask estimation)
    - Delta: Price sensitivity to underlying price changes
    - Gamma: Rate of change of delta
    - Theta: Time decay
    - Vega: Volatility sensitivity
    """
    
    def __init__(self, risk_free_rate: float = 0.05):
        """
        Initialize the Greeks calculator.
        
        Args:
            risk_free_rate: Risk-free interest rate (default 5%)
        """
        self.risk_free_rate = risk_free_rate
        
    def _time_to_expiration(self, expiration_date: str) -> float:
        """
        Calculate time to expiration in years.
        
        Args:
            expiration_date: Expiration date in 'YYYY-MM-DD' format
            
        Returns:
            Time to expiration in years
        """
        try:
            if isinstance(expiration_date, str):
                exp_date = datetime.strptime(expiration_date, '%Y-%m-%d').date()
            else:
                exp_date = expiration_date
                
            today = date.today()
            days_to_exp = (exp_date - today).days
            
            # Minimum 1 day to avoid division by zero
            days_to_exp = max(1, days_to_exp)
            
            return days_to_exp / 365.0
        except Exception as e:
            logger.warning(f"Error calculating time to expiration for {expiration_date}: {e}")
            return 0.01  # Default to ~4 days
    
    def _d1_d2(self, S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float]:
        """
        Calculate d1 and d2 for Black-Scholes formula.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration
            r: Risk-free rate
            sigma: Implied volatility
            
        Returns:
            Tuple of (d1, d2)
        """
        if T <= 0 or sigma <= 0:
            return 0.0, 0.0
            
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2
    
    def black_scholes_call(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate Black-Scholes call option price.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration
            r: Risk-free rate
            sigma: Implied volatility
            
        Returns:
            Call option theoretical price
        """
        if T <= 0:
            return max(S - K, 0)
            
        d1, d2 = self._d1_d2(S, K, T, r, sigma)
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return max(call_price, 0)
    
    def black_scholes_put(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate Black-Scholes put option price.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration
            r: Risk-free rate
            sigma: Implied volatility
            
        Returns:
            Put option theoretical price
        """
        if T <= 0:
            return max(K - S, 0)
            
        d1, d2 = self._d1_d2(S, K, T, r, sigma)
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return max(put_price, 0)
    
    def calculate_delta(self, S: float, K: float, T: float, r: float, sigma: float, is_call: bool = True) -> float:
        """
        Calculate option delta.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration
            r: Risk-free rate
            sigma: Implied volatility
            is_call: True for call, False for put
            
        Returns:
            Delta value
        """
        if T <= 0:
            if is_call:
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0
                
        d1, _ = self._d1_d2(S, K, T, r, sigma)
        
        if is_call:
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1
    
    def calculate_gamma(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate option gamma (same for calls and puts).
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration
            r: Risk-free rate
            sigma: Implied volatility
            
        Returns:
            Gamma value
        """
        if T <= 0 or sigma <= 0:
            return 0.0
            
        d1, _ = self._d1_d2(S, K, T, r, sigma)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        return gamma
    
    def calculate_theta(self, S: float, K: float, T: float, r: float, sigma: float, is_call: bool = True) -> float:
        """
        Calculate option theta (time decay).
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration
            r: Risk-free rate
            sigma: Implied volatility
            is_call: True for call, False for put
            
        Returns:
            Theta value (per day)
        """
        if T <= 0:
            return 0.0
            
        d1, d2 = self._d1_d2(S, K, T, r, sigma)
        
        term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        
        if is_call:
            term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
            theta = term1 + term2
        else:
            term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
            theta = term1 + term2
            
        return theta / 365  # Convert to per-day theta
    
    def calculate_vega(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate option vega (same for calls and puts).
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration
            r: Risk-free rate
            sigma: Implied volatility
            
        Returns:
            Vega value (per 1% volatility change)
        """
        if T <= 0:
            return 0.0
            
        d1, _ = self._d1_d2(S, K, T, r, sigma)
        vega = S * norm.pdf(d1) * np.sqrt(T)
        return vega / 100  # Convert to per 1% change
    
    def estimate_bid_ask(self, theoretical_price: float, spread_pct: float = 0.05) -> Tuple[float, float]:
        """
        Estimate bid/ask prices from theoretical price when market data is missing.
        
        Args:
            theoretical_price: Black-Scholes theoretical price
            spread_pct: Bid-ask spread as percentage of theoretical price
            
        Returns:
            Tuple of (bid, ask) prices
        """
        spread = theoretical_price * spread_pct
        bid = max(0.01, theoretical_price - spread / 2)
        ask = theoretical_price + spread / 2
        return bid, ask
    
    def process_options_dataframe(self, options_df: pd.DataFrame, stock_price: float, 
                                default_iv: float = 0.30) -> pd.DataFrame:
        """
        Process options DataFrame to calculate Greeks and estimate missing prices.
        
        Args:
            options_df: DataFrame with options contract data
            stock_price: Current underlying stock price
            default_iv: Default implied volatility when missing
            
        Returns:
            DataFrame with calculated Greeks and estimated prices
        """
        if options_df.empty:
            logger.warning("Empty options DataFrame provided")
            return options_df
            
        logger.info(f"Processing {len(options_df)} options contracts with stock price ${stock_price:.2f}")
        
        result_df = options_df.copy()
        
        # Ensure required columns exist
        required_cols = ['strike_price', 'expiration_date', 'contract_type']
        missing_cols = [col for col in required_cols if col not in result_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Initialize new columns
        for col in ['bid', 'ask', 'iv', 'delta', 'gamma', 'theta', 'vega', 'oi']:
            if col not in result_df.columns:
                result_df[col] = np.nan
        
        # Process each option
        for idx, row in result_df.iterrows():
            try:
                K = float(row['strike_price'])
                T = self._time_to_expiration(row['expiration_date'])
                is_call = row['contract_type'].upper() == 'CALL'
                
                # Use provided IV or default
                iv = row.get('iv', default_iv) or default_iv
                if pd.isna(iv) or iv <= 0:
                    iv = default_iv
                
                # Calculate theoretical price
                if is_call:
                    theo_price = self.black_scholes_call(stock_price, K, T, self.risk_free_rate, iv)
                else:
                    theo_price = self.black_scholes_put(stock_price, K, T, self.risk_free_rate, iv)
                
                # Estimate bid/ask if missing
                bid = row.get('bid')
                ask = row.get('ask')
                if pd.isna(bid) or pd.isna(ask) or bid is None or ask is None:
                    bid, ask = self.estimate_bid_ask(theo_price)
                
                # Calculate Greeks
                delta = self.calculate_delta(stock_price, K, T, self.risk_free_rate, iv, is_call)
                gamma = self.calculate_gamma(stock_price, K, T, self.risk_free_rate, iv)
                theta = self.calculate_theta(stock_price, K, T, self.risk_free_rate, iv, is_call)
                vega = self.calculate_vega(stock_price, K, T, self.risk_free_rate, iv)
                
                # Update DataFrame
                result_df.at[idx, 'bid'] = bid
                result_df.at[idx, 'ask'] = ask
                result_df.at[idx, 'iv'] = iv
                result_df.at[idx, 'delta'] = delta
                result_df.at[idx, 'gamma'] = gamma
                result_df.at[idx, 'theta'] = theta
                result_df.at[idx, 'vega'] = vega
                
                # Set default open interest if missing
                if pd.isna(row.get('oi')) or row.get('oi') is None:
                    result_df.at[idx, 'oi'] = 100  # Default open interest
                    
            except Exception as e:
                logger.error(f"Error processing option at index {idx}: {e}")
                # Set default values for failed calculations
                result_df.at[idx, 'bid'] = 0.01
                result_df.at[idx, 'ask'] = 0.02
                result_df.at[idx, 'iv'] = default_iv
                result_df.at[idx, 'delta'] = 0.0
                result_df.at[idx, 'gamma'] = 0.0
                result_df.at[idx, 'theta'] = 0.0
                result_df.at[idx, 'vega'] = 0.0
                result_df.at[idx, 'oi'] = 100
        
        logger.info(f"Successfully processed options with Greeks calculated")
        return result_df
    
    def prepare_lstm_features(self, options_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare exactly 9 features for LSTM model inference.
        
        Expected features: ['strike', 'bid', 'ask', 'iv', 'delta', 'gamma', 'theta', 'vega', 'oi']
        
        Args:
            options_df: DataFrame with options data and calculated Greeks
            
        Returns:
            DataFrame with exactly 9 LSTM features
        """
        if options_df.empty:
            logger.warning("Empty options DataFrame for LSTM features")
            return pd.DataFrame(columns=['strike', 'bid', 'ask', 'iv', 'delta', 'gamma', 'theta', 'vega', 'oi'])
        
        # Map column names to expected LSTM features
        feature_mapping = {
            'strike_price': 'strike',
            'bid': 'bid',
            'ask': 'ask', 
            'iv': 'iv',
            'delta': 'delta',
            'gamma': 'gamma',
            'theta': 'theta',
            'vega': 'vega',
            'oi': 'oi'
        }
        
        lstm_features = pd.DataFrame()
        
        for expected_col, source_col in feature_mapping.items():
            if source_col in options_df.columns:
                lstm_features[expected_col] = options_df[source_col]
            elif expected_col in options_df.columns:
                lstm_features[expected_col] = options_df[expected_col]
            else:
                logger.warning(f"Missing feature column: {expected_col}, using default value")
                # Set reasonable defaults
                if expected_col == 'strike':
                    lstm_features[expected_col] = 20.0  # Near Unity's current price
                elif expected_col in ['bid', 'ask']:
                    lstm_features[expected_col] = 0.5
                elif expected_col == 'iv':
                    lstm_features[expected_col] = 0.30
                elif expected_col == 'oi':
                    lstm_features[expected_col] = 100
                else:
                    lstm_features[expected_col] = 0.0
        
        # Ensure all features are numeric
        for col in lstm_features.columns:
            lstm_features[col] = pd.to_numeric(lstm_features[col], errors='coerce').fillna(0)
        
        # Validate we have exactly 9 features
        expected_features = ['strike', 'bid', 'ask', 'iv', 'delta', 'gamma', 'theta', 'vega', 'oi']
        lstm_features = lstm_features[expected_features]
        
        logger.info(f"Prepared {len(lstm_features)} options with 9 LSTM features")
        logger.debug(f"LSTM features shape: {lstm_features.shape}")
        logger.debug(f"LSTM feature columns: {list(lstm_features.columns)}")
        
        return lstm_features
    
    def calculate_confidence_score(self, options_df: pd.DataFrame, stock_price: float) -> pd.DataFrame:
        """
        Calculate confidence scores based on option moneyness and other factors.
        
        Args:
            options_df: DataFrame with options data
            stock_price: Current stock price
            
        Returns:
            DataFrame with added confidence_score column
        """
        result_df = options_df.copy()
        
        if 'strike_price' not in result_df.columns:
            result_df['confidence_score'] = 0.5
            return result_df
        
        # Calculate moneyness (how close strike is to stock price)
        strikes = pd.to_numeric(result_df['strike_price'], errors='coerce')
        moneyness = np.abs(strikes - stock_price) / stock_price
        
        # Higher confidence for options closer to the money
        # Confidence decreases as options get further from the money
        confidence = np.exp(-5 * moneyness)  # Exponential decay
        confidence = np.clip(confidence, 0.1, 1.0)  # Keep between 0.1 and 1.0
        
        result_df['confidence_score'] = confidence
        
        logger.info(f"Calculated confidence scores: mean={confidence.mean():.3f}, "
                   f"min={confidence.min():.3f}, max={confidence.max():.3f}")
        
        return result_df
