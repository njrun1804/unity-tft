#!/usr/bin/env python
"""
feature_engineer.py - Feature engineering for TFT model and strategy
Calculates technical indicators, Greeks fallbacks, and market microstructure
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Feature engineering for wheel strategy
    Produces features compatible with TFT model and strategy decisions
    """
    
    def __init__(self, lookback_windows: List[int] = [10, 30, 60, 390]):
        self.lookback_windows = lookback_windows
        self.risk_free_rate = 0.05  # Current T-bill rate
        
    def engineer_all_features(self, stock_data: Dict, minute_bars: pd.DataFrame, option_chain: pd.DataFrame) -> Dict:
        """Main entry point - engineer all features from raw data"""
        try:
            # Extract features from each data source
            stock_features = self._extract_stock_features(stock_data)
            technical_features = self._calculate_technical_features(minute_bars)
            option_features = self._calculate_option_features(option_chain, stock_data.get('price', 0))
            microstructure_features = self._calculate_microstructure(minute_bars)
            
            # Combine all features
            all_features = {
                **stock_features,
                **technical_features,
                **option_features,
                **microstructure_features,
                'timestamp': pd.Timestamp.now(),
                'ticker': stock_data.get('ticker', 'UNKNOWN')
            }
            
            # Calculate derived features for strategy
            all_features.update(self._calculate_strategy_features(all_features))
            
            logger.info(f"Engineered {len(all_features)} features for {all_features['ticker']}")
            return all_features
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            return self._default_features(stock_data.get('ticker', 'UNKNOWN'))
    
    def _extract_stock_features(self, stock_data: Dict) -> Dict:
        """Extract features from stock snapshot data"""
        if not stock_data or 'day' not in stock_data:
            return self._default_stock_features()
        
        day = stock_data['day']
        prev_day = stock_data.get('prev_day', {})
        
        # Basic price features
        current_price = stock_data.get('price', day.get('c', 0))
        prev_close = prev_day.get('c', current_price)
        
        features = {
            'price': current_price,
            'open': day.get('o', current_price),
            'high': day.get('h', current_price),
            'low': day.get('l', current_price),
            'close': day.get('c', current_price),
            'volume': day.get('v', 0),
            'prev_close': prev_close,
            
            # Returns and changes
            'daily_return': (current_price - prev_close) / prev_close if prev_close > 0 else 0,
            'net_change': stock_data.get('net_change', 0),
            'mark_pct': stock_data.get('mark_pct', 0) / 100,  # Convert to decimal
            
            # Intraday metrics
            'day_range': (day.get('h', 0) - day.get('l', 0)) / current_price if current_price > 0 else 0,
            'gap': (day.get('o', 0) - prev_close) / prev_close if prev_close > 0 else 0,
            'close_position': ((current_price - day.get('l', 0)) / 
                             (day.get('h', 0) - day.get('l', 0))) if (day.get('h', 0) - day.get('l', 0)) > 0 else 0.5,
        }
        
        return features
    
    def _calculate_technical_features(self, minute_bars: pd.DataFrame) -> Dict:
        """Calculate technical indicators from minute bars using vectorized operations"""
        if minute_bars.empty:
            return self._default_technical_features()
        
        try:
            # Ensure we have enough data
            max_window = max(self.lookback_windows)
            if len(minute_bars) < max_window:
                return self._default_technical_features()
            
            # Pre-calculate base time series once (vectorized)
            close_prices = minute_bars['close']
            volume = minute_bars['volume']
            returns = close_prices.pct_change().dropna()
            
            # Annualization factor for volatility
            vol_factor = np.sqrt(252 * 390)
            
            # Pre-calculate all rolling statistics at once using vectorized operations
            features = {}
            
            # Use pandas vectorized rolling operations for all windows simultaneously
            valid_windows = [w for w in self.lookback_windows if len(minute_bars) >= w]
            
            if not valid_windows:
                return self._default_technical_features()
            
            # Vectorized calculations for returns-based features
            returns_last_idx = len(minute_bars) - 1
            for window in valid_windows:
                start_idx = max(0, returns_last_idx - window + 1)
                window_returns = returns.iloc[start_idx:returns_last_idx + 1]
                window_close = close_prices.iloc[start_idx:returns_last_idx + 1]
                window_volume = volume.iloc[start_idx:returns_last_idx + 1]
                
                if len(window_returns) > 1:
                    # Volatility features (vectorized)
                    features[f'ewma_vol_{window}'] = window_returns.ewm(span=window//2).std().iloc[-1] * vol_factor
                    features[f'realized_vol_{window}'] = window_returns.std() * vol_factor
                    features[f'mean_return_{window}'] = window_returns.mean()
                    
                    # Price momentum (vectorized)
                    features[f'momentum_{window}'] = (window_close.iloc[-1] - window_close.iloc[0]) / window_close.iloc[0]
                    
                    # Volume features (vectorized)
                    features[f'avg_volume_{window}'] = window_volume.mean()
                    
                    # Volume trend using numpy correlation (more efficient)
                    if len(window_volume) > 1:
                        x_vals = np.arange(len(window_volume))
                        corr_matrix = np.corrcoef(x_vals, window_volume.values)
                        features[f'volume_trend_{window}'] = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0.0
                    else:
                        features[f'volume_trend_{window}'] = 0.0
                
                # Technical indicators for windows >= 20
                if window >= 20 and len(window_close) >= 20:
                    # Bollinger Bands (vectorized)
                    sma_20 = window_close.rolling(20).mean()
                    std_20 = window_close.rolling(20).std()
                    if not pd.isna(sma_20.iloc[-1]) and std_20.iloc[-1] > 0:
                        features[f'bb_position_{window}'] = ((window_close.iloc[-1] - sma_20.iloc[-1]) / 
                                                           (2 * std_20.iloc[-1]))
                    else:
                        features[f'bb_position_{window}'] = 0.0
                    
                    # RSI approximation (vectorized)
                    if len(window_returns) >= 14:
                        gains = window_returns.where(window_returns > 0, 0).rolling(14).mean()
                        losses = -window_returns.where(window_returns < 0, 0).rolling(14).mean()
                        rs = gains / losses.replace(0, np.nan)
                        rsi_val = 100 - (100 / (1 + rs.iloc[-1])) if not pd.isna(rs.iloc[-1]) else 50
                        features[f'rsi_{window}'] = rsi_val
                    else:
                        features[f'rsi_{window}'] = 50
            
            return features
            
        except Exception as e:
            logger.error(f"Technical feature calculation failed: {e}")
            return self._default_technical_features()
    
    def _calculate_option_features(self, option_chain: pd.DataFrame, spot: float) -> Dict:
        """Calculate option-specific features"""
        if option_chain.empty or spot <= 0:
            return self._default_option_features()
        
        try:
            # Filter for reasonable strikes (±20% from spot)
            min_strike = spot * 0.8
            max_strike = spot * 1.2
            
            options = option_chain[
                (option_chain['strike'] >= min_strike) & 
                (option_chain['strike'] <= max_strike) &
                (option_chain['dte'] >= 1) &
                (option_chain['dte'] <= 60)  # Focus on 1-60 DTE
            ].copy()
            
            if options.empty:
                return self._default_option_features()
            
            # Separate calls and puts
            calls = options[options['cp'] == 'call']
            puts = options[options['cp'] == 'put']
            
            features = {}
            
            # IV features
            if 'iv' in options.columns and options['iv'].notna().any():
                features['iv_mean'] = options['iv'].mean()
                features['iv_std'] = options['iv'].std()
                features['iv_skew'] = options['iv'].skew()
                
                # IV rank (simplified)
                iv_percentile = options['iv'].rank(pct=True)
                features['iv_rank'] = iv_percentile.mean()
            
            # Put/Call ratios
            if not calls.empty and not puts.empty:
                call_volume = calls['volume'].sum()
                put_volume = puts['volume'].sum()
                features['put_call_volume_ratio'] = put_volume / call_volume if call_volume > 0 else 1.0
                
                call_oi = calls['open_interest'].sum()
                put_oi = puts['open_interest'].sum()
                features['put_call_oi_ratio'] = put_oi / call_oi if call_oi > 0 else 1.0
            
            # Greek exposure (approximate)
            if 'delta' in options.columns and options['delta'].notna().any():
                # Portfolio delta
                total_delta = (options['delta'] * options['open_interest']).sum()
                features['market_delta'] = total_delta
                
                # Gamma exposure
                if 'gamma' in options.columns:
                    total_gamma = (options['gamma'] * options['open_interest']).sum()
                    features['market_gamma'] = total_gamma
            
            # Wheel-specific features
            features.update(self._calculate_wheel_option_features(options, spot))
            
            return features
            
        except Exception as e:
            logger.error(f"Option feature calculation failed: {e}")
            return self._default_option_features()
    
    def _calculate_wheel_option_features(self, options: pd.DataFrame, spot: float) -> Dict:
        """Calculate wheel strategy specific option features"""
        features = {}
        
        try:
            # Find best put to sell (cash-secured put)
            puts = options[options['cp'] == 'put'].copy()
            if not puts.empty:
                # Target 0.15-0.30 delta puts
                puts['delta_abs'] = puts['delta'].abs()
                target_puts = puts[
                    (puts['delta_abs'] >= 0.15) & 
                    (puts['delta_abs'] <= 0.30) &
                    (puts['dte'] >= 15) &
                    (puts['dte'] <= 45)
                ]
                
                if not target_puts.empty:
                    # Best put by premium/strike ratio
                    target_puts['premium_yield'] = ((target_puts['bid'] + target_puts['ask']) / 2) / target_puts['strike']
                    best_put = target_puts.loc[target_puts['premium_yield'].idxmax()]
                    
                    features['best_put_strike'] = best_put['strike']
                    features['best_put_premium'] = (best_put['bid'] + best_put['ask']) / 2
                    features['best_put_delta'] = best_put['delta']
                    features['best_put_dte'] = best_put['dte']
                    features['best_put_yield'] = best_put['premium_yield']
                    features['capital_at_risk'] = best_put['strike'] * 100  # Per contract
            
            # Find best call to sell (covered call) - if we owned stock
            calls = options[options['cp'] == 'call'].copy()
            if not calls.empty:
                # Target slightly OTM calls
                otm_calls = calls[calls['strike'] > spot * 1.02].copy()  # At least 2% OTM
                
                if not otm_calls.empty:
                    otm_calls['premium_yield'] = ((otm_calls['bid'] + otm_calls['ask']) / 2) / spot
                    best_call = otm_calls.loc[otm_calls['premium_yield'].idxmax()]
                    
                    features['best_call_strike'] = best_call['strike']
                    features['best_call_premium'] = (best_call['bid'] + best_call['ask']) / 2
                    features['best_call_delta'] = best_call['delta']
                    features['best_call_dte'] = best_call['dte']
                    features['best_call_yield'] = best_call['premium_yield']
            
        except Exception as e:
            logger.error(f"Wheel option feature calculation failed: {e}")
        
        return features
    
    def _calculate_microstructure(self, minute_bars: pd.DataFrame) -> Dict:
        """Calculate market microstructure features"""
        if minute_bars.empty or len(minute_bars) < 30:
            return self._default_microstructure_features()
        
        try:
            recent = minute_bars.tail(30)  # Last 30 minutes
            
            features = {
                'vwap': recent['vwap'].mean() if 'vwap' in recent.columns else recent['close'].mean(),
                'price_vs_vwap': (recent['close'].iloc[-1] - recent['vwap'].mean()) / recent['vwap'].mean() 
                                if 'vwap' in recent.columns and recent['vwap'].mean() > 0 else 0,
                'volume_imbalance': (recent['volume'] - recent['volume'].mean()).sum(),
                'price_efficiency': np.corrcoef(range(len(recent)), recent['close'])[0, 1],
            }
            
            # High frequency returns
            hf_returns = recent['close'].pct_change().dropna()
            if len(hf_returns) > 1:
                features['hf_volatility'] = hf_returns.std()
                features['hf_skewness'] = hf_returns.skew()
                features['hf_kurtosis'] = hf_returns.kurtosis()
            
            return features
            
        except Exception as e:
            logger.error(f"Microstructure calculation failed: {e}")
            return self._default_microstructure_features()
    
    def _calculate_strategy_features(self, features: Dict) -> Dict:
        """Calculate features specifically for wheel strategy decisions"""
        strategy_features = {}
        
        try:
            # Regime detection
            volatility = features.get('ewma_vol_30', 0.2)
            momentum = features.get('momentum_30', 0)
            
            if volatility > 0.3 and abs(momentum) > 0.05:
                regime = 'volatile'
            elif volatility < 0.15 and abs(momentum) < 0.02:
                regime = 'calm'
            elif momentum > 0.03:
                regime = 'bullish'
            elif momentum < -0.03:
                regime = 'bearish'
            else:
                regime = 'neutral'
            
            strategy_features['regime'] = regime
            strategy_features['regime_score'] = self._regime_to_score(regime)
            
            # Wheel strategy signals
            iv_rank = features.get('iv_rank', 0.5)
            put_call_ratio = features.get('put_call_volume_ratio', 1.0)
            
            # Signal for selling puts (main wheel entry)
            put_signal = (
                (iv_rank > 0.6) * 0.3 +  # High IV
                (put_call_ratio > 1.2) * 0.2 +  # Fear in market
                (regime in ['bearish', 'volatile']) * 0.3 +  # Market conditions
                (volatility > 0.25) * 0.2  # High vol
            )
            
            strategy_features['put_sell_signal'] = min(put_signal, 1.0)
            
            # Signal for selling calls (when assigned)
            call_signal = (
                (iv_rank > 0.4) * 0.4 +  # Decent IV
                (regime in ['bullish', 'neutral']) * 0.3 +  # Upward bias OK
                (momentum > 0) * 0.3  # Positive momentum
            )
            
            strategy_features['call_sell_signal'] = min(call_signal, 1.0)
            
        except Exception as e:
            logger.error(f"Strategy feature calculation failed: {e}")
        
        return strategy_features
    
    def _regime_to_score(self, regime: str) -> float:
        """Convert regime to numeric score for TFT model"""
        regime_map = {
            'bullish': 1.0,
            'neutral': 0.5,
            'bearish': 0.0,
            'volatile': 0.25,
            'calm': 0.75
        }
        return regime_map.get(regime, 0.5)
    
    def _default_stock_features(self) -> Dict:
        """Default stock features when data is missing"""
        return {
            'price': 0.0, 'open': 0.0, 'high': 0.0, 'low': 0.0, 'close': 0.0,
            'volume': 0, 'prev_close': 0.0, 'daily_return': 0.0,
            'net_change': 0.0, 'mark_pct': 0.0, 'day_range': 0.0,
            'gap': 0.0, 'close_position': 0.5
        }
    
    def _default_technical_features(self) -> Dict:
        """Default technical features when data is missing - vectorized generation"""
        # Use dictionary comprehension for vectorized generation of defaults
        feature_templates = {
            'ewma_vol': 0.2,
            'realized_vol': 0.2,
            'mean_return': 0.0,
            'momentum': 0.0,
            'avg_volume': 1000000,
            'volume_trend': 0.0
        }
        
        # Generate all features for all windows at once
        defaults = {}
        for feature_name, default_value in feature_templates.items():
            defaults.update({f'{feature_name}_{window}': default_value for window in self.lookback_windows})
        
        # Add additional features for windows >= 20 using vectorized generation
        large_windows = [w for w in self.lookback_windows if w >= 20]
        if large_windows:
            bb_defaults = {f'bb_position_{window}': 0.0 for window in large_windows}
            rsi_defaults = {f'rsi_{window}': 50.0 for window in large_windows}
            defaults.update(bb_defaults)
            defaults.update(rsi_defaults)
        
        return defaults
    
    def _default_option_features(self) -> Dict:
        """Default option features when data is missing"""
        return {
            'iv_mean': 0.25, 'iv_std': 0.05, 'iv_skew': 0.0, 'iv_rank': 0.5,
            'put_call_volume_ratio': 1.0, 'put_call_oi_ratio': 1.0,
            'market_delta': 0.0, 'market_gamma': 0.0,
            'best_put_strike': 0.0, 'best_put_premium': 0.0, 'best_put_delta': -0.2,
            'best_put_dte': 30, 'best_put_yield': 0.01, 'capital_at_risk': 0.0,
            'best_call_strike': 0.0, 'best_call_premium': 0.0, 'best_call_delta': 0.2,
            'best_call_dte': 30, 'best_call_yield': 0.01
        }
    
    def _default_microstructure_features(self) -> Dict:
        """Default microstructure features when data is missing"""
        return {
            'vwap': 0.0, 'price_vs_vwap': 0.0, 'volume_imbalance': 0.0,
            'price_efficiency': 0.0, 'hf_volatility': 0.02,
            'hf_skewness': 0.0, 'hf_kurtosis': 3.0
        }
    
    def _default_features(self, ticker: str) -> Dict:
        """Complete default feature set"""
        return {
            **self._default_stock_features(),
            **self._default_technical_features(),
            **self._default_option_features(),
            **self._default_microstructure_features(),
            'regime': 'neutral',
            'regime_score': 0.5,
            'put_sell_signal': 0.0,
            'call_sell_signal': 0.0,
            'timestamp': pd.Timestamp.now(),
            'ticker': ticker
        }
    
    def calculate_missing_greeks(self, option_type: str, spot: float, strike: float, 
                               time_to_expiry: float, risk_free_rate: float = None, 
                               volatility: float = 0.25) -> Dict:
        """Calculate missing Greeks using Black-Scholes approximation"""
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        try:
            # Prevent division by zero
            if time_to_expiry <= 0 or volatility <= 0 or spot <= 0 or strike <= 0:
                return {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0}
            
            # Black-Scholes calculations
            d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
            d2 = d1 - volatility * np.sqrt(time_to_expiry)
            
            # Delta
            if option_type.lower() == 'call':
                delta = norm.cdf(d1)
            else:  # put
                delta = norm.cdf(d1) - 1
            
            # Gamma (same for calls and puts)
            gamma = norm.pdf(d1) / (spot * volatility * np.sqrt(time_to_expiry))
            
            # Theta
            theta_common = -(spot * norm.pdf(d1) * volatility) / (2 * np.sqrt(time_to_expiry))
            if option_type.lower() == 'call':
                theta = theta_common - risk_free_rate * strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
            else:  # put
                theta = theta_common + risk_free_rate * strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2)
            
            # Vega (same for calls and puts)
            vega = spot * norm.pdf(d1) * np.sqrt(time_to_expiry)
            
            return {
                'delta': delta,
                'gamma': gamma,
                'theta': theta / 365,  # Per day
                'vega': vega / 100  # Per 1% vol change
            }
            
        except Exception as e:
            logger.error(f"Greeks calculation failed: {e}")
            return {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0}
    
    def prepare_tft_features(self, all_features: Dict) -> pd.DataFrame:
        """Prepare features in format expected by TFT model"""
        try:
            # Select key features for TFT model
            tft_feature_cols = [
                'price', 'daily_return', 'ewma_vol_30', 'momentum_30',
                'iv_mean', 'iv_rank', 'put_call_volume_ratio',
                'regime_score', 'put_sell_signal', 'call_sell_signal',
                'best_put_yield', 'best_call_yield'
            ]
            
            # Create DataFrame with required columns
            feature_dict = {}
            for col in tft_feature_cols:
                feature_dict[col] = all_features.get(col, 0.0)
            
            # Add time features
            feature_dict['time_idx'] = 0  # Will be set properly in TFT inference
            feature_dict['ticker'] = all_features.get('ticker', 'UNKNOWN')
            
            df = pd.DataFrame([feature_dict])
            
            # Ensure numeric types
            numeric_cols = [col for col in tft_feature_cols if col != 'ticker']
            df[numeric_cols] = df[numeric_cols].astype(float)
            
            return df
            
        except Exception as e:
            logger.error(f"TFT feature preparation failed: {e}")
            # Return minimal DataFrame
            return pd.DataFrame([{
                'ticker': all_features.get('ticker', 'UNKNOWN'),
                'time_idx': 0,
                'price': 0.0,
                'daily_return': 0.0
            }])

if __name__ == "__main__":
    engineer = FeatureEngineer()
    stock_data = {
        'day': {'c': 24.10, 'h': 24.50, 'l': 23.80, 'o': 24.00, 'v': 1000000},
        'prev_day': {'c': 24.00}
    }
    features = engineer._extract_stock_features(stock_data)
    print("Stock features:", features)
