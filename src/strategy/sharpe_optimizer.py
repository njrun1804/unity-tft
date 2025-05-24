#!/usr/bin/env python
"""
sharpe_optimizer.py - Sharpe ratio optimization for position sizing
Optimizes position sizes based on expected Sharpe ratios and Kelly criterion
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class AlphaInputs:
    """Inputs for Sharpe ratio calculation"""
    mu_tft: float  # TFT predicted return
    mu_skew: float  # Skew-based return prediction
    mu_news: float  # News sentiment return
    ewma_vol: float  # EWMA volatility
    confidence: float  # Model confidence [0,1]

class SharpeOptimizer:
    """
    Optimizes position sizes using ex-ante Sharpe ratio calculations
    """
    
    def __init__(self, capital: float = 100000.0, config: Dict = None):
        self.capital = capital
        self.config = config or {}
        
        # Position sizing parameters
        self.max_kelly_limit = self.config.get('max_kelly_limit', 0.25)
        self.base_size = self.config.get('base_size', 0.05)  # 5% base position size
        self.confidence_threshold = self.config.get('confidence_threshold', 0.2)
        
        logger.info(f"Sharpe optimizer initialized with ${capital:,.2f} capital")
    
    def ex_ante_sharpe(self, inputs: AlphaInputs, time_scale: float = 1.0) -> float:
        """
        Calculate expected Sharpe ratio using TFT predictions and market data
        
        Args:
            inputs: AlphaInputs containing predictions and volatility
            time_scale: Time scaling factor (e.g., 252 for annualized)
            
        Returns:
            Expected Sharpe ratio
        """
        try:
            # Combined expected return (equal weighted for now)
            mu_combined = (inputs.mu_tft + inputs.mu_skew + inputs.mu_news) / 3
            
            # Scale return and volatility
            risk_adjusted_return = mu_combined * time_scale
            volatility_adjusted = inputs.ewma_vol * np.sqrt(time_scale)
            
            # Avoid division by zero
            if volatility_adjusted == 0:
                return 0.0
                
            sharpe = risk_adjusted_return / volatility_adjusted
            
            # Weight by confidence
            weighted_sharpe = sharpe * inputs.confidence
            
            return weighted_sharpe
            
        except Exception as e:
            logger.error(f"Sharpe calculation failed: {e}")
            return 0.0
    
    def kelly_optimal_size(self, win_prob: float, payoff_ratio: float, 
                          confidence: float = 1.0) -> float:
        """
        Calculate Kelly optimal position size
        
        Args:
            win_prob: Probability of winning trade
            payoff_ratio: Ratio of win amount to loss amount
            confidence: Model confidence [0,1]
            
        Returns:
            Optimal position size as fraction of capital
        """
        try:
            # Ensure confidence is between 0 and 1
            confidence = np.clip(confidence, 0, 1)  # Ensure confidence is between 0 and 1
            
            # Kelly formula: f = (bp - q) / b
            # where b = payoff ratio, p = win prob, q = loss prob
            kelly_fraction = (payoff_ratio * win_prob - (1 - win_prob)) / payoff_ratio
            
            # Apply safety cap
            kelly_fraction = max(0, min(self.max_kelly_limit, kelly_fraction))
            
            # Calculate position size
            size = kelly_fraction * confidence * self.capital
            
            return size
            
        except Exception as e:
            logger.error(f"Kelly sizing failed: {e}")
            return 0.0
    
    def optimize_portfolio_positions(self, signals: List[Dict]) -> List[Dict]:
        """
        Optimize position sizes for a portfolio of signals
        
        Args:
            signals: List of trading signals with alpha inputs
            
        Returns:
            Optimized signals with position sizes
        """
        optimized_signals = []
        total_capital_allocated = 0.0
        
        # Calculate Sharpe ratios for all signals
        signal_sharpes = []
        for signal in signals:
            try:
                alpha_inputs = AlphaInputs(
                    mu_tft=signal.get('expected_return_tft', 0),
                    mu_skew=signal.get('expected_return_skew', 0), 
                    mu_news=signal.get('expected_return_news', 0),
                    ewma_vol=signal.get('volatility', 0.2),
                    confidence=signal.get('confidence', 0.5)
                )
                
                sharpe = self.ex_ante_sharpe(alpha_inputs)
                signal_sharpes.append((signal, sharpe, alpha_inputs))
                
            except Exception as e:
                logger.error(f"Failed to calculate Sharpe for signal: {e}")
                continue
        
        # Sort by Sharpe ratio (descending)
        signal_sharpes.sort(key=lambda x: x[1], reverse=True)
        
        # Allocate capital based on Sharpe ratios
        for signal, sharpe, alpha_inputs in signal_sharpes:
            if total_capital_allocated >= self.capital * 0.95:  # Leave 5% cash buffer
                break
                
            if sharpe <= 0:
                continue
                
            # Calculate position size using Kelly criterion
            win_prob = signal.get('win_probability', 0.6)
            payoff_ratio = signal.get('payoff_ratio', 2.0)
            
            optimal_size = self.kelly_optimal_size(
                win_prob=win_prob,
                payoff_ratio=payoff_ratio,
                confidence=alpha_inputs.confidence
            )
            
            # Ensure we don't exceed available capital
            available_capital = self.capital - total_capital_allocated
            position_size = min(optimal_size, available_capital)
            
            if position_size > 0:
                optimized_signal = signal.copy()
                optimized_signal.update({
                    'position_size': position_size,
                    'sharpe_ratio': sharpe,
                    'kelly_fraction': optimal_size / self.capital,
                    'confidence_adjusted': alpha_inputs.confidence
                })
                
                optimized_signals.append(optimized_signal)
                total_capital_allocated += position_size
                
                logger.info(f"Optimized {signal.get('ticker', 'UNKNOWN')}: "
                           f"Size=${position_size:,.0f}, Sharpe={sharpe:.3f}")
        
        logger.info(f"Portfolio optimization complete: "
                   f"{len(optimized_signals)} positions, "
                   f"${total_capital_allocated:,.0f} allocated")
        
        return optimized_signals
    
    def rebalance_portfolio(self, current_positions: List[Dict], 
                           new_signals: List[Dict]) -> List[Dict]:
        """
        Rebalance existing portfolio with new signals
        
        Args:
            current_positions: Current portfolio positions
            new_signals: New trading signals
            
        Returns:
            Rebalancing actions (buy/sell/hold)
        """
        actions = []
        
        # Calculate current portfolio allocation
        current_capital = sum(pos.get('market_value', 0) for pos in current_positions)
        available_capital = self.capital - current_capital
        
        # Optimize new signals
        optimized_signals = self.optimize_portfolio_positions(new_signals)
        
        # Generate rebalancing actions
        for signal in optimized_signals:
            ticker = signal.get('ticker')
            target_size = signal.get('position_size', 0)
            
            # Check if we already have position in this ticker
            current_position = next((p for p in current_positions 
                                   if p.get('ticker') == ticker), None)
            
            if current_position:
                current_value = current_position.get('market_value', 0)
                adjustment = target_size - current_value
                
                if abs(adjustment) > current_value * 0.1:  # 10% threshold for rebalancing
                    action_type = 'increase' if adjustment > 0 else 'decrease'
                    actions.append({
                        'action': action_type,
                        'ticker': ticker,
                        'adjustment': abs(adjustment),
                        'current_value': current_value,
                        'target_value': target_size
                    })
            else:
                # New position
                if target_size > 0 and target_size <= available_capital:
                    actions.append({
                        'action': 'open',
                        'ticker': ticker,
                        'position_size': target_size,
                        'signal': signal
                    })
        
        return actions
    
    def calculate_portfolio_sharpe(self, positions: List[Dict]) -> float:
        """
        Calculate portfolio-level Sharpe ratio
        
        Args:
            positions: List of current positions
            
        Returns:
            Portfolio Sharpe ratio
        """
        try:
            if not positions:
                return 0.0
            
            total_value = sum(pos.get('market_value', 0) for pos in positions)
            if total_value == 0:
                return 0.0
            
            # Weight-average individual Sharpe ratios
            weighted_sharpe = 0.0
            for position in positions:
                weight = position.get('market_value', 0) / total_value
                position_sharpe = position.get('sharpe_ratio', 0)
                weighted_sharpe += weight * position_sharpe
            
            return weighted_sharpe
            
        except Exception as e:
            logger.error(f"Portfolio Sharpe calculation failed: {e}")
            return 0.0

# Global optimizer instance
_sharpe_optimizer = None

def get_sharpe_optimizer(capital: float = None, config: Dict = None) -> SharpeOptimizer:
    """Get or create global Sharpe optimizer instance"""
    global _sharpe_optimizer
    if _sharpe_optimizer is None:
        _sharpe_optimizer = SharpeOptimizer(capital or 100000.0, config)
    return _sharpe_optimizer

if __name__ == "__main__":
    # Example usage
    optimizer = SharpeOptimizer(100000.0)
    
    # Test with sample alpha inputs
    test_inputs = AlphaInputs(
        mu_tft=0.08,
        mu_skew=0.05,
        mu_news=0.03,
        ewma_vol=0.25,
        confidence=0.75
    )
    
    sharpe = optimizer.ex_ante_sharpe(test_inputs, time_scale=252)
    print(f"Expected Sharpe ratio: {sharpe:.3f}")
    
    # Test Kelly sizing
    optimal_size = optimizer.kelly_optimal_size(
        win_prob=0.65,
        payoff_ratio=2.5,
        confidence=0.75
    )
    print(f"Optimal position size: ${optimal_size:,.2f}")
    
    # Test portfolio optimization
    sample_signals = [
        {
            'ticker': 'AAPL',
            'expected_return_tft': 0.12,
            'expected_return_skew': 0.08,
            'expected_return_news': 0.05,
            'volatility': 0.30,
            'confidence': 0.80,
            'win_probability': 0.70,
            'payoff_ratio': 2.0
        },
        {
            'ticker': 'MSFT', 
            'expected_return_tft': 0.10,
            'expected_return_skew': 0.06,
            'expected_return_news': 0.04,
            'volatility': 0.25,
            'confidence': 0.65,
            'win_probability': 0.65,
            'payoff_ratio': 1.8
        }
    ]
    
    optimized = optimizer.optimize_portfolio_positions(sample_signals)
    print(f"\nOptimized portfolio:")
    for signal in optimized:
        print(f"- {signal['ticker']}: ${signal['position_size']:,.0f} "
              f"(Sharpe: {signal['sharpe_ratio']:.3f})")
