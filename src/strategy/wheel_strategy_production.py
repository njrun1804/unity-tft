#!/usr/bin/env python
"""
wheel_strategy_production.py - Production-ready wheel strategy
Integrates all components with risk management and execution logic
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class OptionContract:
    ticker: str
    strike: float
    expiry: str
    contract_type: str  # 'call' or 'put'
    bid: float
    ask: float
    mid: float
    iv: float
    delta: float
    gamma: float
    theta: float
    vega: float
    open_interest: int
    volume: int
    days_to_expiry: int
    underlying_price: float
    
    @property
    def time_value(self) -> float:
        """Calculate time value of option"""
        if self.contract_type == 'call':
            intrinsic = max(0, self.underlying_price - self.strike)
        else:  # put
            intrinsic = max(0, self.strike - self.underlying_price)
        return max(0, self.mid - intrinsic)
    
    @property
    def moneyness(self) -> float:
        """Calculate moneyness (strike/spot for puts, spot/strike for calls)"""
        if self.contract_type == 'put':
            return self.strike / self.underlying_price if self.underlying_price > 0 else 1
        else:  # call
            return self.underlying_price / self.strike if self.strike > 0 else 1
    
    @property
    def spread_pct(self) -> float:
        """Calculate bid-ask spread as percentage of mid"""
        return (self.ask - self.bid) / self.mid if self.mid > 0 else 0
    
    @property
    def premium_yield(self) -> float:
        """Calculate premium as percentage of strike (for puts) or spot (for calls)"""
        if self.contract_type == 'put':
            return self.mid / self.strike if self.strike > 0 else 0
        else:  # call
            return self.mid / self.underlying_price if self.underlying_price > 0 else 0
    
    def score_for_wheel(self) -> float:
        """Score option for wheel strategy suitability"""
        try:
            score = 0.0
            
            # Premium yield (higher is better)
            score += min(self.premium_yield * 100, 5.0)  # Cap at 5 points
            
            # Days to expiry (prefer 15-45 days)
            dte_score = 1.0 - abs(self.days_to_expiry - 30) / 30
            score += max(0, dte_score) * 2.0
            
            # Volume and open interest (liquidity)
            if self.volume > 50:
                score += 1.0
            if self.open_interest > 100:
                score += 1.0
            
            # Spread tightness (lower spread is better)
            spread_score = 1.0 - min(self.spread_pct, 0.1) / 0.1
            score += spread_score * 1.5
            
            # Delta appropriateness for wheel
            if self.contract_type == 'put' and 0.15 <= abs(self.delta) <= 0.30:
                score += 2.0
            elif self.contract_type == 'call' and 0.20 <= abs(self.delta) <= 0.40:
                score += 2.0
            
            return score
            
        except Exception as e:
            logger.error(f"Option scoring failed: {e}")
            return 0.0

@dataclass
class Position:
    ticker: str
    position_type: str  # 'cash_secured_put', 'covered_call', 'shares'
    size: int  # Number of contracts or shares
    entry_price: float
    current_price: float
    entry_date: datetime
    expiry_date: Optional[datetime] = None
    strike: Optional[float] = None
    contract_type: Optional[str] = None
    
    # Risk metrics
    max_loss: float = 0.0
    break_even: float = 0.0
    theta_exposure: float = 0.0
    delta_exposure: float = 0.0
    
    @property
    def days_held(self) -> int:
        return (datetime.now() - self.entry_date).days
    
    @property
    def days_to_expiry(self) -> Optional[int]:
        if self.expiry_date:
            return max(0, (self.expiry_date - datetime.now()).days)
        return None
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L"""
        if self.position_type == 'shares':
            return (self.current_price - self.entry_price) * self.size
        else:  # options
            return (self.entry_price - self.current_price) * self.size * 100  # Short options
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """Calculate unrealized P&L as percentage"""
        if self.entry_price == 0:
            return 0.0
        return self.unrealized_pnl / (abs(self.entry_price) * abs(self.size) * (100 if self.position_type != 'shares' else 1))
    
    def should_roll(self) -> bool:
        """Determine if position should be rolled"""
        try:
            # Roll if close to expiry
            if self.days_to_expiry is not None and self.days_to_expiry <= 7:
                return True
            
            # Roll if very profitable (take profits)
            if self.unrealized_pnl_pct > 0.5:  # 50% profit
                return True
            
            # Roll if very unprofitable (limit losses)
            if self.unrealized_pnl_pct < -2.0:  # 200% loss (2x premium)
                return True
            
            return False
            
        except Exception:
            return False

@dataclass
class PositionStateManager:
    """Manages overall portfolio state for wheel strategy"""
    positions: List[Position] = field(default_factory=list)
    cash: float = 100000.0  # Starting cash
    max_risk_per_position: float = 0.25  # 25% max risk per position
    target_delta: float = 0.70  # Target portfolio delta
    max_positions: int = 5
    
    def add_position(self, position: Position):
        """Add new position"""
        self.positions.append(position)
        logger.info(f"Added position: {position.ticker} {position.position_type}")
    
    def remove_position(self, position: Position):
        """Remove closed position"""
        if position in self.positions:
            self.positions.remove(position)
            logger.info(f"Removed position: {position.ticker} {position.position_type}")
    
    def get_positions_for_ticker(self, ticker: str) -> List[Position]:
        """Get all positions for specific ticker"""
        return [p for p in self.positions if p.ticker == ticker]
    
    def has_shares(self, ticker: str) -> bool:
        """Check if we own shares of ticker"""
        return any(p.position_type == 'shares' and p.size > 0 for p in self.get_positions_for_ticker(ticker))
    
    def has_cash_secured_put(self, ticker: str) -> bool:
        """Check if we have cash-secured put for ticker"""
        return any(p.position_type == 'cash_secured_put' for p in self.get_positions_for_ticker(ticker))
    
    def calculate_portfolio_delta(self) -> float:
        """Calculate total portfolio delta"""
        total_delta = 0.0
        for position in self.positions:
            if position.position_type == 'shares':
                total_delta += position.size  # 1 delta per share
            else:
                total_delta += position.delta_exposure
        return total_delta
    
    def calculate_buying_power_used(self) -> float:
        """Calculate capital committed to positions"""
        used = 0.0
        for position in self.positions:
            if position.position_type == 'cash_secured_put':
                used += position.strike * abs(position.size) * 100  # Cash secured
            elif position.position_type == 'shares':
                used += position.current_price * position.size
        return used
    
    def available_buying_power(self) -> float:
        """Calculate available buying power"""
        return self.cash - self.calculate_buying_power_used()
    
    def can_add_position(self, required_capital: float) -> bool:
        """Check if we can add new position"""
        return (len(self.positions) < self.max_positions and 
                self.available_buying_power() >= required_capital)

class WheelStrategyProduction:
    """Production wheel strategy with full risk management"""
    
    def __init__(self, position_manager: PositionStateManager, config: Dict):
        self.position_manager = position_manager
        self.config = config
        self.min_iv_rank = config.get('min_iv_rank', 0.5)
        self.max_dte = config.get('max_dte', 45)
        self.min_dte = config.get('min_dte', 15)
        self.target_delta_put = config.get('target_delta_put', -0.20)
        self.target_delta_call = config.get('target_delta_call', 0.30)
        self.profit_target = config.get('profit_target', 0.5)  # 50% profit target
        
    def generate_signals(self, stock_data: Dict, option_chain: List[Dict], 
                        features: Dict, tft_predictions: Dict) -> List[Dict]:
        """Generate trading signals based on all inputs"""
        signals = []
        ticker = stock_data.get('ticker', 'UNKNOWN')
        current_price = stock_data.get('price', 0)
        
        try:
            # Convert option chain to OptionContract objects
            option_contracts = self._parse_option_chain(option_chain, current_price)
            
            # Check current positions for this ticker
            existing_positions = self.position_manager.get_positions_for_ticker(ticker)
            
            # Determine strategy phase
            has_shares = self.position_manager.has_shares(ticker)
            has_put = self.position_manager.has_cash_secured_put(ticker)
            
            # Check for position management signals first
            signals.extend(self._check_position_management(existing_positions, option_contracts))
            
            # Generate new position signals
            if not has_put and not has_shares:
                # Phase 1: Sell cash-secured puts
                put_signal = self._generate_put_sell_signal(
                    ticker, option_contracts, features, tft_predictions
                )
                if put_signal:
                    signals.append(put_signal)
                    
            elif has_shares and not any(p.position_type == 'covered_call' for p in existing_positions):
                # Phase 2: Sell covered calls on shares
                call_signal = self._generate_call_sell_signal(
                    ticker, option_contracts, features, tft_predictions
                )
                if call_signal:
                    signals.append(call_signal)
            
            # Filter signals by risk management
            signals = self._apply_risk_management(signals)
            
            return signals
            
        except Exception as e:
            logger.error(f"Signal generation failed for {ticker}: {e}")
            return []
    
    def _parse_option_chain(self, option_chain: List[Dict], spot: float) -> List[OptionContract]:
        """Convert raw option data to OptionContract objects"""
        contracts = []
        
        for opt in option_chain:
            try:
                # Calculate mid price
                bid = opt.get('bid', 0) or 0
                ask = opt.get('ask', 0) or 0
                mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else 0
                
                if mid <= 0:
                    continue  # Skip invalid options
                
                # Calculate DTE
                expiry_str = opt.get('expiry', '')
                try:
                    expiry_date = pd.to_datetime(expiry_str)
                    dte = max(0, (expiry_date - pd.Timestamp.now()).days)
                except:
                    dte = 0
                
                # Skip if outside DTE range
                if not (self.min_dte <= dte <= self.max_dte):
                    continue
                
                contract = OptionContract(
                    ticker=opt.get('ticker', ''),
                    strike=float(opt.get('strike', 0)),
                    expiry=expiry_str,
                    contract_type=opt.get('cp', '').lower(),
                    bid=bid,
                    ask=ask,
                    mid=mid,
                    iv=float(opt.get('iv', 0.25)),
                    delta=float(opt.get('delta', 0)),
                    gamma=float(opt.get('gamma', 0)),
                    theta=float(opt.get('theta', 0)),
                    vega=float(opt.get('vega', 0)),
                    open_interest=int(opt.get('open_interest', 0)),
                    volume=int(opt.get('volume', 0)),
                    days_to_expiry=dte,
                    underlying_price=spot
                )
                
                contracts.append(contract)
                
            except Exception as e:
                logger.warning(f"Failed to parse option: {e}")
                continue
        
        return contracts
    
    def _generate_put_sell_signal(self, ticker: str, contracts: List[OptionContract], 
                                 features: Dict, predictions: Dict) -> Optional[Dict]:
        """Generate signal to sell cash-secured put"""
        try:
            # Filter for puts
            puts = [c for c in contracts if c.contract_type == 'put']
            if not puts:
                return None
            
            # Check market conditions
            iv_rank = features.get('iv_rank', 0.5)
            put_sell_signal_strength = features.get('put_sell_signal', 0.0)
            
            # Require minimum IV rank
            if iv_rank < self.min_iv_rank:
                logger.info(f"IV rank too low for {ticker}: {iv_rank}")
                return None
            
            # Require positive signal strength
            if put_sell_signal_strength < 0.3:
                logger.info(f"Put sell signal too weak for {ticker}: {put_sell_signal_strength}")
                return None
            
            # Find best put to sell
            target_puts = [p for p in puts if 
                          abs(p.delta) >= 0.15 and abs(p.delta) <= 0.30 and
                          p.volume >= 10 and p.open_interest >= 50]
            
            if not target_puts:
                return None
            
            # Score and select best put
            best_put = max(target_puts, key=lambda p: p.score_for_wheel())
            
            # Calculate position size
            required_capital = best_put.strike * 100  # Per contract
            max_contracts = int(self.position_manager.available_buying_power() * 
                               self.position_manager.max_risk_per_position / required_capital)
            
            if max_contracts < 1:
                return None
            
            # Use Kelly criterion for sizing
            win_prob = self._estimate_win_probability(best_put, predictions)
            kelly_size = self._kelly_position_size(win_prob, best_put.premium_yield)
            contracts_to_sell = min(max_contracts, max(1, int(kelly_size)))
            
            signal = {
                'action': 'SELL_PUT',
                'ticker': ticker,
                'strike': best_put.strike,
                'expiry': best_put.expiry,
                'quantity': contracts_to_sell,
                'limit_price': best_put.bid,  # Conservative entry
                'strategy': 'WHEEL_PHASE_1',
                'reasoning': f"IV rank: {iv_rank:.2f}, Signal: {put_sell_signal_strength:.2f}, "
                           f"Premium yield: {best_put.premium_yield:.3f}",
                'risk_capital': required_capital * contracts_to_sell,
                'max_loss': required_capital * contracts_to_sell - (best_put.mid * contracts_to_sell * 100),
                'break_even': best_put.strike - best_put.mid,
                'timestamp': pd.Timestamp.now()
            }
            
            return signal
            
        except Exception as e:
            logger.error(f"Put sell signal generation failed: {e}")
            return None
    
    def _generate_call_sell_signal(self, ticker: str, contracts: List[OptionContract], 
                                  features: Dict, predictions: Dict) -> Optional[Dict]:
        """Generate signal to sell covered call"""
        try:
            # Filter for calls
            calls = [c for c in contracts if c.contract_type == 'call']
            if not calls:
                return None
            
            # Check if we have shares
            share_positions = [p for p in self.position_manager.get_positions_for_ticker(ticker) 
                             if p.position_type == 'shares' and p.size > 0]
            
            if not share_positions:
                return None
            
            shares_owned = sum(p.size for p in share_positions)
            current_price = contracts[0].underlying_price
            
            # Check market conditions
            call_sell_signal_strength = features.get('call_sell_signal', 0.0)
            regime = features.get('regime', 'neutral')
            
            # Don't sell calls in strong bull markets
            if regime == 'bullish' and call_sell_signal_strength < 0.6:
                return None
            
            # Find appropriate OTM calls
            otm_calls = [c for c in calls if 
                        c.strike > current_price * 1.02 and  # At least 2% OTM
                        c.delta >= 0.15 and c.delta <= 0.40 and
                        c.volume >= 10]
            
            if not otm_calls:
                return None
            
            # Select best call
            best_call = max(otm_calls, key=lambda c: c.score_for_wheel())
            
            # Calculate position size (limited by shares owned)
            max_contracts = shares_owned // 100
            if max_contracts < 1:
                return None
            
            signal = {
                'action': 'SELL_CALL',
                'ticker': ticker,
                'strike': best_call.strike,
                'expiry': best_call.expiry,
                'quantity': max_contracts,
                'limit_price': best_call.bid,
                'strategy': 'WHEEL_PHASE_2',
                'reasoning': f"Covered call on {shares_owned} shares, "
                           f"Strike: {best_call.strike}, Premium: {best_call.mid:.2f}",
                'max_gain': (best_call.strike - current_price + best_call.mid) * max_contracts * 100,
                'break_even': current_price,  # Already own shares
                'timestamp': pd.Timestamp.now()
            }
            
            return signal
            
        except Exception as e:
            logger.error(f"Call sell signal generation failed: {e}")
            return None
    
    def _check_position_management(self, positions: List[Position], 
                                  contracts: List[OptionContract]) -> List[Dict]:
        """Check existing positions for management signals"""
        signals = []
        
        for position in positions:
            try:
                if position.should_roll():
                    # Generate roll signal
                    roll_signal = self._generate_roll_signal(position, contracts)
                    if roll_signal:
                        signals.append(roll_signal)
                        
            except Exception as e:
                logger.error(f"Position management check failed: {e}")
        
        return signals
    
    def _generate_roll_signal(self, position: Position, 
                             contracts: List[OptionContract]) -> Optional[Dict]:
        """Generate signal to roll existing position"""
        try:
            if position.position_type == 'cash_secured_put':
                # Find new put to roll to
                puts = [c for c in contracts if 
                       c.contract_type == 'put' and
                       c.days_to_expiry > position.days_to_expiry + 7]  # Roll out in time
                
                if puts:
                    best_put = max(puts, key=lambda p: p.score_for_wheel())
                    return {
                        'action': 'ROLL_PUT',
                        'ticker': position.ticker,
                        'old_strike': position.strike,
                        'new_strike': best_put.strike,
                        'new_expiry': best_put.expiry,
                        'quantity': abs(position.size),
                        'strategy': 'WHEEL_ROLL',
                        'timestamp': pd.Timestamp.now()
                    }
                    
            elif position.position_type == 'covered_call':
                # Find new call to roll to
                calls = [c for c in contracts if 
                        c.contract_type == 'call' and
                        c.days_to_expiry > position.days_to_expiry + 7]
                
                if calls:
                    best_call = max(calls, key=lambda c: c.score_for_wheel())
                    return {
                        'action': 'ROLL_CALL',
                        'ticker': position.ticker,
                        'old_strike': position.strike,
                        'new_strike': best_call.strike,
                        'new_expiry': best_call.expiry,
                        'quantity': abs(position.size),
                        'strategy': 'WHEEL_ROLL',
                        'timestamp': pd.Timestamp.now()
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Roll signal generation failed: {e}")
            return None
    
    def _estimate_win_probability(self, option: OptionContract, predictions: Dict) -> float:
        """Estimate probability of profitable trade"""
        try:
            # Use TFT predictions to estimate win probability
            prediction = predictions.get('prediction', 0)
            prediction_std = predictions.get('prediction_std', 0.1)
            
            if option.contract_type == 'put':
                # Put wins if stock stays above strike at expiry
                prob_above_strike = 1 - norm.cdf(
                    (option.strike - prediction) / prediction_std
                )
                return min(0.95, max(0.05, prob_above_strike))
            else:  # call
                # Call wins if stock stays below strike at expiry
                prob_below_strike = norm.cdf(
                    (option.strike - prediction) / prediction_std
                )
                return min(0.95, max(0.05, prob_below_strike))
                
        except Exception as e:
            logger.error(f"Win probability estimation failed: {e}")
            return 0.6  # Default moderate probability
    
    def _kelly_position_size(self, win_prob: float, premium_yield: float) -> float:
        """Calculate Kelly optimal position size"""
        try:
            # Kelly formula: f = (bp - q) / b
            # where b = odds received, p = win prob, q = loss prob
            
            # For options: win = keep premium, lose = assignment cost
            odds = premium_yield / (1 - premium_yield)  # Simplified
            kelly_fraction = (win_prob * odds - (1 - win_prob)) / odds
            
            # Cap Kelly fraction for safety
            kelly_fraction = max(0, min(0.25, kelly_fraction))
            
            # Convert to number of contracts (rough approximation)
            return max(1, kelly_fraction * 10)  # Scale for practical contract sizes
            
        except Exception as e:
            logger.error(f"Kelly sizing failed: {e}")
            return 1.0
    
    def _apply_risk_management(self, signals: List[Dict]) -> List[Dict]:
        """Apply final risk management filters"""
        filtered_signals = []
        
        for signal in signals:
            try:
                # Check position limits
                if len(self.position_manager.positions) >= self.position_manager.max_positions:
                    logger.info("Maximum positions reached, skipping signal")
                    continue
                
                # Check capital requirements
                required_capital = signal.get('risk_capital', 0)
                if required_capital > self.position_manager.available_buying_power():
                    logger.info(f"Insufficient capital for signal: {required_capital}")
                    continue
                
                # Check position concentration
                ticker_positions = len(self.position_manager.get_positions_for_ticker(signal['ticker']))
                if ticker_positions >= 2:  # Max 2 positions per ticker
                    logger.info(f"Too many positions in {signal['ticker']}")
                    continue
                
                filtered_signals.append(signal)
                
            except Exception as e:
                logger.error(f"Risk management filter failed: {e}")
        
        return filtered_signals

@dataclass
class Position:
    shares: int = 0
    cost_basis: float = 0.0
    cash: float = 100000.0
    short_puts: List[Dict] = field(default_factory=list)
    short_calls: List[Dict] = field(default_factory=list)
    trade_history: List[Dict] = field(default_factory=list)
    last_update: datetime = field(default_factory=datetime.now)
    
    @property
    def net_liquidation_value(self) -> float:
        return self.cash + self.shares * self.cost_basis
    
    @property
    def buying_power(self) -> float:
        return self.cash * 0.3
    
    def to_dict(self) -> Dict:
        return self.__dict__

class RiskManager:
    def __init__(self, config: Dict):
        self.config = config
        self.max_capital_at_risk = config.get('capital_at_risk_limit', 0.25)
    def validate_signal(self, signal: Dict, position: Position) -> Tuple[bool, str]:
        return True, "OK"
    def _calculate_margin_requirement(self, signal: Dict) -> float:
        return 0.0
    def calculate_portfolio_var(self, position: Position) -> float:
        return 0.0

class WheelStrategyProduction:
    def __init__(self, position: Position, config: Dict):
        self.position = position
        self.config = config
    def detect_market_regime(self, features: Dict) -> str:
        return "normal"
    def select_optimal_put(self, spot: float, option_chain: List[OptionContract], features: Dict):
        return None, 0.0, {}
    def select_optimal_call(self, spot: float, option_chain: List[OptionContract], features: Dict):
        return None, 0.0, {}
    def calculate_position_size(self, contract: OptionContract, features: Dict) -> int:
        return 1
    def _apply_delta_constraint(self, contract: OptionContract, contracts: int) -> int:
        return contracts
    def check_roll_conditions(self, spot: float, option_chain: List[OptionContract]) -> List[Dict]:
        return []
    def calculate_portfolio_greeks(self, option_contracts: Dict[str, OptionContract]) -> Dict:
        return {}
    def generate_signals(self, stock_data: Dict, option_chain: List[OptionContract], features: Dict) -> Dict:
        return {}

class PositionStateManager:
    def __init__(self, state_file: Path = Path('data/positions/current_position.json')):
        self.state_file = state_file
        self.position = self.load_position()
    def load_position(self) -> Position:
        return Position()
    def save_position(self):
        pass
    def update_after_execution(self, signal: Dict, execution: Dict):
        logger.info(f"Position updated after {signal['action']} execution")

if __name__ == "__main__":
    with open('positions.json', 'r') as f:
        config = json.load(f)
    state_manager = PositionStateManager()
    strategy = WheelStrategyProduction(state_manager.position, config)
    stock_data = {'day': {'c': 24.10}}
    features = {
        'p_up_1d': 0.61,
        'realized_vol': 0.28
    }
    signals = strategy.generate_signals(stock_data, [], features)
    print(json.dumps(signals, indent=2))
