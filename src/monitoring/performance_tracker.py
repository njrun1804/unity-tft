#!/usr/bin/env python
"""
performance_tracker.py - Real-time performance monitoring and metrics
Tracks P&L, risk metrics, and portfolio statistics in production
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging
from dataclasses import dataclass, field
from collections import defaultdict, deque
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class PositionMetrics:
    """Metrics for a single position"""
    ticker: str
    position_type: str  # 'stock', 'put', 'call'
    quantity: int
    entry_price: float
    current_price: float
    entry_date: datetime
    days_held: int
    unrealized_pnl: float
    realized_pnl: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    
    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def pnl_percent(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / abs(self.entry_price)

@dataclass
class PortfolioMetrics:
    """Portfolio-level performance metrics"""
    timestamp: datetime
    total_value: float
    cash: float
    equity_value: float
    options_value: float
    total_pnl: float
    daily_pnl: float
    unrealized_pnl: float
    realized_pnl: float
    
    # Risk metrics
    portfolio_delta: float = 0.0
    portfolio_gamma: float = 0.0
    portfolio_theta: float = 0.0
    portfolio_vega: float = 0.0
    
    # Performance metrics
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    avg_trade_duration: float = 0.0

class PerformanceTracker:
    """
    Real-time performance tracking system
    Monitors portfolio metrics, risk exposure, and trading performance
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Position tracking
        self.positions: Dict[str, PositionMetrics] = {}
        self.closed_positions: List[PositionMetrics] = []
        
        # Performance history
        self.daily_metrics: List[PortfolioMetrics] = []
        self.intraday_metrics: deque = deque(maxlen=1000)  # Last 1000 updates
        
        # Risk limits and alerts
        self.risk_limits = {
            'max_portfolio_delta': 0.80,
            'max_single_position': 0.25,
            'max_drawdown': 0.20,
            'min_sharpe_ratio': 1.0
        }
        
        # Performance calculation buffers
        self.returns_buffer = deque(maxlen=252)  # 1 year of daily returns
        self.pnl_history = deque(maxlen=1000)
        
        # Trade tracking
        self.trade_log: List[Dict] = []
        
        logger.info(f"Performance tracker initialized with ${initial_capital:,.2f}")
    
    def add_position(self, ticker: str, position_type: str, quantity: int, 
                    entry_price: float, entry_date: datetime = None,
                    greeks: Dict = None) -> str:
        """
        Add a new position to tracking
        Returns position ID for future reference
        """
        if entry_date is None:
            entry_date = datetime.now()
        
        greeks = greeks or {}
        position_id = f"{ticker}_{position_type}_{entry_date.strftime('%Y%m%d_%H%M%S')}"
        
        position = PositionMetrics(
            ticker=ticker,
            position_type=position_type,
            quantity=quantity,
            entry_price=entry_price,
            current_price=entry_price,
            entry_date=entry_date,
            days_held=0,
            unrealized_pnl=0.0,
            delta=greeks.get('delta', 0.0),
            gamma=greeks.get('gamma', 0.0),
            theta=greeks.get('theta', 0.0),
            vega=greeks.get('vega', 0.0)
        )
        
        self.positions[position_id] = position
        
        # Log the trade
        trade_record = {
            'timestamp': entry_date,
            'action': 'OPEN',
            'position_id': position_id,
            'ticker': ticker,
            'type': position_type,
            'quantity': quantity,
            'price': entry_price,
            'value': quantity * entry_price * (100 if 'option' in position_type.lower() else 1)
        }
        self.trade_log.append(trade_record)
        
        logger.info(f"Added position: {position_id} - {quantity} {position_type} @ ${entry_price:.2f}")
        return position_id
    
    def close_position(self, position_id: str, exit_price: float, 
                      exit_date: datetime = None, partial_quantity: int = None) -> float:
        """
        Close a position (full or partial) and calculate realized P&L
        Returns realized P&L from the closed portion
        """
        if position_id not in self.positions:
            logger.warning(f"Position {position_id} not found")
            return 0.0
        
        if exit_date is None:
            exit_date = datetime.now()
        
        position = self.positions[position_id]
        close_quantity = partial_quantity or position.quantity
        
        # Calculate multiplier for options vs stocks
        multiplier = 100 if 'option' in position.position_type.lower() else 1
        
        # Calculate realized P&L
        if position.position_type.lower() in ['put', 'call']:
            # For sold options, profit when price decreases
            realized_pnl = (position.entry_price - exit_price) * close_quantity * multiplier
        else:
            # For stocks, profit when price increases
            realized_pnl = (exit_price - position.entry_price) * close_quantity * multiplier
        
        position.realized_pnl += realized_pnl
        
        # Update cash
        if position.position_type.lower() in ['put', 'call']:
            # Buying back short option
            self.current_capital -= exit_price * close_quantity * multiplier
        else:
            # Selling stock
            self.current_capital += exit_price * close_quantity * multiplier
        
        # Log the trade
        trade_record = {
            'timestamp': exit_date,
            'action': 'CLOSE',
            'position_id': position_id,
            'ticker': position.ticker,
            'type': position.position_type,
            'quantity': close_quantity,
            'price': exit_price,
            'realized_pnl': realized_pnl,
            'days_held': (exit_date - position.entry_date).days
        }
        self.trade_log.append(trade_record)
        
        # Handle partial vs full close
        if partial_quantity and partial_quantity < position.quantity:
            # Partial close - update position
            position.quantity -= close_quantity
            logger.info(f"Partially closed {close_quantity} of {position_id}: ${realized_pnl:.2f} P&L")
        else:
            # Full close - move to closed positions
            self.closed_positions.append(position)
            del self.positions[position_id]
            logger.info(f"Closed position {position_id}: ${realized_pnl:.2f} P&L")
        
        return realized_pnl
    
    def update_position_prices(self, price_updates: Dict[str, float], 
                             greeks_updates: Dict[str, Dict] = None):
        """
        Update current prices and Greeks for all positions
        price_updates: {ticker: current_price}
        greeks_updates: {ticker: {delta: x, gamma: y, ...}}
        """
        greeks_updates = greeks_updates or {}
        
        for position_id, position in self.positions.items():
            if position.ticker in price_updates:
                position.current_price = price_updates[position.ticker]
                
                # Update days held
                position.days_held = (datetime.now() - position.entry_date).days
                
                # Calculate unrealized P&L
                multiplier = 100 if 'option' in position.position_type.lower() else 1
                
                if position.position_type.lower() in ['put', 'call']:
                    # For sold options, profit when price decreases
                    position.unrealized_pnl = (position.entry_price - position.current_price) * position.quantity * multiplier
                else:
                    # For stocks, profit when price increases  
                    position.unrealized_pnl = (position.current_price - position.entry_price) * position.quantity * multiplier
                
                # Update Greeks if provided
                if position.ticker in greeks_updates:
                    greeks = greeks_updates[position.ticker]
                    position.delta = greeks.get('delta', position.delta)
                    position.gamma = greeks.get('gamma', position.gamma)
                    position.theta = greeks.get('theta', position.theta)
                    position.vega = greeks.get('vega', position.vega)
    
    def calculate_portfolio_metrics(self) -> PortfolioMetrics:
        """Calculate current portfolio-level metrics"""
        now = datetime.now()
        
        # Calculate position values
        equity_value = 0.0
        options_value = 0.0
        total_unrealized_pnl = 0.0
        total_realized_pnl = sum(pos.realized_pnl for pos in self.closed_positions)
        
        # Portfolio Greeks
        portfolio_delta = 0.0
        portfolio_gamma = 0.0
        portfolio_theta = 0.0
        portfolio_vega = 0.0
        
        for position in self.positions.values():
            position_value = position.current_price * position.quantity
            
            if position.position_type.lower() == 'stock':
                equity_value += position_value
            else:
                options_value += position_value * 100  # Options multiplier
            
            total_unrealized_pnl += position.unrealized_pnl
            total_realized_pnl += position.realized_pnl
            
            # Aggregate Greeks (weighted by position size)
            portfolio_delta += position.delta * position.quantity
            portfolio_gamma += position.gamma * position.quantity
            portfolio_theta += position.theta * position.quantity
            portfolio_vega += position.vega * position.quantity
        
        total_value = self.current_capital + equity_value + options_value
        total_pnl = total_realized_pnl + total_unrealized_pnl
        
        # Calculate daily P&L
        daily_pnl = 0.0
        if self.daily_metrics:
            yesterday_value = self.daily_metrics[-1].total_value
            daily_pnl = total_value - yesterday_value
        
        # Calculate performance metrics
        sharpe_ratio = self._calculate_sharpe_ratio()
        max_drawdown = self._calculate_max_drawdown()
        win_rate, profit_factor = self._calculate_trade_metrics()
        
        metrics = PortfolioMetrics(
            timestamp=now,
            total_value=total_value,
            cash=self.current_capital,
            equity_value=equity_value,
            options_value=options_value,
            total_pnl=total_pnl,
            daily_pnl=daily_pnl,
            unrealized_pnl=total_unrealized_pnl,
            realized_pnl=total_realized_pnl,
            portfolio_delta=portfolio_delta,
            portfolio_gamma=portfolio_gamma,
            portfolio_theta=portfolio_theta,
            portfolio_vega=portfolio_vega,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(self.trade_log),
            winning_trades=len([t for t in self.trade_log if t.get('realized_pnl', 0) > 0])
        )
        
        # Store for history
        self.intraday_metrics.append(metrics)
        
        return metrics
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate rolling Sharpe ratio"""
        if len(self.returns_buffer) < 30:
            return 0.0
        
        returns = np.array(self.returns_buffer)
        if returns.std() == 0:
            return 0.0
        
        # Assume 5% risk-free rate
        excess_returns = returns - (0.05 / 252)
        return np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from peak"""
        if len(self.daily_metrics) < 2:
            return 0.0
        
        values = [m.total_value for m in self.daily_metrics]
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        return np.min(drawdown)
    
    def _calculate_trade_metrics(self) -> Tuple[float, float]:
        """Calculate win rate and profit factor"""
        closed_trades = [t for t in self.trade_log if t.get('realized_pnl') is not None]
        
        if not closed_trades:
            return 0.0, 0.0
        
        winning_trades = [t for t in closed_trades if t['realized_pnl'] > 0]
        losing_trades = [t for t in closed_trades if t['realized_pnl'] < 0]
        
        win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0.0
        
        total_wins = sum(t['realized_pnl'] for t in winning_trades)
        total_losses = abs(sum(t['realized_pnl'] for t in losing_trades))
        
        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0
        
        return win_rate, profit_factor
    
    def check_risk_alerts(self, current_metrics: PortfolioMetrics) -> List[Dict]:
        """Check for risk limit breaches and generate alerts"""
        alerts = []
        
        # Portfolio delta check
        portfolio_delta_pct = abs(current_metrics.portfolio_delta) / (current_metrics.total_value / 100)
        if portfolio_delta_pct > self.risk_limits['max_portfolio_delta']:
            alerts.append({
                'type': 'RISK_BREACH',
                'level': 'WARNING',
                'message': f"Portfolio delta {portfolio_delta_pct:.1%} exceeds limit {self.risk_limits['max_portfolio_delta']:.1%}",
                'metric': 'portfolio_delta',
                'current': portfolio_delta_pct,
                'limit': self.risk_limits['max_portfolio_delta']
            })
        
        # Maximum drawdown check
        if current_metrics.max_drawdown < -self.risk_limits['max_drawdown']:
            alerts.append({
                'type': 'RISK_BREACH',
                'level': 'CRITICAL',
                'message': f"Maximum drawdown {current_metrics.max_drawdown:.1%} exceeds limit {self.risk_limits['max_drawdown']:.1%}",
                'metric': 'max_drawdown',
                'current': current_metrics.max_drawdown,
                'limit': -self.risk_limits['max_drawdown']
            })
        
        # Single position concentration check
        for position in self.positions.values():
            position_value = abs(position.current_price * position.quantity * 
                               (100 if 'option' in position.position_type.lower() else 1))
            position_pct = position_value / current_metrics.total_value
            
            if position_pct > self.risk_limits['max_single_position']:
                alerts.append({
                    'type': 'CONCENTRATION_RISK',
                    'level': 'WARNING',
                    'message': f"Position {position.ticker} represents {position_pct:.1%} of portfolio (limit: {self.risk_limits['max_single_position']:.1%})",
                    'ticker': position.ticker,
                    'current': position_pct,
                    'limit': self.risk_limits['max_single_position']
                })
        
        return alerts
    
    def generate_daily_report(self) -> Dict:
        """Generate end-of-day performance report"""
        current_metrics = self.calculate_portfolio_metrics()
        
        # Add to daily history
        self.daily_metrics.append(current_metrics)
        
        # Calculate daily return
        daily_return = current_metrics.daily_pnl / (current_metrics.total_value - current_metrics.daily_pnl)
        self.returns_buffer.append(daily_return)
        
        report = {
            'date': current_metrics.timestamp.strftime('%Y-%m-%d'),
            'portfolio_value': current_metrics.total_value,
            'daily_pnl': current_metrics.daily_pnl,
            'daily_return': daily_return,
            'total_return': (current_metrics.total_value / self.initial_capital) - 1,
            'sharpe_ratio': current_metrics.sharpe_ratio,
            'max_drawdown': current_metrics.max_drawdown,
            'win_rate': current_metrics.win_rate,
            'open_positions': len(self.positions),
            'portfolio_delta': current_metrics.portfolio_delta,
            'risk_alerts': self.check_risk_alerts(current_metrics)
        }
        
        return report
    
    def save_performance_data(self, filepath: Path = None):
        """Save performance tracking data to file"""
        if filepath is None:
            filepath = Path(f"data/performance/performance_{datetime.now().strftime('%Y%m%d')}.json")
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'positions': {
                pid: {
                    'ticker': pos.ticker,
                    'type': pos.position_type,
                    'quantity': pos.quantity,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'entry_date': pos.entry_date.isoformat(),
                    'unrealized_pnl': pos.unrealized_pnl,
                    'realized_pnl': pos.realized_pnl
                } for pid, pos in self.positions.items()
            },
            'daily_metrics': [
                {
                    'timestamp': m.timestamp.isoformat(),
                    'total_value': m.total_value,
                    'total_pnl': m.total_pnl,
                    'sharpe_ratio': m.sharpe_ratio,
                    'max_drawdown': m.max_drawdown,
                    'portfolio_delta': m.portfolio_delta
                } for m in self.daily_metrics
            ],
            'trade_log': self.trade_log
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Performance data saved to {filepath}")
    
    def load_performance_data(self, filepath: Path):
        """Load performance tracking data from file"""
        if not filepath.exists():
            logger.warning(f"Performance data file not found: {filepath}")
            return
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.initial_capital = data.get('initial_capital', self.initial_capital)
        self.current_capital = data.get('current_capital', self.current_capital)
        self.trade_log = data.get('trade_log', [])
        
        # Restore positions
        for pid, pos_data in data.get('positions', {}).items():
            position = PositionMetrics(
                ticker=pos_data['ticker'],
                position_type=pos_data['type'],
                quantity=pos_data['quantity'],
                entry_price=pos_data['entry_price'],
                current_price=pos_data['current_price'],
                entry_date=datetime.fromisoformat(pos_data['entry_date']),
                days_held=0,
                unrealized_pnl=pos_data['unrealized_pnl'],
                realized_pnl=pos_data['realized_pnl']
            )
            self.positions[pid] = position
        
        logger.info(f"Performance data loaded from {filepath}")
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics for current session"""
        current_metrics = self.calculate_portfolio_metrics()
        
        return {
            'portfolio_value': current_metrics.total_value,
            'total_return': (current_metrics.total_value / self.initial_capital) - 1,
            'unrealized_pnl': current_metrics.unrealized_pnl,
            'realized_pnl': current_metrics.realized_pnl,
            'sharpe_ratio': current_metrics.sharpe_ratio,
            'max_drawdown': current_metrics.max_drawdown,
            'open_positions': len(self.positions),
            'total_trades': len(self.trade_log),
            'win_rate': current_metrics.win_rate,
            'portfolio_delta': current_metrics.portfolio_delta,
            'last_updated': current_metrics.timestamp.isoformat()
        }

# Global instance for easy access
_performance_tracker = None

def get_performance_tracker(initial_capital: float = None) -> PerformanceTracker:
    """Get or create global performance tracker instance"""
    global _performance_tracker
    if _performance_tracker is None:
        _performance_tracker = PerformanceTracker(initial_capital or 100000.0)
    return _performance_tracker

if __name__ == "__main__":
    # Example usage
    tracker = PerformanceTracker(100000.0)
    
    # Add a position
    pos_id = tracker.add_position('U', 'put', -5, 2.50, greeks={'delta': -0.25})
    
    # Update prices
    tracker.update_position_prices({'U': 24.10})
    
    # Get current metrics
    metrics = tracker.calculate_portfolio_metrics()
    print(f"Portfolio Value: ${metrics.total_value:,.2f}")
    print(f"Total P&L: ${metrics.total_pnl:.2f}")
    print(f"Portfolio Delta: {metrics.portfolio_delta:.2f}")
    
    # Generate daily report
    report = tracker.generate_daily_report()
    print(json.dumps(report, indent=2, default=str))
