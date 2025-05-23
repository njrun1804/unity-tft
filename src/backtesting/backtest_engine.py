#!/usr/bin/env python
"""
backtest_engine.py - Historical backtesting for wheel strategy
Simulates trading performance using historical data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import json
import logging
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import seaborn as sns

from src.strategy.wheel_strategy_production import WheelStrategyProduction, Position, OptionContract
from src.features.feature_engineer import FeatureEngineer
from src.models.tft_inference import get_tft_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters"""
    start_date: str
    end_date: str
    initial_capital: float = 100000.0
    commission_per_contract: float = 1.0
    commission_per_share: float = 0.005
    margin_requirement: float = 0.20  # 20% margin for cash-secured puts
    risk_free_rate: float = 0.05
    benchmark_ticker: str = "SPY"

@dataclass
class Trade:
    """Individual trade record"""
    timestamp: datetime
    action: str  # 'BUY', 'SELL', 'ASSIGN', 'EXPIRE'
    contract_type: str  # 'PUT', 'CALL', 'STOCK'
    ticker: str
    quantity: int
    price: float
    strike: Optional[float] = None
    expiry: Optional[str] = None
    commission: float = 0.0
    pnl: float = 0.0
    
@dataclass
class BacktestResults:
    """Comprehensive backtest results"""
    trades: List[Trade] = field(default_factory=list)
    daily_returns: pd.Series = field(default_factory=pd.Series)
    portfolio_values: pd.Series = field(default_factory=pd.Series)
    drawdowns: pd.Series = field(default_factory=pd.Series)
    
    # Performance metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    # Trading statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_trade_duration: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0

class BacktestEngine:
    """
    Historical backtesting engine for wheel strategy
    Simulates trades using historical market data
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.portfolio_value = config.initial_capital
        self.cash = config.initial_capital
        self.positions = {}
        self.trades = []
        self.daily_values = []
        
        # Initialize components
        self.feature_engineer = FeatureEngineer()
        
        # Try to load TFT model if available
        model_path = Path("models/tft/best.ckpt")
        if model_path.exists():
            self.tft_model = get_tft_model(model_path)
        else:
            logger.warning("TFT model not found, using dummy predictions")
            self.tft_model = None
            
        logger.info(f"Backtest initialized: {config.start_date} to {config.end_date}")
        logger.info(f"Initial capital: ${config.initial_capital:,.2f}")
    
    def load_historical_data(self, ticker: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load historical stock and options data
        Returns: (stock_data, options_data)
        """
        # Try to load from existing data files
        stock_file = Path(f"data/{ticker}_historical_stock.parquet")
        options_file = Path(f"data/{ticker}_historical_options.parquet")
        
        if stock_file.exists() and options_file.exists():
            stock_data = pd.read_parquet(stock_file)
            options_data = pd.read_parquet(options_file)
            logger.info(f"Loaded historical data for {ticker}")
        else:
            # Generate synthetic data for demo purposes
            logger.warning(f"Historical data files not found, generating synthetic data")
            stock_data, options_data = self._generate_synthetic_data(ticker)
        
        # Filter by date range
        start_dt = pd.to_datetime(self.config.start_date)
        end_dt = pd.to_datetime(self.config.end_date)
        
        stock_data = stock_data[
            (stock_data['date'] >= start_dt) & 
            (stock_data['date'] <= end_dt)
        ].copy()
        
        options_data = options_data[
            (options_data['date'] >= start_dt) & 
            (options_data['date'] <= end_dt)
        ].copy()
        
        return stock_data, options_data
    
    def _generate_synthetic_data(self, ticker: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate synthetic market data for backtesting"""
        start_dt = pd.to_datetime(self.config.start_date)
        end_dt = pd.to_datetime(self.config.end_date)
        dates = pd.date_range(start_dt, end_dt, freq='D')
        
        # Generate stock price data with realistic volatility
        np.random.seed(42)  # For reproducible results
        n_days = len(dates)
        
        # Stock price simulation (geometric Brownian motion)
        initial_price = 25.0
        mu = 0.08 / 252  # 8% annual return
        sigma = 0.25 / np.sqrt(252)  # 25% annual volatility
        
        price_changes = np.random.normal(mu, sigma, n_days)
        prices = [initial_price]
        for change in price_changes[1:]:
            prices.append(prices[-1] * (1 + change))
        
        stock_data = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.02))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.02))) for p in prices],
            'close': prices,
            'volume': np.random.randint(500000, 2000000, n_days)
        })
        
        # Generate options data
        options_data = self._generate_synthetic_options(stock_data)
        
        logger.info(f"Generated synthetic data: {len(stock_data)} days, {len(options_data)} option quotes")
        return stock_data, options_data
    
    def _generate_synthetic_options(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic options chain data"""
        options_rows = []
        
        for _, row in stock_data.iterrows():
            spot = row['close']
            date = row['date']
            
            # Generate options for multiple expiration dates
            for days_ahead in [7, 14, 21, 30, 45, 60]:
                expiry = date + timedelta(days=days_ahead)
                
                # Generate strikes around current price
                strikes = [spot * (1 + i * 0.05) for i in range(-4, 5)]
                
                for strike in strikes:
                    for option_type in ['call', 'put']:
                        # Simple Black-Scholes approximation for demo
                        moneyness = strike / spot
                        time_to_expiry = days_ahead / 365.0
                        
                        # Simplified IV calculation
                        iv = 0.20 + 0.10 * abs(moneyness - 1.0)
                        
                        # Simplified Greeks calculation
                        if option_type == 'call':
                            delta = 0.5 if moneyness == 1.0 else (0.7 if moneyness < 1.0 else 0.3)
                        else:
                            delta = -0.5 if moneyness == 1.0 else (-0.3 if moneyness < 1.0 else -0.7)
                        
                        # Approximate option price
                        intrinsic = max(0, spot - strike) if option_type == 'call' else max(0, strike - spot)
                        time_value = iv * spot * np.sqrt(time_to_expiry)
                        price = intrinsic + time_value
                        
                        options_rows.append({
                            'date': date,
                            'ticker': stock_data.iloc[0].get('ticker', 'U'),
                            'expiry': expiry.strftime('%Y-%m-%d'),
                            'strike': strike,
                            'option_type': option_type,
                            'bid': price * 0.98,
                            'ask': price * 1.02,
                            'mid': price,
                            'iv': iv,
                            'delta': delta,
                            'gamma': 0.05,
                            'theta': -0.02,
                            'vega': 0.10,
                            'volume': np.random.randint(0, 1000),
                            'open_interest': np.random.randint(0, 5000)
                        })
        
        return pd.DataFrame(options_rows)
    
    def run_backtest(self, ticker: str = "U") -> BacktestResults:
        """
        Execute complete backtest for the specified ticker
        """
        logger.info(f"Starting backtest for {ticker}")
        
        # Load historical data
        stock_data, options_data = self.load_historical_data(ticker)
        
        if stock_data.empty:
            raise ValueError(f"No historical data available for {ticker}")
        
        # Initialize strategy
        position = Position()
        
        # Load configuration
        with open('positions.json', 'r') as f:
            strategy_config = json.load(f)
        
        strategy = WheelStrategyProduction(position, strategy_config)
        
        # Group data by date for day-by-day simulation
        dates = sorted(stock_data['date'].unique())
        
        for date in dates:
            self._simulate_trading_day(date, ticker, stock_data, options_data, strategy)
        
        # Calculate final results
        results = self._calculate_results()
        
        logger.info("Backtest completed")
        logger.info(f"Total return: {results.total_return:.2%}")
        logger.info(f"Sharpe ratio: {results.sharpe_ratio:.2f}")
        logger.info(f"Max drawdown: {results.max_drawdown:.2%}")
        logger.info(f"Win rate: {results.win_rate:.2%}")
        
        return results
    
    def _simulate_trading_day(self, date: pd.Timestamp, ticker: str, 
                            stock_data: pd.DataFrame, options_data: pd.DataFrame,
                            strategy: WheelStrategyProduction):
        """Simulate trading for a single day"""
        
        # Get data for this day
        day_stock = stock_data[stock_data['date'] == date].iloc[0]
        day_options = options_data[options_data['date'] == date]
        
        if day_options.empty:
            self.daily_values.append(self.portfolio_value)
            return
        
        # Create stock data dictionary
        stock_dict = {
            'day': {
                'c': day_stock['close'],
                'h': day_stock['high'], 
                'l': day_stock['low'],
                'o': day_stock['open'],
                'v': day_stock['volume']
            }
        }
        
        # Convert options data to OptionContract objects
        option_contracts = []
        for _, opt in day_options.iterrows():
            contract = OptionContract(
                ticker=opt['ticker'],
                strike=opt['strike'],
                expiry=opt['expiry'],
                contract_type=opt['option_type'],
                bid=opt['bid'],
                ask=opt['ask'],
                mid=opt['mid'],
                iv=opt['iv'],
                delta=opt['delta'],
                gamma=opt['gamma'],
                theta=opt['theta'],
                vega=opt['vega'],
                open_interest=opt['open_interest'],
                volume=opt['volume'],
                days_to_expiry=self._calculate_dte(opt['expiry'], date),
                underlying_price=day_stock['close']
            )
            option_contracts.append(contract)
        
        # Engineer features
        features = self._create_features(stock_dict, day_options)
        
        # Generate TFT predictions if model available
        if self.tft_model:
            # Create feature dataframe for TFT
            feature_df = self._create_tft_features(stock_dict, date)
            tft_predictions = self.tft_model.predict(feature_df)
            features.update(tft_predictions)
        else:
            # Use dummy predictions
            features.update({
                'p_up_1d': 0.55,
                'prediction_std': 0.08,
                'confidence_interval_lower': 0.47,
                'confidence_interval_upper': 0.67
            })
        
        # Generate trading signals
        signals = strategy.generate_signals(stock_dict, option_contracts, features)
        
        # Execute trades based on signals
        self._execute_signals(signals, date, day_stock['close'])
        
        # Update portfolio value
        self._update_portfolio_value(date, day_stock['close'], option_contracts)
        self.daily_values.append(self.portfolio_value)
    
    def _calculate_dte(self, expiry_str: str, current_date: pd.Timestamp) -> int:
        """Calculate days to expiration"""
        expiry = pd.to_datetime(expiry_str)
        return max(0, (expiry - current_date).days)
    
    def _create_features(self, stock_data: Dict, options_data: pd.DataFrame) -> Dict:
        """Create feature dictionary for strategy"""
        spot = stock_data['day']['c']
        
        # Calculate basic features
        features = {
            'spot_price': spot,
            'realized_vol': 0.25,  # Placeholder
            'iv_rank': options_data['iv'].rank(pct=True).mean() if not options_data.empty else 0.5,
            'market_regime': 'normal'
        }
        
        return features
    
    def _create_tft_features(self, stock_data: Dict, date: pd.Timestamp) -> pd.DataFrame:
        """Create feature dataframe compatible with TFT model"""
        # This is a simplified version - in practice you'd need proper feature engineering
        return pd.DataFrame({
            'date': [date],
            'ticker': ['U'],
            'close': [stock_data['day']['c']],
            'volume': [stock_data['day']['v']],
            'time_idx': [1]  # Simplified
        })
    
    def _execute_signals(self, signals: Dict, date: pd.Timestamp, spot_price: float):
        """Execute trading signals and record trades"""
        if not signals or signals.get('action') == 'HOLD':
            return
        
        action = signals.get('action', 'HOLD')
        
        if action in ['SELL_PUT', 'SELL_CALL']:
            self._execute_option_sell(signals, date, spot_price)
        elif action in ['BUY_PUT', 'BUY_CALL']:
            self._execute_option_buy(signals, date, spot_price)
        elif action == 'BUY_STOCK':
            self._execute_stock_buy(signals, date, spot_price)
        elif action == 'SELL_STOCK':
            self._execute_stock_sell(signals, date, spot_price)
    
    def _execute_option_sell(self, signals: Dict, date: pd.Timestamp, spot_price: float):
        """Execute option selling trade"""
        premium = signals.get('premium', 1.0)
        quantity = signals.get('quantity', 1)
        commission = quantity * self.config.commission_per_contract
        
        # Credit received (minus commission)
        net_credit = (premium * quantity * 100) - commission
        self.cash += net_credit
        
        trade = Trade(
            timestamp=date,
            action='SELL',
            contract_type=signals.get('contract_type', 'PUT'),
            ticker=signals.get('ticker', 'U'),
            quantity=quantity,
            price=premium,
            strike=signals.get('strike'),
            expiry=signals.get('expiry'),
            commission=commission,
            pnl=net_credit
        )
        
        self.trades.append(trade)
        logger.debug(f"Executed {trade.action} {trade.contract_type}: +${net_credit:.2f}")
    
    def _execute_option_buy(self, signals: Dict, date: pd.Timestamp, spot_price: float):
        """Execute option buying trade"""
        premium = signals.get('premium', 1.0)
        quantity = signals.get('quantity', 1)
        commission = quantity * self.config.commission_per_contract
        
        # Debit paid (plus commission)
        net_debit = (premium * quantity * 100) + commission
        
        if self.cash >= net_debit:
            self.cash -= net_debit
            
            trade = Trade(
                timestamp=date,
                action='BUY',
                contract_type=signals.get('contract_type', 'PUT'),
                ticker=signals.get('ticker', 'U'),
                quantity=quantity,
                price=premium,
                strike=signals.get('strike'),
                expiry=signals.get('expiry'),
                commission=commission,
                pnl=-net_debit
            )
            
            self.trades.append(trade)
            logger.debug(f"Executed {trade.action} {trade.contract_type}: -${net_debit:.2f}")
    
    def _execute_stock_buy(self, signals: Dict, date: pd.Timestamp, spot_price: float):
        """Execute stock purchase"""
        quantity = signals.get('quantity', 100)
        commission = quantity * self.config.commission_per_share
        total_cost = (spot_price * quantity) + commission
        
        if self.cash >= total_cost:
            self.cash -= total_cost
            
            trade = Trade(
                timestamp=date,
                action='BUY',
                contract_type='STOCK',
                ticker=signals.get('ticker', 'U'),
                quantity=quantity,
                price=spot_price,
                commission=commission,
                pnl=-total_cost
            )
            
            self.trades.append(trade)
    
    def _execute_stock_sell(self, signals: Dict, date: pd.Timestamp, spot_price: float):
        """Execute stock sale"""
        quantity = signals.get('quantity', 100)
        commission = quantity * self.config.commission_per_share
        total_proceeds = (spot_price * quantity) - commission
        
        self.cash += total_proceeds
        
        trade = Trade(
            timestamp=date,
            action='SELL',
            contract_type='STOCK',
            ticker=signals.get('ticker', 'U'),
            quantity=quantity,
            price=spot_price,
            commission=commission,
            pnl=total_proceeds
        )
        
        self.trades.append(trade)
    
    def _update_portfolio_value(self, date: pd.Timestamp, spot_price: float, 
                              option_contracts: List[OptionContract]):
        """Update total portfolio value"""
        # Start with cash
        total_value = self.cash
        
        # Add stock positions (simplified - assumes no stock positions for wheel strategy)
        # In a full implementation, you'd track stock positions from assignments
        
        # Add/subtract option positions
        # This is simplified - in practice you'd track open option positions
        # and mark them to market
        
        self.portfolio_value = total_value
    
    def _calculate_results(self) -> BacktestResults:
        """Calculate comprehensive backtest results"""
        results = BacktestResults()
        results.trades = self.trades
        
        if not self.daily_values:
            return results
        
        # Convert daily values to pandas series
        portfolio_series = pd.Series(self.daily_values)
        results.portfolio_values = portfolio_series
        
        # Calculate returns
        daily_returns = portfolio_series.pct_change().dropna()
        results.daily_returns = daily_returns
        
        # Calculate drawdowns
        peak = portfolio_series.expanding().max()
        drawdown = (portfolio_series - peak) / peak
        results.drawdowns = drawdown
        
        # Performance metrics
        if len(portfolio_series) > 1:
            results.total_return = (portfolio_series.iloc[-1] / portfolio_series.iloc[0]) - 1
            
            # Annualized return
            days = len(portfolio_series)
            years = days / 252.0  # Trading days per year
            results.annualized_return = (1 + results.total_return) ** (1/years) - 1
            
            # Sharpe ratio
            if daily_returns.std() > 0:
                excess_returns = daily_returns - (self.config.risk_free_rate / 252)
                results.sharpe_ratio = excess_returns.mean() / daily_returns.std() * np.sqrt(252)
            
            # Maximum drawdown
            results.max_drawdown = drawdown.min()
        
        # Trading statistics
        results.total_trades = len(self.trades)
        
        if self.trades:
            winning_trades = [t for t in self.trades if t.pnl > 0]
            losing_trades = [t for t in self.trades if t.pnl < 0]
            
            results.winning_trades = len(winning_trades)
            results.losing_trades = len(losing_trades)
            
            if results.total_trades > 0:
                results.win_rate = results.winning_trades / results.total_trades
            
            if winning_trades:
                results.avg_win = np.mean([t.pnl for t in winning_trades])
            
            if losing_trades:
                results.avg_loss = np.mean([t.pnl for t in losing_trades])
                
            # Profit factor
            total_wins = sum(t.pnl for t in winning_trades)
            total_losses = abs(sum(t.pnl for t in losing_trades))
            
            if total_losses > 0:
                results.profit_factor = total_wins / total_losses
        
        return results
    
    def generate_report(self, results: BacktestResults, output_dir: Path = Path("outputs/backtests")):
        """Generate comprehensive backtest report"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"backtest_report_{timestamp}.html"
        
        # Create visualizations
        self._create_performance_plots(results, output_dir, timestamp)
        
        # Generate HTML report
        html_content = self._generate_html_report(results, timestamp)
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Backtest report saved: {report_path}")
        return report_path
    
    def _create_performance_plots(self, results: BacktestResults, output_dir: Path, timestamp: str):
        """Create performance visualization plots"""
        plt.style.use('seaborn-v0_8')
        
        # Portfolio value over time
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Portfolio Value
        results.portfolio_values.plot(ax=ax1, title='Portfolio Value Over Time')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True)
        
        # 2. Drawdown
        results.drawdowns.plot(ax=ax2, title='Drawdown', color='red')
        ax2.fill_between(results.drawdowns.index, results.drawdowns, 0, alpha=0.3, color='red')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True)
        
        # 3. Daily Returns Distribution
        ax3.hist(results.daily_returns.dropna(), bins=50, alpha=0.7, edgecolor='black')
        ax3.set_title('Daily Returns Distribution')
        ax3.set_xlabel('Daily Return')
        ax3.set_ylabel('Frequency')
        ax3.grid(True)
        
        # 4. Rolling Sharpe Ratio
        rolling_sharpe = results.daily_returns.rolling(60).mean() / results.daily_returns.rolling(60).std() * np.sqrt(252)
        rolling_sharpe.plot(ax=ax4, title='Rolling 60-Day Sharpe Ratio')
        ax4.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Target Sharpe')
        ax4.set_ylabel('Sharpe Ratio')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"performance_charts_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_html_report(self, results: BacktestResults, timestamp: str) -> str:
        """Generate HTML report content"""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Unity Wheel Strategy Backtest Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
        .metric-label {{ color: #7f8c8d; font-size: 14px; }}
        .section {{ margin: 30px 0; }}
        .positive {{ color: #27ae60; }}
        .negative {{ color: #e74c3c; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Unity Wheel Strategy Backtest Report</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Period: {self.config.start_date} to {self.config.end_date}</p>
    </div>
    
    <div class="section">
        <h2>Performance Summary</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value {'positive' if results.total_return > 0 else 'negative'}">
                    {results.total_return:.2%}
                </div>
                <div class="metric-label">Total Return</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'positive' if results.annualized_return > 0 else 'negative'}">
                    {results.annualized_return:.2%}
                </div>
                <div class="metric-label">Annualized Return</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'positive' if results.sharpe_ratio > 1 else 'negative'}">
                    {results.sharpe_ratio:.2f}
                </div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
            <div class="metric-card">
                <div class="metric-value negative">
                    {results.max_drawdown:.2%}
                </div>
                <div class="metric-label">Max Drawdown</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">
                    {results.win_rate:.2%}
                </div>
                <div class="metric-label">Win Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">
                    {results.profit_factor:.2f}
                </div>
                <div class="metric-label">Profit Factor</div>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>Trading Statistics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Trades</td><td>{results.total_trades}</td></tr>
            <tr><td>Winning Trades</td><td>{results.winning_trades}</td></tr>
            <tr><td>Losing Trades</td><td>{results.losing_trades}</td></tr>
            <tr><td>Average Win</td><td>${results.avg_win:.2f}</td></tr>
            <tr><td>Average Loss</td><td>${results.avg_loss:.2f}</td></tr>
            <tr><td>Initial Capital</td><td>${self.config.initial_capital:,.2f}</td></tr>
            <tr><td>Final Value</td><td>${results.portfolio_values.iloc[-1] if len(results.portfolio_values) > 0 else 0:,.2f}</td></tr>
        </table>
    </div>
    
    <div class="section">
        <h2>Performance Charts</h2>
        <img src="performance_charts_{timestamp}.png" alt="Performance Charts" style="max-width: 100%; height: auto;">
    </div>
    
    <div class="section">
        <h2>Recent Trades</h2>
        <table>
            <tr>
                <th>Date</th>
                <th>Action</th>
                <th>Type</th>
                <th>Ticker</th>
                <th>Quantity</th>
                <th>Price</th>
                <th>Strike</th>
                <th>P&L</th>
            </tr>
            {''.join([f'''
            <tr>
                <td>{trade.timestamp.strftime('%Y-%m-%d')}</td>
                <td>{trade.action}</td>
                <td>{trade.contract_type}</td>
                <td>{trade.ticker}</td>
                <td>{trade.quantity}</td>
                <td>${trade.price:.2f}</td>
                <td>{trade.strike or 'N/A'}</td>
                <td class="{'positive' if trade.pnl > 0 else 'negative'}">${trade.pnl:.2f}</td>
            </tr>
            ''' for trade in results.trades[-20:]])}
        </table>
    </div>
</body>
</html>
        """

def main():
    parser = argparse.ArgumentParser(description='Run wheel strategy backtest')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    parser.add_argument('--ticker', type=str, default='U', help='Ticker to backtest')
    parser.add_argument('--commission', type=float, default=1.0, help='Commission per contract')
    parser.add_argument('--output', type=str, default='outputs/backtests', help='Output directory')
    
    args = parser.parse_args()
    
    # Create backtest configuration
    config = BacktestConfig(
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        commission_per_contract=args.commission
    )
    
    # Run backtest
    engine = BacktestEngine(config)
    results = engine.run_backtest(args.ticker)
    
    # Generate report
    output_dir = Path(args.output)
    report_path = engine.generate_report(results, output_dir)
    
    print(f"\n✅ Backtest completed!")
    print(f"📊 Report saved: {report_path}")
    print(f"📈 Total Return: {results.total_return:.2%}")
    print(f"📉 Max Drawdown: {results.max_drawdown:.2%}")
    print(f"🎯 Sharpe Ratio: {results.sharpe_ratio:.2f}")

if __name__ == '__main__':
    main()
