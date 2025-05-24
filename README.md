# Unity Price Trading System

A complete production-ready algorithmic trading pipeline that integrates **Temporal Fusion Transformer (TFT)** forecasting with **mathematical wheel options strategies**, real-time data feeds, and comprehensive risk management.

## 🏗️ System Architecture

### Core Components

1. **Data Pipeline** - Real-time Polygon.io feeds with intelligent rate limiting
2. **TFT Prediction Engine** - Temporal Fusion Transformer with quantile forecasting and confidence intervals
3. **Wheel Strategy Engine** - Mathematical options strategy with Sharpe ratio optimization
4. **Risk Management** - Portfolio Greeks tracking, position sizing, and Kelly criterion
5. **Execution Pipeline** - Automated signal generation and trade orchestration
6. **Monitoring & Alerts** - Real-time performance tracking and notification systems

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           UNITY TRADING SYSTEM                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   DATA      │    │  FEATURES   │    │     TFT     │    │  STRATEGY   │  │
│  │  FETCHER    │───▶│ ENGINEER    │───▶│  INFERENCE  │───▶│   ENGINE    │  │
│  │             │    │             │    │             │    │             │  │
│  │ Polygon.io  │    │ Technical   │    │ Quantiles   │    │ Wheel Math  │  │
│  │ Rate Limited│    │ Greeks      │    │ Confidence  │    │ Risk Mgmt   │  │
│  │ Async       │    │ Microstruc  │    │ Error Est   │    │ Sharpe Opt  │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│         │                   │                   │                   │      │
│         │                   │                   │                   │      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │ PARQUET     │    │ FEATURE     │    │ PREDICTION  │    │ SIGNALS &   │  │
│  │ STORAGE     │    │ VECTORS     │    │ OUTPUTS     │    │ EXECUTION   │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│                                                                             │
│         ┌─────────────┐              ┌─────────────┐              ┌─────────────┐
│         │ MONITORING  │              │ ALERTING    │              │ BACKTEST    │
│         │ & TRACKING  │              │ SYSTEM      │              │ ENGINE      │
│         └─────────────┘              └─────────────┘              └─────────────┘
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

```bash
# Environment setup
python >= 3.9
pip install -r requirements.txt

# Environment variables
```bash
export POLYGON_API_KEY="iNPRlzOE1hupYJT0uepDbhRW0BrvWPkP"
export RUN_MODE="once"  # or "continuous"
```

### Basic Usage

#### 1. Production Pipeline (Automated)
```bash
# Single execution cycle
python -m src.pipelines.wheel_orchestrator

# Continuous trading (production)
export RUN_MODE=continuous
python -m src.pipelines.wheel_orchestrator
```

#### 2. Train TFT Model
```bash
# Train with hyperparameter optimization
python train_tft.py data/train.csv data/val.csv models/tft/ --optimize

# Single training run
python train_tft.py data/train.csv data/val.csv models/tft/
```

## 📊 System Components

### Data Pipeline (`src/data/`)

**Polygon Data Fetcher** - `polygon_fetcher.py`
- **Async rate-limited API calls** (5 requests/second default)
- **Real-time stock quotes** and **options chains**
- **Parquet-based storage** with automatic partitioning
- **Error handling** and retry logic

```python
from src.data.polygon_fetcher import PolygonDataFetcher

fetcher = PolygonDataFetcher(api_key="your_key")
# Fetches stock data + options for all tickers
data = await fetcher.fetch_all_data(["U", "AAPL"])
```

### Feature Engineering (`src/features/`)

**Feature Engineer** - `feature_engineer.py`
- **Technical indicators** (RSI, Bollinger Bands, MACD)
- **Options Greeks** calculations and fallbacks
- **Market microstructure** features
- **TFT-compatible** feature vectors

```python
from src.features.feature_engineer import FeatureEngineer

engineer = FeatureEngineer(lookback_windows=[10, 30, 60])
features = engineer.engineer_all_features(stock_data, minute_bars, option_chain)
```

### TFT Model (`src/models/`)

**TFT Inference Engine** - `tft_inference.py`
- **Quantile predictions** (P10, P50, P90)
- **Confidence intervals** and error estimates
- **Ensemble predictions** from multiple checkpoints
- **Production-optimized** inference

```python
from src.models.tft_inference import get_tft_model

model = get_tft_model(Path("models/tft/best.ckpt"))
predictions = model.predict(features_df)
# Returns: {'prediction': 0.65, 'std': 0.08, 'quantiles': {...}, 'confidence': [...]}
```

### Wheel Strategy (`src/strategy/`)

**Production Wheel Strategy** - `wheel_strategy_production.py`
- **Mathematical optimization** using Sharpe ratios
- **Greeks-based position sizing** with Kelly criterion
- **Risk management** and portfolio constraints
- **Regime detection** (bull/bear/sideways markets)

Key Features:
- **Put selling** in cash-secured mode
- **Covered call** writing on assigned shares
- **Dynamic position sizing** based on IV rank
- **Automatic rolling** of positions near expiry

```python
from src.strategy.wheel_strategy_production import WheelStrategyProduction

strategy = WheelStrategyProduction(position, config)
signals = strategy.generate_signals(stock_data, option_chain, features)
```

### Orchestration (`src/pipelines/`)

**Main Orchestrator** - `wheel_orchestrator.py`
- **End-to-end pipeline** coordination
- **Market hours detection** and scheduling
- **Error handling** and recovery
- **Production vs development** modes

### Monitoring (`src/monitoring/`)

**Performance Tracking** - `performance_tracker.py`
- **Real-time P&L** calculation
- **Sharpe ratio** and risk metrics
- **Trade execution** analysis
- **Alert generation** for significant events

## 🧮 Mathematical Models

### Sharpe Ratio Optimization

The system uses **ex-ante Sharpe ratio** calculations for position sizing:

```python
# From src/utils/position_recommender.py
def ex_ante_sharpe(inputs: AlphaInputs, time_scale: float) -> float:
    """
    Calculate expected Sharpe ratio using TFT predictions and market data
    """
    mu_combined = (inputs.mu_tft + inputs.mu_skew + inputs.mu_news) / 3
    risk_adjusted_return = mu_combined * time_scale
    volatility_adjusted = inputs.ewma_vol * np.sqrt(time_scale)
    return risk_adjusted_return / volatility_adjusted
```

### Kelly Criterion Position Sizing

Position sizes use **Kelly criterion** with safety limits:

```python
kelly_fraction = (probability * payoff - (1 - probability)) / payoff
position_size = base_size * min(kelly_fraction, max_kelly_limit)
```

### Options Pricing & Greeks

**Black-Scholes calculations** with fallbacks:
- **Delta** for directional exposure
- **Gamma** for convexity risk  
- **Theta** for time decay
- **Vega** for volatility sensitivity

## 📋 Configuration

### Main Config (`positions.json`)

Core trading parameters and rules:

```json
{
  "sizing": {
    "base_size": 100,
    "max_size": 500,
    "sharpe_target": 1.5,
    "wheel": {
      "min_premium_yield": 0.01,
      "min_DTE": 7,
      "max_DTE": 45,
      "capital_at_risk_limit": 0.25
    }
  },
  "meta": {
    "portfolio_delta_target": [0.60, 0.80],
    "certainty_ladder": [
      {"min": 0.80, "scale": 1.30},
      {"min": 0.60, "scale": 1.00},
      {"min": 0.40, "scale": 0.70}
    ]
  }
}
```

### Environment Variables

```bash
# Required
POLYGON_API_KEY=your_polygon_api_key

# Optional
RUN_MODE=once                    # once|continuous
LOG_LEVEL=INFO                   # DEBUG|INFO|WARNING|ERROR
RISK_LIMIT_OVERRIDE=false        # Override risk limits (dangerous)
```

## 🎯 Trading Strategy

### Wheel Strategy Logic

1. **Cash-Secured Puts**
   - Sell puts on high-IV rank stocks
   - Strike selection based on TFT downside probability
   - Position sizing via Kelly criterion

2. **Covered Calls** (if assigned)
   - Write calls on assigned shares
   - Strike selection balances premium vs. upside
   - Rolling management for ITM positions

3. **Risk Management**
   - Maximum 25% of capital at risk per position
   - Portfolio delta between 60-80%
   - Stop losses at 200% of premium received

### Signal Generation Flow

```python
# Core signal generation pipeline
def generate_trading_signals():
    # 1. Fetch real-time data
    data = await polygon_fetcher.fetch_all_data(tickers)
    
    # 2. Engineer features
    features = feature_engineer.engineer_all_features(data)
    
    # 3. Generate TFT predictions
    predictions = tft_model.predict(features)
    
    # 4. Calculate strategy signals
    signals = wheel_strategy.generate_signals(data, predictions)
    
    # 5. Apply risk management
    final_signals = risk_manager.validate_signals(signals)
    
    return final_signals
```

## 📈 Performance Monitoring

### Real-time Metrics

- **Portfolio Value** and daily P&L
- **Sharpe Ratio** (rolling 30/60/252 days)
- **Maximum Drawdown** and recovery periods
- **Win Rate** and average trade duration
- **Greeks Exposure** (delta, gamma, theta, vega)

### Alerting System

Automatic alerts for:
- **Position assignments** and expirations
- **Risk limit breaches** (delta, portfolio value)
- **Market regime changes** detected by TFT
- **System errors** and data feed issues

## 🔧 Development & Testing

### Running Tests

```bash
# Unit tests
pytest tests/ -v

# Strategy backtests
python -m src.backtesting.backtest_engine --start 2023-01-01 --end 2024-12-31

# Model validation
python tests/test_tft_accuracy.py
```

### Code Structure

```
src/
├── data/           # Data fetching and storage
├── features/       # Feature engineering
├── models/         # TFT inference engine
├── strategy/       # Wheel strategy logic
├── risk/           # Risk management
├── pipelines/      # Orchestration
├── monitoring/     # Performance tracking
├── utils/          # Utilities and helpers
└── decision/       # Strategy selection logic
```

## 🚨 Risk Disclaimers

⚠️ **Important Trading Warnings:**

1. **This is experimental software** - Use at your own risk
2. **Start with paper trading** - Validate all signals manually initially
3. **Monitor positions closely** - Automated systems can fail
4. **Capital at risk** - Options trading involves substantial risk of loss
5. **Past performance** ≠ Future results

### Safety Features

- **Position size limits** (configurable maximum exposure)
- **Delta limits** (portfolio-level Greeks constraints)
- **Kill switch** (emergency stop via environment variable)
- **Dry run mode** (signal generation without execution)

## 📚 Additional Resources

- **Training Data:** Historical price and options data via Polygon.io
- **Model Artifacts:** Saved in `models/tft/` directory
- **Logs:** Structured logging in `logs/` directory
- **Backtests:** Historical performance in `outputs/backtests/`

---

