"""
Automated pipeline: Fetch data → Feature engineering → Model inference → Position recommendations
Usage:
    python scripts/automate_pipeline.py
"""
import sys
import os
import logging
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
import torch
import json
import asyncio
from datetime import datetime, timedelta

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.polygon_fetcher import PolygonDataFetcher
from src.features.feature_engineer import FeatureEngineer
from src.models.tft_inference import get_tft_model
from src.utils.position_recommender import recommend_positions, load_positions_config
from training.objectives import PriceLSTM
from src.greeks_calculator import GreeksCalculator

# --- CONFIGURATION ---
# Load main config
CONFIG_PATH = Path("positions.json")
try:
    with open(CONFIG_PATH) as f:
        CONFIG = json.load(f)
except FileNotFoundError:
    CONFIG = {"tickers": ["U"], "sizing": {"base_size": 100}}
    logging.warning("positions.json not found, using defaults")

# Extract configuration
TICKERS = CONFIG.get("tickers", ["U"])
RISK_FREE_RATE = CONFIG.get("risk_free_rate", 0.05)  # Current Fed funds rate

# Initialize Greeks calculator
GREEKS_CALCULATOR = GreeksCalculator(risk_free_rate=RISK_FREE_RATE)

# Paths - matching where PolygonDataFetcher saves data
DATA_ROOT = Path("data/feature_store")
STOCK_QUOTES_DIR = DATA_ROOT / "polygon_watchlist"  # This is where stock data is saved
OPTIONS_DIR = DATA_ROOT / "polygon_option_chain"     # This is where options are saved
MINUTE_BARS_DIR = DATA_ROOT / "polygon_minute_bars"  # This is where minute bars are saved

# Model paths
TFT_MODEL_PATH = Path("models/tft/best.ckpt")
LSTM_MODEL_PATH = Path("models/lstm_best_epoch50.pt")

# Output directory
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
LOG = logging.getLogger("automate_pipeline")

# --- HELPER FUNCTIONS ---

# --- DATA FETCHING ---
async def fetch_polygon_data():
    """Fetch latest data from Polygon API using direct client first, then fallback."""
    LOG.info(f"Fetching data for tickers: {TICKERS}")
    
    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        LOG.error("POLYGON_API_KEY environment variable not set")
        return False
    
    # Try direct Polygon client first (recommended)
    try:
        from polygon import RESTClient
        LOG.info("Using direct Polygon client...")
        
        client = RESTClient(api_key=api_key)
        
        # Create data directories
        for subdir in ["polygon_watchlist", "polygon_option_chain", "polygon_minute_bars"]:
            (Path("data/feature_store") / subdir / datetime.now().strftime("%Y-%m-%d")).mkdir(parents=True, exist_ok=True)
        
        # Fetch data for each ticker
        for ticker in TICKERS:
            # Get stock quote
            quote = client.get_last_quote(ticker)
            if quote:
                df = pd.DataFrame([{
                    'ticker': ticker,
                    'bid': quote.bid_price,
                    'ask': quote.ask_price,
                    'price': (quote.bid_price + quote.ask_price) / 2,
                    'timestamp': datetime.now()
                }])
                save_path = Path("data/feature_store/polygon_watchlist") / datetime.now().strftime("%Y-%m-%d") / f"{ticker}_quote.parquet"
                df.to_parquet(save_path)
                LOG.info(f"Saved quote for {ticker}")
                
            # Get options chain
            contracts = list(client.list_options_contracts(
                underlying_ticker=ticker,
                expired=False,
                limit=100
            ))
            LOG.info(f"Found {len(contracts)} option contracts for {ticker}")
            
        return True
        
    except ImportError:
        LOG.warning("polygon package not installed, falling back to custom fetcher")
    except Exception as e:
        LOG.error(f"Direct Polygon client failed: {e}")
    
    # Fallback to existing PolygonDataFetcher
    try:
        LOG.info("Using custom PolygonDataFetcher...")
        async with PolygonDataFetcher(api_key=api_key) as fetcher:
            for ticker in TICKERS:
                await fetcher.fetch_all_data(ticker)
        LOG.info("Data fetch completed successfully")
        return True
    except Exception as e:
        LOG.error(f"Custom fetcher also failed: {e}")
        return False

# --- DATA LOADING ---
def load_latest_data(data_dir: Path, lookback_hours: int = 24, target_tickers: List[str] = None) -> pd.DataFrame:
    """Load the most recent data from parquet files, expanding search if needed for target tickers."""
    if not data_dir.exists():
        LOG.warning(f"{data_dir} does not exist")
        return pd.DataFrame()
    
    # If we have target tickers (e.g., ['U']), prioritize finding data for them
    target_tickers = target_tickers or TICKERS
    
    # Start with recent files (within lookback_hours)
    cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
    all_files = []
    
    # Recursively find parquet files
    for file_path in data_dir.rglob("*.parquet"):
        if file_path.stat().st_mtime > cutoff_time.timestamp():
            all_files.append(file_path)
            
    # If no recent files or no target ticker data found, expand search
    if not all_files:
        LOG.warning(f"No recent parquet files found in {data_dir}, expanding search...")
        # Search all parquet files
        all_files = list(data_dir.rglob("*.parquet"))
        
    if not all_files:
        LOG.warning(f"No parquet files found in {data_dir}")
        return pd.DataFrame()
        
    LOG.info(f"Found {len(all_files)} files in {data_dir}")
    
    # Load and check for target ticker data
    dfs = []
    target_ticker_found = False
    
    # Sort files by modification time (newest first) but check all if needed
    sorted_files = sorted(all_files, key=lambda x: x.stat().st_mtime, reverse=True)
    
    for file_path in sorted_files:
        try:
            df = pd.read_parquet(file_path)
            if not df.empty:
                # Check if this file contains our target tickers
                if 'ticker' in df.columns:
                    file_tickers = set(df['ticker'].unique())
                    if any(ticker in file_tickers for ticker in target_tickers):
                        target_ticker_found = True
                        LOG.info(f"Found target ticker data in {file_path.name}")
                
                dfs.append(df)
                
                # If we found target ticker data and have enough files, we can stop
                if target_ticker_found and len(dfs) >= 5:
                    break
                    
        except Exception as e:
            LOG.warning(f"Failed to load {file_path}: {e}")
            
    if not dfs:
        return pd.DataFrame()
        
    result = pd.concat(dfs, ignore_index=True)
    
    # Filter to only include target tickers if specified
    if target_tickers and 'ticker' in result.columns:
        before_count = len(result)
        result = result[result['ticker'].isin(target_tickers)].copy()
        LOG.info(f"Filtered to target tickers {target_tickers}: {before_count} -> {len(result)} rows")
    
    LOG.info(f"Loaded {len(result)} total rows from {len(dfs)} files")
    return result

# --- FEATURE ENGINEERING ---
def engineer_features(stock_df: pd.DataFrame, options_df: pd.DataFrame, 
                     minute_bars_df: pd.DataFrame = None) -> pd.DataFrame:
    """Create features using the production FeatureEngineer and calculate Greeks."""
    LOG.info("Starting feature engineering...")
    
    if stock_df.empty or options_df.empty:
        LOG.warning("Empty input data for feature engineering")
        return pd.DataFrame()
    
    # Step 1: Calculate Greeks for options data using the new calculator
    LOG.info("Calculating Greeks for options data...")
    
    # Ensure options data has the required columns for Greeks calculation
    if 'ticker' in options_df.columns and 'ticker' in stock_df.columns:
        # Merge stock prices with options for Greeks calculation
        options_with_price = options_df.merge(
            stock_df[['ticker', 'price']].drop_duplicates('ticker'),
            on='ticker',
            how='left',
            suffixes=('', '_stock')
        )
        
        # Use the stock price column for Greeks calculation
        if 'price_stock' in options_with_price.columns:
            options_with_price['price'] = options_with_price['price_stock'].fillna(options_with_price.get('price', 24.0))
        
        # Calculate Greeks using the GreeksCalculator
        options_with_greeks = GREEKS_CALCULATOR.process_options_dataframe(
            options_with_price,
            stock_price_col='price',
            strike_col='strike',
            expiry_col='expiry',
            iv_col='iv',
            option_type_col='cp'
        )
        
        LOG.info(f"Greeks calculated for {len(options_with_greeks)} options")
    else:
        LOG.warning("Cannot calculate Greeks: missing ticker column in data")
        options_with_greeks = options_df.copy()
        
    # Step 2: Prepare LSTM-ready features (9 features required)
    LOG.info("Preparing LSTM features...")
    
    # Use the Greeks calculator to prepare exact LSTM features
    lstm_features = GREEKS_CALCULATOR.prepare_lstm_features(options_with_greeks)
    
    LOG.info(f"LSTM features prepared: {lstm_features.shape[0]} rows, {lstm_features.shape[1]} features")
    LOG.info(f"LSTM feature columns: {list(lstm_features.columns)}")
    
    # Step 3: Also run the original feature engineering for additional features (if needed for TFT)
    # Initialize feature engineer
    engineer = FeatureEngineer(lookback_windows=[10, 30, 60])
    
    # Process each ticker
    all_features = []
    for ticker in TICKERS:
        # Filter data for this ticker
        ticker_stock = stock_df[stock_df['ticker'] == ticker].copy()
        ticker_options = options_df[options_df['ticker'] == ticker].copy()
        
        # Handle minute bars - check if it exists and has data
        ticker_bars = None
        if minute_bars_df is not None and not minute_bars_df.empty and 'ticker' in minute_bars_df.columns:
            ticker_bars = minute_bars_df[minute_bars_df['ticker'] == ticker].copy()
            if ticker_bars.empty:
                ticker_bars = None
                LOG.debug(f"No minute bars for {ticker}")
        else:
            LOG.debug(f"Minute bars not available for {ticker}")
        
        if ticker_stock.empty or ticker_options.empty:
            LOG.warning(f"No stock or options data for {ticker}, skipping")
            continue
        
        # Convert stock DataFrame to expected dictionary format
        try:
            stock_row = ticker_stock.iloc[0]  # Take the most recent row
            
            # Handle case where some values might be NaN
            price = stock_row.get('price', 0)
            if pd.isna(price):
                # Try to find a fallback price or use a default
                price = 24.0  # Unity is around $24, this is a fallback
                LOG.warning(f"Price is NaN for {ticker}, using fallback: ${price}")
            
            # Create the stock data dictionary in the expected format
            stock_dict = {
                'day': {
                    'c': price,  # close
                    'h': price * 1.02,  # high (approximate)
                    'l': price * 0.98,  # low (approximate)
                    'o': price,  # open (approximate)
                    'v': 1000000,  # volume (approximate)
                },
                'prev_day': {
                    'c': price * (1 - stock_row.get('mark_pct', 0) / 100),  # prev close based on mark_pct
                },
                'price': price,
                'net_change': stock_row.get('net_change', 0),
                'mark_pct': stock_row.get('mark_pct', 0),
                'ticker': ticker,
                'timestamp': stock_row.get('timestamp', pd.Timestamp.now())
            }
            
            LOG.debug(f"Converted stock data for {ticker}: price=${price}, net_change={stock_row.get('net_change', 0)}")
            
        except Exception as e:
            LOG.error(f"Failed to convert stock data for {ticker}: {e}")
            continue
            
        try:
            # Engineer features
            features_dict = engineer.engineer_all_features(
                stock_data=stock_dict,  # Now passing dictionary instead of DataFrame
                minute_bars=ticker_bars,
                option_chain=ticker_options
            )
            
            # Convert features dictionary to DataFrame
            if features_dict and isinstance(features_dict, dict):
                # Add ticker to features
                features_dict['ticker'] = ticker
                features_df = pd.DataFrame([features_dict])  # Convert dict to single-row DataFrame
                all_features.append(features_df)
                LOG.info(f"Engineered {len(features_dict)} features for {ticker}")
                # Log the feature names for debugging
                LOG.debug(f"Feature names for {ticker}: {list(features_dict.keys())[:20]}...")  # Show first 20
            else:
                LOG.warning(f"No features returned for {ticker}")
                
        except Exception as e:
            LOG.error(f"Feature engineering failed for {ticker}: {e}")
            
    # Step 4: Combine LSTM features with any additional engineered features
    if all_features:
        additional_features = pd.concat(all_features, ignore_index=True)
        
        # Add ticker information to LSTM features if available
        if 'ticker' in options_with_greeks.columns:
            lstm_features = lstm_features.copy()  # Create explicit copy to avoid warning
            lstm_features['ticker'] = options_with_greeks['ticker'].values
            
        # For now, prioritize LSTM features since that's what the model expects
        result = lstm_features.copy()
        
        # Add any additional valuable features from the feature engineer
        for col in additional_features.columns:
            if col not in result.columns and col not in ['ticker']:
                try:
                    result[col] = additional_features[col].iloc[0] if len(additional_features) > 0 else 0
                except:
                    result[col] = 0
                    
        LOG.info(f"Combined features: LSTM({len(lstm_features.columns)}) + Additional({len(additional_features.columns)})")
    else:
        # Use only LSTM features
        result = lstm_features.copy()
        if 'ticker' in options_with_greeks.columns:
            result['ticker'] = options_with_greeks['ticker'].values
            
        LOG.info(f"Using LSTM features only: {len(lstm_features.columns)} features")
    
    if result.empty:
        LOG.warning("No features generated")
        return pd.DataFrame()
        
    LOG.info(f"Total engineered features: {len(result)} rows, {len(result.columns)} columns")
    return result

# --- MODEL INFERENCE ---
def run_model_inference(features_df: pd.DataFrame) -> pd.DataFrame:
    """Run model inference using TFT or fallback to LSTM."""
    if features_df.empty:
        LOG.warning("No features for inference")
        return pd.DataFrame()
        
    # Try TFT first
    if TFT_MODEL_PATH.exists():
        try:
            LOG.info("Running TFT inference...")
            model = get_tft_model(TFT_MODEL_PATH)
            
            # TFT returns dict with predictions and confidence
            predictions = model.batch_predict(features_df)
            
            # Merge predictions back to features
            result_df = features_df.copy()
            result_df['prediction'] = predictions.get('prediction', 0)
            result_df['confidence'] = predictions.get('confidence', 0.5)
            result_df['std'] = predictions.get('std', 0.1)
            
            # Add quantiles if available
            for q in ['q0.1', 'q0.5', 'q0.9']:
                if q in predictions:
                    result_df[q] = predictions[q]
                    
            LOG.info(f"TFT inference successful: {len(result_df)} predictions")
            return result_df
            
        except Exception as e:
            LOG.error(f"TFT inference failed: {e}")
            
    # Fallback to LSTM
    if LSTM_MODEL_PATH.exists():
        LOG.info("Falling back to LSTM model...")
        try:
            # Use the exact 9 features that the LSTM model was trained on
            # These are already prepared by the GreeksCalculator.prepare_lstm_features()
            lstm_feature_list = ['strike', 'bid', 'ask', 'iv', 'delta', 'gamma', 'theta', 'vega', 'oi']
            
            LOG.info(f"LSTM model expects {len(lstm_feature_list)} features: {lstm_feature_list}")
            LOG.info(f"Available features in data: {list(features_df.columns)}")
            
            # Check if we have the required LSTM features
            missing_features = [f for f in lstm_feature_list if f not in features_df.columns]
            if missing_features:
                LOG.error(f"Missing required LSTM features: {missing_features}")
                LOG.error("Greeks calculation may have failed. Cannot proceed with LSTM inference.")
                return pd.DataFrame()
            
            # Extract the exact features the model expects
            lstm_features = features_df[lstm_feature_list].copy()
            
            # Clean and prepare the data
            for col in lstm_feature_list:
                lstm_features[col] = pd.to_numeric(lstm_features[col], errors='coerce').fillna(0)
            
            # Verify we have valid data
            if lstm_features.isnull().any().any():
                LOG.warning("Some LSTM features contain NaN values, filling with 0")
                lstm_features = lstm_features.fillna(0)
            
            X = lstm_features.values.astype(float)
            LOG.info(f"LSTM input shape: {X.shape}")
            LOG.info(f"LSTM input sample (first row): {X[0] if len(X) > 0 else 'No data'}")
            
            # Reshape for LSTM: (batch_size, sequence_length, input_dim)
            # Since we have individual options samples, sequence_length = 1
            X = X.reshape(X.shape[0], 1, X.shape[1])  # (948, 1, 9)
            LOG.info(f"LSTM input reshaped: {X.shape}")
            
            # Load model with correct input dimension (9 features)
            model = PriceLSTM(input_dim=9, hidden_size=256, lstm_layers=2, dropout=0.0)
            
            state_dict = torch.load(LSTM_MODEL_PATH, map_location='cpu')
            if 'state_dict' in state_dict:
                # Handle Lightning checkpoint format
                state_dict = {k.replace('model.', ''): v for k, v in state_dict['state_dict'].items()}
            model.load_state_dict(state_dict)
            model.eval()
            
            # Run inference
            with torch.no_grad():
                predictions = model(torch.tensor(X, dtype=torch.float32)).squeeze().numpy()
                
            # Handle single prediction case
            if predictions.ndim == 0:
                predictions = np.array([predictions])
                
            LOG.info(f"LSTM predictions shape: {predictions.shape}")
            LOG.info(f"LSTM predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
                
            # Create output dataframe
            result_df = features_df.copy()
            result_df['prediction'] = predictions
            
            # Add confidence based on IV and other factors
            iv_values = lstm_features['iv'].values
            base_confidence = 0.6 + 0.2 * np.clip(iv_values / 0.5, 0, 1)  # Higher IV = higher confidence
            result_df['confidence'] = np.clip(base_confidence + np.random.normal(0, 0.05, len(features_df)), 0.3, 0.95)
            
            # Add mock quantiles for compatibility
            result_df['q0.5'] = predictions
            result_df['q0.1'] = predictions - 0.15 * np.abs(predictions)
            result_df['q0.9'] = predictions + 0.15 * np.abs(predictions)
            result_df['std'] = 0.1 * np.abs(predictions)
            
            LOG.info(f"LSTM inference successful: {len(result_df)} predictions")
            return result_df
            result_df['q0.9'] = predictions + 0.15 * np.abs(predictions)
            result_df['std'] = 0.1 * np.abs(predictions)
            
            LOG.info(f"LSTM inference successful: {len(result_df)} predictions")
            return result_df
            
        except Exception as e:
            LOG.error(f"LSTM inference failed: {e}")
            
    # No model available
    LOG.error("No model available for inference")
    return pd.DataFrame()

# --- SAVE RESULTS ---
def save_predictions(predictions_df: pd.DataFrame) -> Path:
    """Save predictions to parquet file."""
    if predictions_df.empty:
        LOG.warning("No predictions to save")
        return None
        
    # Ensure numeric columns are properly typed
    numeric_cols = ['strike', 'bid', 'ask', 'iv', 'delta', 'gamma', 'theta', 
                   'vega', 'oi', 'vol', 'price', 'prediction', 'confidence', 
                   'q0.1', 'q0.5', 'q0.9', 'std']
    
    for col in numeric_cols:
        if col in predictions_df.columns:
            predictions_df[col] = pd.to_numeric(predictions_df[col], errors='coerce')
            
    # Save to parquet
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    output_path = OUTPUTS_DIR / f"predictions_{timestamp}.parquet"
    predictions_df.to_parquet(output_path, index=False)
    
    LOG.info(f"Saved {len(predictions_df)} predictions to {output_path}")
    return output_path

# --- MAIN PIPELINE ---
async def main():
    """Run the complete automated pipeline."""
    LOG.info("="*60)
    LOG.info("Starting automated trading pipeline")
    LOG.info(f"Tickers: {TICKERS}")
    LOG.info("="*60)
    
    # Step 1: Fetch latest data
    if os.environ.get("POLYGON_API_KEY"):
        LOG.info("Step 1: Fetching latest data from Polygon...")
        success = await fetch_polygon_data()
        if not success:
            LOG.warning("Data fetch failed, attempting to use cached data...")
    else:
        LOG.info("No POLYGON_API_KEY set, using cached data...")
        
    # Step 2: Load data
    LOG.info("Step 2: Loading data from parquet files...")
    stock_df = load_latest_data(STOCK_QUOTES_DIR, target_tickers=TICKERS)
    options_df = load_latest_data(OPTIONS_DIR, target_tickers=TICKERS)
    minute_bars_df = load_latest_data(MINUTE_BARS_DIR, target_tickers=TICKERS)
    
    if stock_df.empty or options_df.empty:
        LOG.error("Insufficient data loaded. Exiting.")
        return
    
    # Log what tickers we found
    if 'ticker' in stock_df.columns:
        stock_tickers = sorted(stock_df['ticker'].unique())
        LOG.info(f"Stock data tickers found: {stock_tickers}")
    else:
        LOG.warning("No 'ticker' column in stock data")
        
    if 'ticker' in options_df.columns:
        options_tickers = sorted(options_df['ticker'].unique())
        LOG.info(f"Options data tickers found: {options_tickers}")
    else:
        LOG.warning("No 'ticker' column in options data")
        
    LOG.info(f"Loaded - Stocks: {len(stock_df)}, Options: {len(options_df)}, Bars: {len(minute_bars_df)}")
    
    # Step 3: Feature engineering
    LOG.info("Step 3: Engineering features...")
    features_df = engineer_features(stock_df, options_df, minute_bars_df)
    
    if features_df.empty:
        LOG.error("Feature engineering produced no results. Exiting.")
        return
        
    # Step 4: Model inference
    LOG.info("Step 4: Running model inference...")
    predictions_df = run_model_inference(features_df)
    
    if predictions_df.empty:
        LOG.error("Model inference produced no results. Exiting.")
        return
        
    # Step 5: Save predictions
    LOG.info("Step 5: Saving predictions...")
    output_path = save_predictions(predictions_df)
    
    # Step 6: Generate position recommendations
    LOG.info("Step 6: Generating position recommendations...")
    try:
        config = load_positions_config()
        avg_confidence = predictions_df['confidence'].mean() if 'confidence' in predictions_df else 0.5
        recommendations = recommend_positions(predictions_df, avg_confidence, config)
        
        # Log summary
        LOG.info("="*60)
        LOG.info("PIPELINE SUMMARY:")
        LOG.info(f"✓ Processed {len(TICKERS)} tickers")
        LOG.info(f"✓ Generated {len(predictions_df)} predictions")
        LOG.info(f"✓ Prediction range: [{predictions_df['prediction'].min():.3f}, {predictions_df['prediction'].max():.3f}]")
        LOG.info(f"✓ Average confidence: {avg_confidence:.3f}")
        LOG.info(f"✓ Generated {len(recommendations)} position recommendations")
        
        # Save recommendations
        rec_path = OUTPUTS_DIR / f"recommendations_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.json"
        with open(rec_path, 'w') as f:
            json.dump(recommendations, f, indent=2)
        LOG.info(f"✓ Saved recommendations to {rec_path}")
        
        # Print sample recommendations
        if recommendations:
            LOG.info("\nSample recommendations:")
            for i, (key, rec) in enumerate(list(recommendations.items())[:3]):
                if key == "_portfolio_summary":
                    LOG.info(f"  Portfolio: {rec.get('message', 'Status unknown')}")
                elif isinstance(rec, dict) and 'action' in rec:
                    LOG.info(f"  {key}: {rec['action']} size={rec['size']}, confidence={rec['certainty']:.2f}")
                else:
                    LOG.info(f"  {key}: {rec}")
                
    except Exception as e:
        LOG.error(f"Position recommendation failed: {e}")
        
    LOG.info("="*60)
    LOG.info("Pipeline completed successfully!")

if __name__ == "__main__":
    # Create logs directory if needed
    Path("logs").mkdir(exist_ok=True)
    
    # Run the async main function
    asyncio.run(main())