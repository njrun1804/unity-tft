#!/usr/bin/env python
"""
polygon_fetcher.py - Production-ready Polygon.io data fetcher
Writes directly to your Parquet-based feature store
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

class PolygonDataFetcher:
    """
    Production-ready Polygon.io data fetcher with:
    - Async/concurrent requests
    - Automatic retries
    - Rate limiting
    - Direct Parquet output to match your feature store
    """
    
    def __init__(self, api_key: str, feature_store_path: Path = Path("data/feature_store")):
        self.api_key = api_key
        self.feature_store_path = feature_store_path
        self.feature_store_path.mkdir(parents=True, exist_ok=True)
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limit_remaining = 5  # Conservative default
        self.rate_limit_reset = None
        self.base_url = "https://api.polygon.io"
        
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def _rate_limited_request(self, url: str, params: Dict) -> Dict:
        """Make rate-limited API request with retry logic"""
        await self._check_rate_limit()
        
        params["apikey"] = self.api_key
        try:
            async with self.session.get(url, params=params) as response:
                # Update rate limit info from headers
                self.rate_limit_remaining = int(response.headers.get('X-RateLimit-Remaining', 0))
                reset_time = response.headers.get('X-RateLimit-Reset')
                if reset_time:
                    self.rate_limit_reset = datetime.fromtimestamp(int(reset_time))
                
                if response.status == 429:
                    logger.warning("Rate limit exceeded, waiting...")
                    await asyncio.sleep(60)
                    return await self._rate_limited_request(url, params)
                
                response.raise_for_status()
                return await response.json()
                
        except Exception as e:
            logger.error(f"API request failed: {e}")
            raise
        
    async def _check_rate_limit(self):
        """Check rate limit status and pause if needed"""
        if self.rate_limit_remaining <= 1:
            if self.rate_limit_reset:
                wait_time = max(0, (self.rate_limit_reset - datetime.now()).total_seconds())
                logger.info(f"Rate limit low, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time + 1)
            else:
                await asyncio.sleep(12)  # Default 12s between requests

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def fetch_stock_snapshot(self, ticker: str) -> Dict:
        """Fetch real-time stock snapshot"""
        url = f"{self.base_url}/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"
        
        try:
            response = await self._rate_limited_request(url, {})
            
            if 'results' not in response:
                logger.warning(f"No results for {ticker}")
                return {}
                
            result = response['results']
            
            # Extract stock data similar to existing scripts/polygon_ingest.py
            stock_data = {
                'ticker': ticker,
                'price': result.get('last_trade', {}).get('p'),
                'net_change': result.get('todaysChange'),
                'mark_pct': result.get('todaysChangePerc'),
                'volume': result.get('day', {}).get('v'),
                'day': {
                    'o': result.get('day', {}).get('o'),
                    'h': result.get('day', {}).get('h'),
                    'l': result.get('day', {}).get('l'),
                    'c': result.get('day', {}).get('c'),
                    'v': result.get('day', {}).get('v'),
                },
                'prev_day': {
                    'c': result.get('prevDay', {}).get('c'),
                    'o': result.get('prevDay', {}).get('o'),
                    'h': result.get('prevDay', {}).get('h'),
                    'l': result.get('prevDay', {}).get('l'),
                    'v': result.get('prevDay', {}).get('v'),
                },
                'timestamp': pd.Timestamp.now(tz='UTC')
            }
            
            logger.info(f"Fetched stock snapshot for {ticker}: ${stock_data['price']}")
            return stock_data
            
        except Exception as e:
            logger.error(f"Failed to fetch stock snapshot for {ticker}: {e}")
            return {}

    async def fetch_option_chain(self, ticker: str, min_volume: int = 10):
        """Fetch option chain with filtering"""
        url = f"{self.base_url}/v3/snapshot/options/{ticker}"
        
        try:
            response = await self._rate_limited_request(url, {})
            
            if 'results' not in response:
                logger.warning(f"No option chain for {ticker}")
                return []
                
            options = []
            for option in response['results']:
                # Extract option data similar to scripts/polygon_ingest.py
                details = option.get('details', {})
                day_data = option.get('day', {})
                greeks = option.get('greeks', {})
                
                option_data = {
                    'ticker': ticker,
                    'timestamp': pd.Timestamp.now(tz='UTC'),
                    'expiry': details.get('expiration_date'),
                    'strike': details.get('strike_price'),
                    'cp': details.get('contract_type'),  # 'call' or 'put'
                    'bid': day_data.get('bid'),
                    'ask': day_data.get('ask'),
                    'iv': option.get('implied_volatility'),
                    'delta': greeks.get('delta'),
                    'gamma': greeks.get('gamma'),
                    'theta': greeks.get('theta'),
                    'vega': greeks.get('vega'),
                    'volume': day_data.get('volume', 0),
                    'open_interest': option.get('open_interest', 0),
                    'price': option.get('last_quote', {}).get('p'),
                    'dte': self._calculate_dte(details.get('expiration_date', ''))
                }
                
                # Filter by minimum volume
                if option_data['volume'] and option_data['volume'] >= min_volume:
                    options.append(option_data)
            
            logger.info(f"Fetched {len(options)} options for {ticker}")
            return options
            
        except Exception as e:
            logger.error(f"Failed to fetch option chain for {ticker}: {e}")
            return []

    async def fetch_minute_bars(self, ticker: str, lookback_days: int = 5):
        """Fetch historical minute bars"""
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=lookback_days)
        
        url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/minute/{start_date}/{end_date}"
        
        try:
            response = await self._rate_limited_request(url, {
                'adjusted': 'true',
                'sort': 'asc',
                'limit': 50000
            })
            
            if 'results' not in response:
                logger.warning(f"No minute bars for {ticker}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            bars = []
            for bar in response['results']:
                bars.append({
                    'ticker': ticker,
                    'timestamp': pd.to_datetime(bar['t'], unit='ms', utc=True),
                    'open': bar['o'],
                    'high': bar['h'],
                    'low': bar['l'],
                    'close': bar['c'],
                    'volume': bar['v'],
                    'vwap': bar.get('vw')
                })
            
            df = pd.DataFrame(bars)
            if not df.empty:
                df = df.set_index('timestamp').sort_index()
            
            logger.info(f"Fetched {len(df)} minute bars for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch minute bars for {ticker}: {e}")
            return pd.DataFrame()
    
    def _calculate_dte(self, expiry_date: str) -> int:
        """Calculate days to expiration"""
        if not expiry_date:
            return 0
        try:
            expiry = pd.to_datetime(expiry_date)
            return max(0, (expiry - pd.Timestamp.now()).days)
        except:
            return 0

    async def fetch_all_data(self, ticker: str) -> Dict:
        """Orchestrate fetching all data types for a ticker"""
        logger.info(f"Fetching all data for {ticker}")
        
        # Fetch all data concurrently
        tasks = [
            self.fetch_stock_snapshot(ticker),
            self.fetch_option_chain(ticker),
            self.fetch_minute_bars(ticker)
        ]
        
        try:
            stock_data, option_chain, minute_bars = await asyncio.gather(*tasks)
            
            result = {
                'stock_data': stock_data,
                'option_chain': option_chain,
                'minute_bars': minute_bars,
                'ticker': ticker,
                'timestamp': datetime.now()
            }
            
            # Save to feature store
            self.save_to_feature_store(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to fetch all data for {ticker}: {e}")
            return {'ticker': ticker, 'error': str(e)}

    def save_to_feature_store(self, data: Dict):
        """Save data to Parquet feature store matching existing format"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        ticker = data['ticker']
        
        try:
            # Save stock snapshot
            if data.get('stock_data'):
                watchlist_df = self._create_watchlist_df(data['stock_data'])
                if not watchlist_df.empty:
                    self._write_parquet(watchlist_df, "watchlist", timestamp)
            
            # Save option chain
            if data.get('option_chain'):
                option_df = self._create_option_chain_df(data['option_chain'])
                if not option_df.empty:
                    self._write_parquet(option_df, "option_chain", timestamp)
            
            # Save minute bars
            if data.get('minute_bars') is not None and not data['minute_bars'].empty:
                bars_df = data['minute_bars'].reset_index()
                self._write_parquet(bars_df, "minute_bars", timestamp)
            
            logger.info(f"Saved all data to feature store: {timestamp}")
            
        except Exception as e:
            logger.error(f"Failed to save data to feature store: {e}")
    
    def _create_watchlist_df(self, stock_data: Dict) -> pd.DataFrame:
        """Convert stock data to watchlist DataFrame format"""
        if not stock_data:
            return pd.DataFrame()
        
        return pd.DataFrame([{
            'ticker': stock_data['ticker'],
            'price': stock_data['price'],
            'net_change': stock_data['net_change'],
            'mark_pct': stock_data['mark_pct'],
            'timestamp': stock_data['timestamp']
        }])
    
    def _create_option_chain_df(self, option_chain: List[Dict]) -> pd.DataFrame:
        """Convert option chain to DataFrame format"""
        if not option_chain:
            return pd.DataFrame()
        
        return pd.DataFrame(option_chain)
    
    def _write_parquet(self, df: pd.DataFrame, data_type: str, timestamp: str):
        """Write DataFrame to Parquet with proper schema"""
        if df.empty:
            return
        
        # Create schema based on data type
        if data_type == "watchlist":
            schema = pa.schema([
                ("ticker", pa.string()),
                ("price", pa.float64()),
                ("net_change", pa.float64()),
                ("mark_pct", pa.float64()),
                ("timestamp", pa.timestamp("ms"))
            ])
        elif data_type == "option_chain":
            schema = pa.schema([
                ("ticker", pa.string()),
                ("timestamp", pa.timestamp("ms")),
                ("expiry", pa.string()),
                ("strike", pa.float64()),
                ("cp", pa.string()),
                ("bid", pa.float64()),
                ("ask", pa.float64()),
                ("iv", pa.float64()),
                ("delta", pa.float64()),
                ("price", pa.float64())
            ])
        else:
            schema = None  # Let pandas infer for minute bars
        
        # Write to feature store
        out_dir = self.feature_store_path / f"polygon_{data_type}"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        fname = out_dir / f"part_{timestamp}.parquet"
        
        if schema:
            table = pa.Table.from_pandas(df, preserve_index=False, schema=schema)
            pq.write_table(table, fname, compression="zstd")
        else:
            df.to_parquet(fname, compression="zstd")
        
        logger.info(f"Wrote {len(df)} rows to {fname}")

async def fetch_and_save_data(ticker: str, api_key: str):
    """Convenience function for single ticker fetching"""
    async with PolygonDataFetcher(api_key) as fetcher:
        data = await fetcher.fetch_all_data(ticker)
        return data

if __name__ == "__main__":
    import os
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        raise ValueError("Please set POLYGON_API_KEY environment variable")
    asyncio.run(fetch_and_save_data('U', api_key))
