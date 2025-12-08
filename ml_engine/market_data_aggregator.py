"""
Market Data Aggregator - Multi-source data aggregation with fallback priority
Sources: Yahoo Finance, Polygon.io, FMP, Alpha Vantage
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import logging
import time
import requests

logger = logging.getLogger(__name__)


class MarketDataAggregator:
    """
    Multi-source market data aggregator with fallback priority.

    Fallback Priority:
    1. Yahoo Finance (free, reliable)
    2. Polygon.io (if enabled)
    3. FMP (if enabled)
    4. Alpha Vantage (if enabled)
    """

    # Symbol mappings for different data sources
    SYMBOL_MAPPINGS = {
        'BTC/USDT': {'yahoo': 'BTC-USD', 'polygon': 'X:BTCUSD', 'fmp': 'BTCUSD', 'alpha': 'BTC'},
        'ETH/USDT': {'yahoo': 'ETH-USD', 'polygon': 'X:ETHUSD', 'fmp': 'ETHUSD', 'alpha': 'ETH'},
        'SOL/USDT': {'yahoo': 'SOL-USD', 'polygon': 'X:SOLUSD', 'fmp': 'SOLUSD', 'alpha': 'SOL'},
        'AVAX/USDT': {'yahoo': 'AVAX-USD', 'polygon': 'X:AVAXUSD', 'fmp': 'AVAXUSD', 'alpha': 'AVAX'},
        'MATIC/USDT': {'yahoo': 'MATIC-USD', 'polygon': 'X:MATICUSD', 'fmp': 'MATICUSD', 'alpha': 'MATIC'},
        'LINK/USDT': {'yahoo': 'LINK-USD', 'polygon': 'X:LINKUSD', 'fmp': 'LINKUSD', 'alpha': 'LINK'},
        'UNI/USDT': {'yahoo': 'UNI-USD', 'polygon': 'X:UNIUSD', 'fmp': 'UNIUSD', 'alpha': 'UNI'},
        'AAVE/USDT': {'yahoo': 'AAVE-USD', 'polygon': 'X:AAVEUSD', 'fmp': 'AAVEUSD', 'alpha': 'AAVE'},
        'SUSHI/USDT': {'yahoo': 'SUSHI-USD', 'polygon': 'X:SUSHIUSD', 'fmp': 'SUSHIUSD', 'alpha': 'SUSHI'},
        'CRV/USDT': {'yahoo': 'CRV-USD', 'polygon': 'X:CRVUSD', 'fmp': 'CRVUSD', 'alpha': 'CRV'},
        'LDO/USDT': {'yahoo': 'LDO-USD', 'polygon': 'X:LDOUSD', 'fmp': 'LDOUSD', 'alpha': 'LDO'},
        'ARB/USDT': {'yahoo': 'ARB-USD', 'polygon': 'X:ARBUSD', 'fmp': 'ARBUSD', 'alpha': 'ARB'},
        'OP/USDT': {'yahoo': 'OP-USD', 'polygon': 'X:OPUSD', 'fmp': 'OPUSD', 'alpha': 'OP'},
        'IMX/USDT': {'yahoo': 'IMX-USD', 'polygon': 'X:IMXUSD', 'fmp': 'IMXUSD', 'alpha': 'IMX'},
        'HYPE/USDT': {'yahoo': 'HYPE-USD', 'polygon': 'X:HYPEUSD', 'fmp': 'HYPEUSD', 'alpha': 'HYPE'},
    }

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize market data aggregator.

        Args:
            config: Configuration dictionary with API keys
        """
        self.config = config
        self._setup_sources()
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_ttl = 300  # 5 minutes

    def _setup_sources(self):
        """Setup data source configurations"""
        self.sources = {
            'yahoo': {
                'enabled': self.config.get('YAHOO_FINANCE_ENABLED', 'true').lower() == 'true',
                'fetch_fn': self._fetch_yahoo,
                'priority': 1,
            },
            'polygon': {
                'enabled': bool(self.config.get('POLYGON_API_KEY')),
                'api_key': self.config.get('POLYGON_API_KEY', ''),
                'fetch_fn': self._fetch_polygon,
                'priority': 2,
            },
            'fmp': {
                'enabled': bool(self.config.get('FMP_API_KEY')),
                'api_key': self.config.get('FMP_API_KEY', ''),
                'fetch_fn': self._fetch_fmp,
                'priority': 3,
            },
            'alpha_vantage': {
                'enabled': bool(self.config.get('ALPHA_VANTAGE_API_KEY')),
                'api_key': self.config.get('ALPHA_VANTAGE_API_KEY', ''),
                'fetch_fn': self._fetch_alpha_vantage,
                'priority': 4,
            },
        }

        # Sort sources by priority
        self.source_order = sorted(
            [(k, v) for k, v in self.sources.items() if v['enabled']],
            key=lambda x: x[1]['priority']
        )

        logger.info(f"Enabled data sources: {[s[0] for s in self.source_order]}")

    def fetch_aggregated_data(
        self,
        symbol: str,
        lookback_days: int = 30,
        interval: str = '1h'
    ) -> pd.DataFrame:
        """
        Fetch aggregated data with automatic fallback.

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            lookback_days: Number of days to look back
            interval: Data interval ('1h', '1d', etc.)

        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{symbol}_{lookback_days}_{interval}"

        # Check cache
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Try each source in priority order
        for source_name, source_config in self.source_order:
            try:
                logger.info(f"Trying {source_name} for {symbol}...")
                fetch_fn = source_config['fetch_fn']
                df = fetch_fn(symbol, lookback_days, interval)

                if df is not None and not df.empty:
                    logger.info(f"Successfully fetched {len(df)} rows from {source_name} for {symbol}")
                    df['data_source'] = source_name

                    # Cache the result
                    self._cache[cache_key] = df
                    return df

            except Exception as e:
                logger.warning(f"Failed to fetch from {source_name}: {e}")
                continue

        logger.error(f"All data sources failed for {symbol}")
        return pd.DataFrame()

    def _fetch_yahoo(
        self,
        symbol: str,
        lookback_days: int,
        interval: str
    ) -> Optional[pd.DataFrame]:
        """Fetch data from Yahoo Finance"""
        try:
            import yfinance as yf

            # Get Yahoo symbol mapping
            yahoo_symbol = self._get_mapped_symbol(symbol, 'yahoo')
            if not yahoo_symbol:
                return None

            # Map interval
            yahoo_interval = self._map_interval_yahoo(interval)

            # Fetch data
            ticker = yf.Ticker(yahoo_symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)

            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=yahoo_interval
            )

            if df.empty:
                return None

            # Standardize columns
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            df = df[['open', 'high', 'low', 'close', 'volume']]
            df['symbol'] = symbol
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

            return df

        except ImportError:
            logger.error("yfinance not installed. Run: pip install yfinance")
            return None
        except Exception as e:
            logger.error(f"Yahoo Finance error: {e}")
            return None

    def _fetch_polygon(
        self,
        symbol: str,
        lookback_days: int,
        interval: str
    ) -> Optional[pd.DataFrame]:
        """Fetch data from Polygon.io"""
        try:
            api_key = self.sources['polygon']['api_key']
            if not api_key:
                return None

            polygon_symbol = self._get_mapped_symbol(symbol, 'polygon')
            if not polygon_symbol:
                return None

            # Map interval to Polygon format
            multiplier, timespan = self._map_interval_polygon(interval)

            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)

            url = (
                f"https://api.polygon.io/v2/aggs/ticker/{polygon_symbol}/range/"
                f"{multiplier}/{timespan}/{start_date.strftime('%Y-%m-%d')}/"
                f"{end_date.strftime('%Y-%m-%d')}?apiKey={api_key}&limit=50000"
            )

            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            if 'results' not in data or not data['results']:
                return None

            df = pd.DataFrame(data['results'])
            df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
            df.set_index('timestamp', inplace=True)

            df = df.rename(columns={
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume'
            })

            df = df[['open', 'high', 'low', 'close', 'volume']]
            df['symbol'] = symbol
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

            return df

        except Exception as e:
            logger.error(f"Polygon.io error: {e}")
            return None

    def _fetch_fmp(
        self,
        symbol: str,
        lookback_days: int,
        interval: str
    ) -> Optional[pd.DataFrame]:
        """Fetch data from Financial Modeling Prep"""
        try:
            api_key = self.sources['fmp']['api_key']
            if not api_key:
                return None

            fmp_symbol = self._get_mapped_symbol(symbol, 'fmp')
            if not fmp_symbol:
                return None

            # FMP uses different endpoints for different intervals
            if interval in ['1h', '4h']:
                url = (
                    f"https://financialmodelingprep.com/api/v3/historical-chart/"
                    f"{interval}/{fmp_symbol}?apikey={api_key}"
                )
            else:
                url = (
                    f"https://financialmodelingprep.com/api/v3/historical-price-full/"
                    f"{fmp_symbol}?apikey={api_key}"
                )

            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif 'historical' in data:
                df = pd.DataFrame(data['historical'])
            else:
                return None

            if df.empty:
                return None

            # Handle different column names
            if 'date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)

            # Standardize columns
            df = df.rename(columns={
                'Open': 'open', 'open': 'open',
                'High': 'high', 'high': 'high',
                'Low': 'low', 'low': 'low',
                'Close': 'close', 'close': 'close',
                'Volume': 'volume', 'volume': 'volume'
            })

            df = df[['open', 'high', 'low', 'close', 'volume']]
            df['symbol'] = symbol
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

            # Filter to lookback period
            cutoff = datetime.now() - timedelta(days=lookback_days)
            df = df[df.index >= cutoff]

            return df

        except Exception as e:
            logger.error(f"FMP error: {e}")
            return None

    def _fetch_alpha_vantage(
        self,
        symbol: str,
        lookback_days: int,
        interval: str
    ) -> Optional[pd.DataFrame]:
        """Fetch data from Alpha Vantage"""
        try:
            api_key = self.sources['alpha_vantage']['api_key']
            if not api_key:
                return None

            alpha_symbol = self._get_mapped_symbol(symbol, 'alpha')
            if not alpha_symbol:
                return None

            # Map interval
            av_interval = self._map_interval_alpha(interval)

            url = (
                f"https://www.alphavantage.co/query?function=CRYPTO_INTRADAY"
                f"&symbol={alpha_symbol}&market=USD&interval={av_interval}"
                f"&outputsize=full&apikey={api_key}"
            )

            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Handle rate limiting
            if 'Note' in data:
                logger.warning("Alpha Vantage rate limit hit")
                return None

            # Find the time series key
            time_series_key = None
            for key in data.keys():
                if 'Time Series' in key:
                    time_series_key = key
                    break

            if not time_series_key or not data[time_series_key]:
                return None

            df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)

            # Standardize columns
            df.columns = [col.split('. ')[1] if '. ' in col else col for col in df.columns]
            df = df.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })

            df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            df['symbol'] = symbol
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

            # Filter to lookback period
            cutoff = datetime.now() - timedelta(days=lookback_days)
            df = df[df.index >= cutoff]

            return df

        except Exception as e:
            logger.error(f"Alpha Vantage error: {e}")
            return None

    def _get_mapped_symbol(self, symbol: str, source: str) -> Optional[str]:
        """Get mapped symbol for a data source"""
        if symbol in self.SYMBOL_MAPPINGS:
            return self.SYMBOL_MAPPINGS[symbol].get(source)

        # Default mapping: remove /USDT and format for source
        base = symbol.replace('/USDT', '').replace('/USD', '')
        if source == 'yahoo':
            return f"{base}-USD"
        elif source == 'polygon':
            return f"X:{base}USD"
        elif source == 'fmp':
            return f"{base}USD"
        elif source == 'alpha':
            return base
        return None

    def _map_interval_yahoo(self, interval: str) -> str:
        """Map interval to Yahoo Finance format"""
        mapping = {
            '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '4h': '1h', '1d': '1d', '1w': '1wk'
        }
        return mapping.get(interval, '1h')

    def _map_interval_polygon(self, interval: str) -> tuple:
        """Map interval to Polygon format (multiplier, timespan)"""
        mapping = {
            '1m': (1, 'minute'), '5m': (5, 'minute'), '15m': (15, 'minute'),
            '30m': (30, 'minute'), '1h': (1, 'hour'), '4h': (4, 'hour'),
            '1d': (1, 'day'), '1w': (1, 'week')
        }
        return mapping.get(interval, (1, 'hour'))

    def _map_interval_alpha(self, interval: str) -> str:
        """Map interval to Alpha Vantage format"""
        mapping = {
            '1m': '1min', '5m': '5min', '15m': '15min',
            '30m': '30min', '1h': '60min'
        }
        return mapping.get(interval, '60min')

    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        lookback_days: int = 30,
        interval: str = '1h'
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols.

        Args:
            symbols: List of trading pair symbols
            lookback_days: Number of days to look back
            interval: Data interval

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}
        for symbol in symbols:
            try:
                df = self.fetch_aggregated_data(symbol, lookback_days, interval)
                if not df.empty:
                    results[symbol] = df
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
        return results

    def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get current prices for symbols.

        Args:
            symbols: List of trading pair symbols

        Returns:
            Dictionary mapping symbols to current prices
        """
        prices = {}
        for symbol in symbols:
            try:
                df = self.fetch_aggregated_data(symbol, lookback_days=1, interval='1h')
                if not df.empty:
                    prices[symbol] = df['close'].iloc[-1]
            except Exception as e:
                logger.error(f"Error getting price for {symbol}: {e}")
        return prices

    def clear_cache(self):
        """Clear data cache"""
        self._cache.clear()
        logger.info("Data cache cleared")
