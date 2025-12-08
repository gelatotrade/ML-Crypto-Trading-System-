"""
Data Collector - CCXT-based exchange integration for OHLCV, orderbook, and ticker data
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import Config

logger = logging.getLogger(__name__)


@dataclass
class MarketData:
    """Container for market data"""
    symbol: str
    ohlcv: pd.DataFrame
    orderbook: Optional[Dict] = None
    ticker: Optional[Dict] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class DataCollector:
    """
    CCXT-based data collector for cryptocurrency market data.
    Supports OHLCV, orderbook, and ticker data from multiple exchanges.
    """

    SUPPORTED_EXCHANGES = ['binance', 'bybit', 'okx', 'kraken', 'coinbase']

    def __init__(self, config: Config, exchange_id: str = 'binance'):
        """
        Initialize data collector.

        Args:
            config: Configuration object
            exchange_id: Exchange to use for data collection
        """
        self.config = config
        self.exchange_id = exchange_id
        self.exchange = self._init_exchange(exchange_id)
        self._cache: Dict[str, MarketData] = {}
        self._cache_ttl = config.data.cache_ttl

    def _init_exchange(self, exchange_id: str) -> ccxt.Exchange:
        """Initialize CCXT exchange instance"""
        exchange_class = getattr(ccxt, exchange_id, None)
        if exchange_class is None:
            raise ValueError(f"Exchange {exchange_id} not supported by CCXT")

        exchange = exchange_class({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
            }
        })

        return exchange

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1h',
        lookback_days: int = 30,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV (Open, High, Low, Close, Volume) data.

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe for candles (e.g., '1h', '4h', '1d')
            lookback_days: Number of days to look back
            limit: Maximum number of candles to fetch

        Returns:
            DataFrame with OHLCV data
        """
        try:
            if limit is None:
                # Calculate limit based on timeframe and lookback
                timeframe_hours = self._timeframe_to_hours(timeframe)
                limit = int((lookback_days * 24) / timeframe_hours)

            # Fetch from exchange
            since = int((datetime.utcnow() - timedelta(days=lookback_days)).timestamp() * 1000)
            ohlcv_data = self.exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=since,
                limit=limit
            )

            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv_data,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df.astype(float)

            # Add derived columns
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['hl_range'] = (df['high'] - df['low']) / df['close']
            df['symbol'] = symbol

            logger.info(f"Fetched {len(df)} OHLCV rows for {symbol}")
            return df

        except ccxt.BaseError as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_orderbook(
        self,
        symbol: str,
        depth: int = 20
    ) -> Dict[str, Any]:
        """
        Fetch orderbook data.

        Args:
            symbol: Trading pair symbol
            depth: Number of levels to fetch

        Returns:
            Dictionary with bids, asks, and derived metrics
        """
        try:
            orderbook = self.exchange.fetch_order_book(symbol, limit=depth)

            # Process orderbook
            bids = np.array(orderbook['bids'][:depth])
            asks = np.array(orderbook['asks'][:depth])

            # Calculate metrics
            if len(bids) > 0 and len(asks) > 0:
                bid_prices = bids[:, 0]
                bid_volumes = bids[:, 1]
                ask_prices = asks[:, 0]
                ask_volumes = asks[:, 1]

                mid_price = (bid_prices[0] + ask_prices[0]) / 2
                spread = (ask_prices[0] - bid_prices[0]) / mid_price
                bid_depth = np.sum(bid_prices * bid_volumes)
                ask_depth = np.sum(ask_prices * ask_volumes)
                imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)

                return {
                    'symbol': symbol,
                    'timestamp': datetime.utcnow(),
                    'bids': bids.tolist(),
                    'asks': asks.tolist(),
                    'mid_price': mid_price,
                    'spread': spread,
                    'bid_depth': bid_depth,
                    'ask_depth': ask_depth,
                    'imbalance': imbalance,
                    'best_bid': bid_prices[0],
                    'best_ask': ask_prices[0],
                }

            return {'symbol': symbol, 'error': 'Empty orderbook'}

        except ccxt.BaseError as e:
            logger.error(f"Error fetching orderbook for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}

    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch ticker data.

        Args:
            symbol: Trading pair symbol

        Returns:
            Dictionary with ticker information
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)

            return {
                'symbol': symbol,
                'timestamp': datetime.utcnow(),
                'last': ticker.get('last', 0),
                'bid': ticker.get('bid', 0),
                'ask': ticker.get('ask', 0),
                'high': ticker.get('high', 0),
                'low': ticker.get('low', 0),
                'volume': ticker.get('baseVolume', 0),
                'quote_volume': ticker.get('quoteVolume', 0),
                'vwap': ticker.get('vwap', 0),
                'change': ticker.get('change', 0),
                'percentage': ticker.get('percentage', 0),
            }

        except ccxt.BaseError as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}

    def fetch_all_data(
        self,
        symbol: str,
        timeframe: str = '1h',
        lookback_days: int = 30,
        include_orderbook: bool = True,
        include_ticker: bool = True
    ) -> MarketData:
        """
        Fetch all market data for a symbol.

        Args:
            symbol: Trading pair symbol
            timeframe: OHLCV timeframe
            lookback_days: Days of history
            include_orderbook: Whether to include orderbook
            include_ticker: Whether to include ticker

        Returns:
            MarketData object with all data
        """
        # Check cache
        cache_key = f"{symbol}_{timeframe}_{lookback_days}"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            age = (datetime.utcnow() - cached.timestamp).total_seconds()
            if age < self._cache_ttl:
                return cached

        # Fetch fresh data
        ohlcv = self.fetch_ohlcv(symbol, timeframe, lookback_days)

        orderbook = None
        if include_orderbook:
            orderbook = self.fetch_orderbook(symbol, self.config.data.orderbook_depth)

        ticker = None
        if include_ticker:
            ticker = self.fetch_ticker(symbol)

        market_data = MarketData(
            symbol=symbol,
            ohlcv=ohlcv,
            orderbook=orderbook,
            ticker=ticker,
        )

        # Update cache
        self._cache[cache_key] = market_data

        return market_data

    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        timeframe: str = '1h',
        lookback_days: int = 30,
        max_workers: int = 5
    ) -> Dict[str, MarketData]:
        """
        Fetch data for multiple symbols in parallel.

        Args:
            symbols: List of trading pair symbols
            timeframe: OHLCV timeframe
            lookback_days: Days of history
            max_workers: Number of parallel workers

        Returns:
            Dictionary mapping symbols to MarketData
        """
        results = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.fetch_all_data,
                    symbol,
                    timeframe,
                    lookback_days
                ): symbol
                for symbol in symbols
            }

            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    results[symbol] = future.result()
                except Exception as e:
                    logger.error(f"Error fetching data for {symbol}: {e}")
                    results[symbol] = MarketData(
                        symbol=symbol,
                        ohlcv=pd.DataFrame()
                    )

        return results

    def get_market_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get current prices for multiple symbols.

        Args:
            symbols: List of trading pair symbols

        Returns:
            Dictionary mapping symbols to current prices
        """
        prices = {}
        for symbol in symbols:
            ticker = self.fetch_ticker(symbol)
            if 'last' in ticker and ticker['last']:
                prices[symbol] = ticker['last']
        return prices

    def fetch_funding_rate(self, symbol: str) -> Optional[float]:
        """
        Fetch funding rate for perpetual contracts.

        Args:
            symbol: Trading pair symbol

        Returns:
            Current funding rate or None if not available
        """
        try:
            # This requires a futures/perp exchange
            if hasattr(self.exchange, 'fetch_funding_rate'):
                funding = self.exchange.fetch_funding_rate(symbol)
                return funding.get('fundingRate', None)
            return None
        except ccxt.BaseError as e:
            logger.error(f"Error fetching funding rate for {symbol}: {e}")
            return None

    def _timeframe_to_hours(self, timeframe: str) -> float:
        """Convert timeframe string to hours"""
        multipliers = {
            'm': 1/60,  # minutes
            'h': 1,     # hours
            'd': 24,    # days
            'w': 168,   # weeks
        }
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        return value * multipliers.get(unit, 1)

    def clear_cache(self):
        """Clear all cached data"""
        self._cache.clear()
        logger.info("Data cache cleared")

    def get_exchange_info(self) -> Dict[str, Any]:
        """Get exchange information"""
        return {
            'id': self.exchange.id,
            'name': self.exchange.name,
            'has': self.exchange.has,
            'timeframes': self.exchange.timeframes,
            'rate_limit': self.exchange.rateLimit,
        }
