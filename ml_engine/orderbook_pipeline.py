"""
Orderbook Scraping Pipeline - Real-time orderbook analysis

Data Sources:
- Bybit WebSocket API
- Binance WebSocket API
- Hyperliquid WebSocket API
- Lighter Exchange API

Features:
- Large order block detection
- Bid/ask imbalance analysis
- Liquidity depth monitoring
- Whale activity tracking
"""

import asyncio
import aiohttp
import websockets
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


class Exchange(Enum):
    """Supported exchanges"""
    BYBIT = "bybit"
    BINANCE = "binance"
    HYPERLIQUID = "hyperliquid"
    LIGHTER = "lighter"


class OrderBlockType(Enum):
    """Type of order block"""
    BID_WALL = "bid_wall"      # Large buy wall
    ASK_WALL = "ask_wall"      # Large sell wall
    ICEBERG_BID = "iceberg_bid"  # Hidden buy orders
    ICEBERG_ASK = "iceberg_ask"  # Hidden sell orders
    SPOOFING = "spoofing"      # Likely spoofing detected


class ImbalanceSignal(Enum):
    """Order book imbalance signal"""
    STRONG_BUY = "strong_buy"     # Heavy bid imbalance
    BUY = "buy"                   # Moderate bid imbalance
    NEUTRAL = "neutral"          # Balanced
    SELL = "sell"                # Moderate ask imbalance
    STRONG_SELL = "strong_sell"  # Heavy ask imbalance


@dataclass
class OrderLevel:
    """Single price level in orderbook"""
    price: float
    quantity: float
    total_value: float = 0.0
    num_orders: int = 1  # If available

    def __post_init__(self):
        self.total_value = self.price * self.quantity


@dataclass
class OrderBook:
    """Orderbook snapshot"""
    exchange: Exchange
    symbol: str
    timestamp: datetime
    bids: List[OrderLevel] = field(default_factory=list)  # Sorted high to low
    asks: List[OrderLevel] = field(default_factory=list)  # Sorted low to high

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None

    @property
    def spread_bps(self) -> Optional[float]:
        """Spread in basis points"""
        if self.mid_price and self.spread:
            return (self.spread / self.mid_price) * 10000
        return None


@dataclass
class OrderBlock:
    """Significant order block detected"""
    exchange: Exchange
    symbol: str
    timestamp: datetime
    block_type: OrderBlockType
    price: float
    quantity: float
    value_usd: float
    price_distance_pct: float  # Distance from current price
    significance_score: float  # 0-1, higher = more significant


@dataclass
class OrderbookAnalysis:
    """Analysis result from orderbook data"""
    symbol: str
    timestamp: datetime

    # Best prices
    best_bid: float = 0.0
    best_ask: float = 0.0
    mid_price: float = 0.0
    spread_bps: float = 0.0

    # Depth analysis
    bid_depth_usd: float = 0.0  # Total bid depth in USD
    ask_depth_usd: float = 0.0  # Total ask depth in USD
    depth_imbalance: float = 0.0  # -1 to 1, positive = more bids

    # Order blocks
    detected_blocks: List[OrderBlock] = field(default_factory=list)
    largest_bid_wall: Optional[OrderBlock] = None
    largest_ask_wall: Optional[OrderBlock] = None

    # Signals
    imbalance_signal: ImbalanceSignal = ImbalanceSignal.NEUTRAL
    signal_strength: float = 0.0  # 0 to 1

    # Multi-exchange consensus
    exchange_consensus: Dict[str, ImbalanceSignal] = field(default_factory=dict)

    # Notes
    analysis_notes: List[str] = field(default_factory=list)


class OrderbookPipeline:
    """
    Pipeline for real-time orderbook analysis across multiple exchanges

    Features:
    - WebSocket connections to Bybit, Binance, Hyperliquid
    - REST API fallback for Lighter
    - Large order block detection
    - Bid/ask imbalance signals
    """

    # WebSocket endpoints
    WEBSOCKET_URLS = {
        Exchange.BYBIT: "wss://stream.bybit.com/v5/public/linear",
        Exchange.BINANCE: "wss://fstream.binance.com/stream",
        Exchange.HYPERLIQUID: "wss://api.hyperliquid.xyz/ws",
    }

    # REST API endpoints (fallback)
    REST_URLS = {
        Exchange.BYBIT: "https://api.bybit.com/v5/market/orderbook",
        Exchange.BINANCE: "https://fapi.binance.com/fapi/v1/depth",
        Exchange.HYPERLIQUID: "https://api.hyperliquid.xyz/info",
        Exchange.LIGHTER: "https://api.lighter.xyz/orderbook",
    }

    # Detection thresholds
    LARGE_ORDER_USD = 100_000      # $100K+ = large order
    WHALE_ORDER_USD = 1_000_000    # $1M+ = whale order
    WALL_THRESHOLD_PCT = 5.0       # 5% of visible depth = wall

    def __init__(
        self,
        symbols: List[str],
        exchanges: List[Exchange] = None,
        depth_levels: int = 50,
        update_interval: float = 1.0
    ):
        self.symbols = symbols
        self.exchanges = exchanges or [Exchange.BYBIT, Exchange.BINANCE, Exchange.HYPERLIQUID]
        self.depth_levels = depth_levels
        self.update_interval = update_interval

        # State
        self._orderbooks: Dict[str, Dict[Exchange, OrderBook]] = defaultdict(dict)
        self._websocket_tasks: List[asyncio.Task] = []
        self._running = False
        self._callbacks: List[Callable] = []

        # Historical data for pattern detection
        self._depth_history: Dict[str, List[Tuple[datetime, float, float]]] = defaultdict(list)

    async def start(self):
        """Start orderbook pipeline with WebSocket connections"""
        self._running = True
        logger.info(f"Starting orderbook pipeline for {len(self.symbols)} symbols on {len(self.exchanges)} exchanges")

        # Start WebSocket connections for each exchange
        for exchange in self.exchanges:
            if exchange in self.WEBSOCKET_URLS:
                task = asyncio.create_task(self._websocket_loop(exchange))
                self._websocket_tasks.append(task)
            else:
                # Use REST polling for exchanges without WebSocket
                task = asyncio.create_task(self._rest_polling_loop(exchange))
                self._websocket_tasks.append(task)

    async def stop(self):
        """Stop orderbook pipeline"""
        self._running = False
        for task in self._websocket_tasks:
            task.cancel()
        self._websocket_tasks.clear()

    def add_callback(self, callback: Callable[[OrderbookAnalysis], None]):
        """Add callback for orderbook updates"""
        self._callbacks.append(callback)

    async def _websocket_loop(self, exchange: Exchange):
        """WebSocket connection loop for an exchange"""
        url = self.WEBSOCKET_URLS[exchange]
        reconnect_delay = 1

        while self._running:
            try:
                async with websockets.connect(url) as ws:
                    logger.info(f"Connected to {exchange.value} WebSocket")
                    reconnect_delay = 1

                    # Subscribe to orderbook channels
                    await self._subscribe_orderbooks(ws, exchange)

                    # Process messages
                    async for message in ws:
                        if not self._running:
                            break
                        await self._process_ws_message(exchange, message)

            except websockets.ConnectionClosed:
                logger.warning(f"{exchange.value} WebSocket closed, reconnecting in {reconnect_delay}s")
            except Exception as e:
                logger.error(f"{exchange.value} WebSocket error: {e}")

            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 60)

    async def _subscribe_orderbooks(self, ws, exchange: Exchange):
        """Subscribe to orderbook channels"""
        if exchange == Exchange.BYBIT:
            for symbol in self.symbols:
                bybit_symbol = self._convert_symbol(symbol, exchange)
                msg = {
                    "op": "subscribe",
                    "args": [f"orderbook.{self.depth_levels}.{bybit_symbol}"]
                }
                await ws.send(json.dumps(msg))

        elif exchange == Exchange.BINANCE:
            streams = []
            for symbol in self.symbols:
                binance_symbol = self._convert_symbol(symbol, exchange)
                streams.append(f"{binance_symbol.lower()}@depth{self.depth_levels}@100ms")
            msg = {
                "method": "SUBSCRIBE",
                "params": streams,
                "id": 1
            }
            await ws.send(json.dumps(msg))

        elif exchange == Exchange.HYPERLIQUID:
            for symbol in self.symbols:
                hl_symbol = self._convert_symbol(symbol, exchange)
                msg = {
                    "method": "subscribe",
                    "subscription": {
                        "type": "l2Book",
                        "coin": hl_symbol
                    }
                }
                await ws.send(json.dumps(msg))

    async def _process_ws_message(self, exchange: Exchange, message: str):
        """Process WebSocket message from exchange"""
        try:
            data = json.loads(message)
            orderbook = self._parse_orderbook_message(exchange, data)

            if orderbook:
                # Store orderbook
                self._orderbooks[orderbook.symbol][exchange] = orderbook

                # Update depth history
                if orderbook.bids and orderbook.asks:
                    bid_depth = sum(b.total_value for b in orderbook.bids)
                    ask_depth = sum(a.total_value for a in orderbook.asks)
                    self._depth_history[orderbook.symbol].append(
                        (orderbook.timestamp, bid_depth, ask_depth)
                    )
                    # Keep last 1000 updates
                    if len(self._depth_history[orderbook.symbol]) > 1000:
                        self._depth_history[orderbook.symbol] = self._depth_history[orderbook.symbol][-1000:]

                # Trigger analysis and callbacks
                analysis = self.analyze_orderbook(orderbook.symbol)
                for callback in self._callbacks:
                    try:
                        callback(analysis)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")

        except json.JSONDecodeError:
            pass
        except Exception as e:
            logger.error(f"Error processing {exchange.value} message: {e}")

    def _parse_orderbook_message(self, exchange: Exchange, data: Dict) -> Optional[OrderBook]:
        """Parse exchange-specific orderbook message"""
        try:
            if exchange == Exchange.BYBIT:
                return self._parse_bybit_orderbook(data)
            elif exchange == Exchange.BINANCE:
                return self._parse_binance_orderbook(data)
            elif exchange == Exchange.HYPERLIQUID:
                return self._parse_hyperliquid_orderbook(data)
        except Exception as e:
            logger.debug(f"Parse error for {exchange.value}: {e}")
        return None

    def _parse_bybit_orderbook(self, data: Dict) -> Optional[OrderBook]:
        """Parse Bybit orderbook message"""
        if 'topic' not in data or 'orderbook' not in data.get('topic', ''):
            return None

        book_data = data.get('data', {})
        symbol = book_data.get('s', '').replace('USDT', '/USDT')

        bids = [
            OrderLevel(float(b[0]), float(b[1]))
            for b in book_data.get('b', [])
        ]
        asks = [
            OrderLevel(float(a[0]), float(a[1]))
            for a in book_data.get('a', [])
        ]

        return OrderBook(
            exchange=Exchange.BYBIT,
            symbol=symbol,
            timestamp=datetime.utcnow(),
            bids=sorted(bids, key=lambda x: x.price, reverse=True),
            asks=sorted(asks, key=lambda x: x.price)
        )

    def _parse_binance_orderbook(self, data: Dict) -> Optional[OrderBook]:
        """Parse Binance orderbook message"""
        if 'stream' not in data:
            return None

        stream = data.get('stream', '')
        if '@depth' not in stream:
            return None

        symbol = stream.split('@')[0].upper()
        symbol = symbol.replace('USDT', '/USDT')

        book_data = data.get('data', {})
        bids = [
            OrderLevel(float(b[0]), float(b[1]))
            for b in book_data.get('b', [])
        ]
        asks = [
            OrderLevel(float(a[0]), float(a[1]))
            for a in book_data.get('a', [])
        ]

        return OrderBook(
            exchange=Exchange.BINANCE,
            symbol=symbol,
            timestamp=datetime.utcnow(),
            bids=sorted(bids, key=lambda x: x.price, reverse=True),
            asks=sorted(asks, key=lambda x: x.price)
        )

    def _parse_hyperliquid_orderbook(self, data: Dict) -> Optional[OrderBook]:
        """Parse Hyperliquid orderbook message"""
        if data.get('channel') != 'l2Book':
            return None

        book_data = data.get('data', {})
        symbol = book_data.get('coin', '') + '/USDT'

        levels = book_data.get('levels', [[], []])
        bids = [
            OrderLevel(float(b['px']), float(b['sz']))
            for b in levels[0] if 'px' in b
        ]
        asks = [
            OrderLevel(float(a['px']), float(a['sz']))
            for a in levels[1] if 'px' in a
        ]

        return OrderBook(
            exchange=Exchange.HYPERLIQUID,
            symbol=symbol,
            timestamp=datetime.utcnow(),
            bids=sorted(bids, key=lambda x: x.price, reverse=True),
            asks=sorted(asks, key=lambda x: x.price)
        )

    async def _rest_polling_loop(self, exchange: Exchange):
        """REST API polling loop for exchanges without WebSocket"""
        while self._running:
            for symbol in self.symbols:
                try:
                    orderbook = await self._fetch_rest_orderbook(exchange, symbol)
                    if orderbook:
                        self._orderbooks[symbol][exchange] = orderbook
                except Exception as e:
                    logger.error(f"REST fetch error for {symbol} on {exchange.value}: {e}")

            await asyncio.sleep(self.update_interval)

    async def _fetch_rest_orderbook(self, exchange: Exchange, symbol: str) -> Optional[OrderBook]:
        """Fetch orderbook via REST API"""
        url = self.REST_URLS.get(exchange)
        if not url:
            return None

        try:
            async with aiohttp.ClientSession() as session:
                params = self._get_rest_params(exchange, symbol)
                async with session.get(url, params=params, timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return self._parse_rest_orderbook(exchange, symbol, data)
        except Exception as e:
            logger.error(f"REST API error for {exchange.value}: {e}")

        return None

    def _get_rest_params(self, exchange: Exchange, symbol: str) -> Dict:
        """Get REST API parameters for exchange"""
        converted = self._convert_symbol(symbol, exchange)

        if exchange == Exchange.BYBIT:
            return {"category": "linear", "symbol": converted, "limit": self.depth_levels}
        elif exchange == Exchange.BINANCE:
            return {"symbol": converted, "limit": self.depth_levels}
        elif exchange == Exchange.HYPERLIQUID:
            return {"type": "l2Book", "coin": converted}
        elif exchange == Exchange.LIGHTER:
            return {"market": converted, "depth": self.depth_levels}

        return {}

    def _parse_rest_orderbook(self, exchange: Exchange, symbol: str, data: Dict) -> Optional[OrderBook]:
        """Parse REST API orderbook response"""
        bids = []
        asks = []

        if exchange == Exchange.BYBIT:
            result = data.get('result', {})
            bids = [OrderLevel(float(b[0]), float(b[1])) for b in result.get('b', [])]
            asks = [OrderLevel(float(a[0]), float(a[1])) for a in result.get('a', [])]

        elif exchange == Exchange.BINANCE:
            bids = [OrderLevel(float(b[0]), float(b[1])) for b in data.get('bids', [])]
            asks = [OrderLevel(float(a[0]), float(a[1])) for a in data.get('asks', [])]

        elif exchange == Exchange.LIGHTER:
            bids = [OrderLevel(float(b['price']), float(b['size'])) for b in data.get('bids', [])]
            asks = [OrderLevel(float(a['price']), float(a['size'])) for a in data.get('asks', [])]

        if not bids and not asks:
            return None

        return OrderBook(
            exchange=exchange,
            symbol=symbol,
            timestamp=datetime.utcnow(),
            bids=sorted(bids, key=lambda x: x.price, reverse=True),
            asks=sorted(asks, key=lambda x: x.price)
        )

    def _convert_symbol(self, symbol: str, exchange: Exchange) -> str:
        """Convert standard symbol to exchange format"""
        # Standard format: BTC/USDT
        base = symbol.replace('/USDT', '').replace('/USDC', '')

        if exchange == Exchange.BYBIT:
            return f"{base}USDT"
        elif exchange == Exchange.BINANCE:
            return f"{base}USDT"
        elif exchange == Exchange.HYPERLIQUID:
            return base
        elif exchange == Exchange.LIGHTER:
            return f"{base}-USDC"

        return symbol

    def analyze_orderbook(self, symbol: str) -> OrderbookAnalysis:
        """
        Analyze orderbook data for a symbol across all exchanges

        Returns comprehensive analysis including:
        - Depth imbalance
        - Large order blocks
        - Trading signals
        """
        analysis = OrderbookAnalysis(
            symbol=symbol,
            timestamp=datetime.utcnow()
        )

        exchange_books = self._orderbooks.get(symbol, {})
        if not exchange_books:
            return analysis

        # Aggregate data across exchanges
        total_bid_depth = 0.0
        total_ask_depth = 0.0
        all_bids: List[OrderLevel] = []
        all_asks: List[OrderLevel] = []

        for exchange, book in exchange_books.items():
            if book.bids:
                all_bids.extend(book.bids)
                total_bid_depth += sum(b.total_value for b in book.bids)
            if book.asks:
                all_asks.extend(book.asks)
                total_ask_depth += sum(a.total_value for a in book.asks)

            # Per-exchange imbalance
            if book.bids and book.asks:
                ex_bid = sum(b.total_value for b in book.bids[:10])
                ex_ask = sum(a.total_value for a in book.asks[:10])
                if ex_bid + ex_ask > 0:
                    ex_imbalance = (ex_bid - ex_ask) / (ex_bid + ex_ask)
                    analysis.exchange_consensus[exchange.value] = self._imbalance_to_signal(ex_imbalance)

        # Set best prices from most recent book
        recent_book = list(exchange_books.values())[0]
        analysis.best_bid = recent_book.best_bid or 0
        analysis.best_ask = recent_book.best_ask or 0
        analysis.mid_price = recent_book.mid_price or 0
        analysis.spread_bps = recent_book.spread_bps or 0

        # Calculate overall depth imbalance
        analysis.bid_depth_usd = total_bid_depth
        analysis.ask_depth_usd = total_ask_depth

        if total_bid_depth + total_ask_depth > 0:
            analysis.depth_imbalance = (total_bid_depth - total_ask_depth) / (total_bid_depth + total_ask_depth)

        # Detect order blocks
        analysis.detected_blocks = self._detect_order_blocks(symbol, all_bids, all_asks, analysis.mid_price)

        # Find largest walls
        bid_blocks = [b for b in analysis.detected_blocks if b.block_type == OrderBlockType.BID_WALL]
        ask_blocks = [b for b in analysis.detected_blocks if b.block_type == OrderBlockType.ASK_WALL]

        if bid_blocks:
            analysis.largest_bid_wall = max(bid_blocks, key=lambda x: x.value_usd)
        if ask_blocks:
            analysis.largest_ask_wall = max(ask_blocks, key=lambda x: x.value_usd)

        # Calculate trading signal
        analysis.imbalance_signal = self._imbalance_to_signal(analysis.depth_imbalance)
        analysis.signal_strength = abs(analysis.depth_imbalance)

        # Add analysis notes
        if analysis.largest_bid_wall:
            analysis.analysis_notes.append(
                f"Bid wall: ${analysis.largest_bid_wall.value_usd/1000:.0f}K at ${analysis.largest_bid_wall.price:.2f}"
            )
        if analysis.largest_ask_wall:
            analysis.analysis_notes.append(
                f"Ask wall: ${analysis.largest_ask_wall.value_usd/1000:.0f}K at ${analysis.largest_ask_wall.price:.2f}"
            )

        # Multi-exchange consensus
        if len(analysis.exchange_consensus) > 1:
            buy_votes = sum(1 for s in analysis.exchange_consensus.values()
                          if s in [ImbalanceSignal.BUY, ImbalanceSignal.STRONG_BUY])
            sell_votes = sum(1 for s in analysis.exchange_consensus.values()
                           if s in [ImbalanceSignal.SELL, ImbalanceSignal.STRONG_SELL])
            total = len(analysis.exchange_consensus)

            if buy_votes > total / 2:
                analysis.analysis_notes.append(f"Exchange consensus: BUY ({buy_votes}/{total})")
            elif sell_votes > total / 2:
                analysis.analysis_notes.append(f"Exchange consensus: SELL ({sell_votes}/{total})")

        return analysis

    def _detect_order_blocks(
        self,
        symbol: str,
        bids: List[OrderLevel],
        asks: List[OrderLevel],
        mid_price: float
    ) -> List[OrderBlock]:
        """Detect significant order blocks (walls)"""
        blocks = []

        if not mid_price or mid_price <= 0:
            return blocks

        # Sort by size to find largest orders
        sorted_bids = sorted(bids, key=lambda x: x.total_value, reverse=True)
        sorted_asks = sorted(asks, key=lambda x: x.total_value, reverse=True)

        # Total depth for significance calculation
        total_bid_value = sum(b.total_value for b in bids)
        total_ask_value = sum(a.total_value for a in asks)

        # Check top bids for walls
        for bid in sorted_bids[:20]:
            if bid.total_value >= self.LARGE_ORDER_USD:
                significance = bid.total_value / total_bid_value if total_bid_value > 0 else 0
                price_dist = (mid_price - bid.price) / mid_price * 100

                if significance >= self.WALL_THRESHOLD_PCT / 100:
                    blocks.append(OrderBlock(
                        exchange=Exchange.BYBIT,  # Will be updated with actual exchange
                        symbol=symbol,
                        timestamp=datetime.utcnow(),
                        block_type=OrderBlockType.BID_WALL,
                        price=bid.price,
                        quantity=bid.quantity,
                        value_usd=bid.total_value,
                        price_distance_pct=price_dist,
                        significance_score=min(significance * 10, 1.0)
                    ))

        # Check top asks for walls
        for ask in sorted_asks[:20]:
            if ask.total_value >= self.LARGE_ORDER_USD:
                significance = ask.total_value / total_ask_value if total_ask_value > 0 else 0
                price_dist = (ask.price - mid_price) / mid_price * 100

                if significance >= self.WALL_THRESHOLD_PCT / 100:
                    blocks.append(OrderBlock(
                        exchange=Exchange.BYBIT,
                        symbol=symbol,
                        timestamp=datetime.utcnow(),
                        block_type=OrderBlockType.ASK_WALL,
                        price=ask.price,
                        quantity=ask.quantity,
                        value_usd=ask.total_value,
                        price_distance_pct=price_dist,
                        significance_score=min(significance * 10, 1.0)
                    ))

        return blocks

    def _imbalance_to_signal(self, imbalance: float) -> ImbalanceSignal:
        """Convert numeric imbalance to signal"""
        if imbalance >= 0.5:
            return ImbalanceSignal.STRONG_BUY
        elif imbalance >= 0.2:
            return ImbalanceSignal.BUY
        elif imbalance <= -0.5:
            return ImbalanceSignal.STRONG_SELL
        elif imbalance <= -0.2:
            return ImbalanceSignal.SELL
        else:
            return ImbalanceSignal.NEUTRAL

    def get_current_analysis(self, symbol: str) -> Optional[OrderbookAnalysis]:
        """Get current orderbook analysis for a symbol"""
        if symbol in self._orderbooks:
            return self.analyze_orderbook(symbol)
        return None

    def get_all_analyses(self) -> Dict[str, OrderbookAnalysis]:
        """Get orderbook analysis for all tracked symbols"""
        return {
            symbol: self.analyze_orderbook(symbol)
            for symbol in self._orderbooks.keys()
        }

    def get_depth_stats(self, symbol: str, lookback_minutes: int = 60) -> Dict[str, float]:
        """
        Get depth statistics over time

        Returns average, min, max bid/ask depth
        """
        history = self._depth_history.get(symbol, [])
        if not history:
            return {}

        cutoff = datetime.utcnow() - timedelta(minutes=lookback_minutes)
        recent = [(ts, bid, ask) for ts, bid, ask in history if ts >= cutoff]

        if not recent:
            return {}

        bid_depths = [bid for _, bid, _ in recent]
        ask_depths = [ask for _, _, ask in recent]

        return {
            "avg_bid_depth": statistics.mean(bid_depths),
            "avg_ask_depth": statistics.mean(ask_depths),
            "min_bid_depth": min(bid_depths),
            "max_bid_depth": max(bid_depths),
            "min_ask_depth": min(ask_depths),
            "max_ask_depth": max(ask_depths),
            "depth_volatility": statistics.stdev(bid_depths) if len(bid_depths) > 1 else 0
        }


class OrderBlockDetector:
    """
    Specialized detector for large order blocks and whale activity

    Used for:
    - Identifying support/resistance from large orders
    - Detecting whale accumulation/distribution
    - Spoofing detection
    """

    def __init__(
        self,
        whale_threshold_usd: float = 1_000_000,
        wall_threshold_pct: float = 5.0
    ):
        self.whale_threshold = whale_threshold_usd
        self.wall_threshold = wall_threshold_pct / 100

        # Track historical blocks for pattern detection
        self._block_history: Dict[str, List[OrderBlock]] = defaultdict(list)

    def analyze_whale_activity(
        self,
        analysis: OrderbookAnalysis
    ) -> Dict[str, Any]:
        """
        Analyze orderbook for whale activity patterns

        Returns:
            Dict with whale activity indicators
        """
        result = {
            "whale_buying": False,
            "whale_selling": False,
            "accumulation_signal": 0.0,
            "distribution_signal": 0.0,
            "whale_blocks": []
        }

        # Find whale-sized blocks
        whale_bids = [b for b in analysis.detected_blocks
                     if b.block_type == OrderBlockType.BID_WALL and b.value_usd >= self.whale_threshold]
        whale_asks = [b for b in analysis.detected_blocks
                     if b.block_type == OrderBlockType.ASK_WALL and b.value_usd >= self.whale_threshold]

        result["whale_blocks"] = whale_bids + whale_asks

        # Calculate accumulation/distribution signals
        whale_bid_value = sum(b.value_usd for b in whale_bids)
        whale_ask_value = sum(a.value_usd for a in whale_asks)

        total_whale = whale_bid_value + whale_ask_value

        if total_whale > 0:
            result["accumulation_signal"] = whale_bid_value / total_whale
            result["distribution_signal"] = whale_ask_value / total_whale
            result["whale_buying"] = whale_bid_value > whale_ask_value * 1.5
            result["whale_selling"] = whale_ask_value > whale_bid_value * 1.5

        return result

    def detect_spoofing(
        self,
        current_analysis: OrderbookAnalysis,
        previous_analyses: List[OrderbookAnalysis]
    ) -> List[OrderBlock]:
        """
        Detect potential spoofing (orders that disappear)

        Compares current orderbook to previous snapshots
        """
        spoofed_blocks = []

        if not previous_analyses:
            return spoofed_blocks

        # Get blocks from 1 minute ago
        prev = previous_analyses[-1] if previous_analyses else None
        if not prev:
            return spoofed_blocks

        # Check if large blocks disappeared without being filled
        prev_blocks = {(b.price, b.block_type): b for b in prev.detected_blocks}
        curr_blocks = {(b.price, b.block_type): b for b in current_analysis.detected_blocks}

        for key, prev_block in prev_blocks.items():
            if key not in curr_blocks and prev_block.value_usd >= self.whale_threshold:
                # Block disappeared - possible spoof
                prev_block.block_type = OrderBlockType.SPOOFING
                spoofed_blocks.append(prev_block)

        return spoofed_blocks
