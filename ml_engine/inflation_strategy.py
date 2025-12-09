"""
Inflation/Deflation Trading Strategy

Core Strategy:
- SHORT high-inflation assets (meme tokens, team/VC unlocks)
- LONG deflationary assets (buybacks, burns like HYPE)

This is a low-risk strategy based on:
1. Token unlock pressure (team, investors sell)
2. Buyback support (protocol buying back tokens)
3. Supply dynamics (inflation vs deflation)

Data Sources:
- Token Unlock Pipeline (tokenomist.ai, cryptorank.io)
- Buyback Analyzer (hypurrscan.io)
- Orderbook Pipeline (whale activity)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from ml_engine.token_unlock_pipeline import (
    TokenUnlockPipeline,
    TokenomicsProfile,
    UnlockAnalysis,
    UnlockRisk,
    AllocationCategory
)
from ml_engine.buyback_analyzer import (
    BuybackAnalyzer,
    DeflationaryAssetAnalyzer,
    BuybackAnalysis,
    BuybackSentiment
)
from ml_engine.orderbook_pipeline import (
    OrderbookPipeline,
    OrderbookAnalysis,
    ImbalanceSignal
)

logger = logging.getLogger(__name__)


class AssetInflationType(Enum):
    """Token inflation classification"""
    HIGHLY_INFLATIONARY = "highly_inflationary"   # > 5% monthly inflation
    INFLATIONARY = "inflationary"                  # 2-5% monthly inflation
    NEUTRAL = "neutral"                            # < 2% inflation, no buybacks
    DEFLATIONARY = "deflationary"                  # Active buyback/burn
    HIGHLY_DEFLATIONARY = "highly_deflationary"   # Large buyback activity


class TradingDirection(Enum):
    """Recommended trading direction"""
    STRONG_SHORT = "strong_short"
    SHORT = "short"
    NEUTRAL = "neutral"
    LONG = "long"
    STRONG_LONG = "strong_long"


@dataclass
class InflationSignal:
    """Trading signal based on inflation/deflation analysis"""
    symbol: str
    timestamp: datetime

    # Classification
    inflation_type: AssetInflationType
    direction: TradingDirection

    # Signal components
    unlock_signal: float = 0.0      # -1 (bearish) to 0 (neutral)
    buyback_signal: float = 0.0     # 0 (neutral) to 1 (bullish)
    orderbook_signal: float = 0.0   # -1 to 1

    # Combined signal
    combined_signal: float = 0.0    # -1 (strong short) to 1 (strong long)
    confidence: float = 0.0         # 0 to 1

    # Position sizing
    recommended_weight: float = 0.0  # -1 to 1 (negative = short)

    # Risk metrics
    risk_level: str = "medium"      # low, medium, high
    max_position_pct: float = 0.10  # Max position as % of portfolio

    # Analysis details
    unlock_analysis: Optional[UnlockAnalysis] = None
    buyback_analysis: Optional[BuybackAnalysis] = None
    orderbook_analysis: Optional[OrderbookAnalysis] = None

    # Notes
    analysis_notes: List[str] = field(default_factory=list)


@dataclass
class StrategyConfig:
    """Configuration for inflation/deflation strategy"""
    # Weights for signal combination
    unlock_weight: float = 0.40      # 40% unlock pressure
    buyback_weight: float = 0.35     # 35% buyback support
    orderbook_weight: float = 0.25   # 25% orderbook imbalance

    # Position limits
    max_single_position: float = 0.15     # 15% max per position
    max_short_exposure: float = 0.50      # 50% max short exposure
    max_long_exposure: float = 0.50       # 50% max long exposure
    max_net_exposure: float = 0.20        # 20% max net exposure

    # Signal thresholds
    min_signal_for_trade: float = 0.25    # Minimum signal to trade
    strong_signal_threshold: float = 0.60  # Threshold for strong signal

    # Risk parameters
    max_unlock_risk: UnlockRisk = UnlockRisk.CRITICAL  # Max acceptable unlock risk
    min_buyback_for_long: float = 0.30    # Min buyback signal to go long

    # Update intervals
    unlock_refresh_hours: int = 6         # Refresh unlock data every 6 hours
    buyback_refresh_minutes: int = 15     # Refresh buyback data every 15 min
    orderbook_refresh_seconds: int = 5    # Refresh orderbook every 5 seconds


class InflationDeflationStrategy:
    """
    Trading strategy based on token inflation/deflation dynamics

    Core Principle:
    - Tokens with upcoming large unlocks face sell pressure -> SHORT
    - Tokens with active buybacks have price support -> LONG
    - Low correlation to market direction -> Market neutral alpha
    """

    def __init__(
        self,
        config: Optional[StrategyConfig] = None,
        tokenomist_api_key: Optional[str] = None,
        cryptorank_api_key: Optional[str] = None
    ):
        self.config = config or StrategyConfig()

        # Initialize pipelines
        self.unlock_pipeline = TokenUnlockPipeline(
            tokenomist_api_key=tokenomist_api_key,
            cryptorank_api_key=cryptorank_api_key
        )
        self.buyback_analyzer = BuybackAnalyzer(
            tokenomist_api_key=tokenomist_api_key
        )
        self.deflationary_analyzer = DeflationaryAssetAnalyzer()

        # State
        self._unlock_cache: Dict[str, Tuple[datetime, TokenomicsProfile]] = {}
        self._buyback_cache: Dict[str, Tuple[datetime, BuybackAnalysis]] = {}
        self._signals: Dict[str, InflationSignal] = {}

        # Known token classifications
        self._meme_tokens = {
            "PEPE", "DOGE", "SHIB", "WIF", "BONK", "FLOKI",
            "MEME", "TURBO", "LADYS", "WOJAK"
        }
        self._deflationary_tokens = {
            "HYPE", "BNB", "ETH"  # Tokens with known buyback/burn mechanisms
        }

    async def analyze_symbol(
        self,
        symbol: str,
        current_price: float,
        orderbook_analysis: Optional[OrderbookAnalysis] = None
    ) -> InflationSignal:
        """
        Generate inflation/deflation trading signal for a symbol

        Args:
            symbol: Trading pair (e.g., 'ARB/USDT')
            current_price: Current token price
            orderbook_analysis: Optional orderbook data

        Returns:
            InflationSignal with trading recommendation
        """
        base_token = symbol.upper().replace('/USDT', '').replace('/USDC', '')
        now = datetime.utcnow()

        signal = InflationSignal(
            symbol=symbol,
            timestamp=now,
            inflation_type=AssetInflationType.NEUTRAL,
            direction=TradingDirection.NEUTRAL
        )

        # Step 1: Analyze token unlocks
        unlock_signal, unlock_analysis = await self._analyze_unlocks(
            symbol, current_price
        )
        signal.unlock_signal = unlock_signal
        signal.unlock_analysis = unlock_analysis

        if unlock_analysis:
            signal.analysis_notes.extend(unlock_analysis.analysis_notes)

        # Step 2: Analyze buybacks (if deflationary token)
        buyback_signal, buyback_analysis = await self._analyze_buybacks(
            symbol, current_price
        )
        signal.buyback_signal = buyback_signal
        signal.buyback_analysis = buyback_analysis

        if buyback_analysis:
            signal.analysis_notes.extend(buyback_analysis.analysis_notes)

        # Step 3: Include orderbook analysis if available
        if orderbook_analysis:
            signal.orderbook_signal = self._orderbook_to_signal(orderbook_analysis)
            signal.orderbook_analysis = orderbook_analysis
            signal.analysis_notes.extend(orderbook_analysis.analysis_notes)

        # Step 4: Combine signals
        signal = self._combine_signals(signal, base_token)

        # Store signal
        self._signals[symbol] = signal

        return signal

    async def _analyze_unlocks(
        self,
        symbol: str,
        current_price: float
    ) -> Tuple[float, Optional[UnlockAnalysis]]:
        """
        Analyze token unlock pressure

        Returns:
            Tuple of (signal, analysis)
            Signal: 0 (no pressure) to -1 (extreme sell pressure)
        """
        base_token = symbol.upper().replace('/USDT', '').replace('/USDC', '')

        # Check cache
        if base_token in self._unlock_cache:
            cache_time, cached_profile = self._unlock_cache[base_token]
            if (datetime.utcnow() - cache_time).total_seconds() < self.config.unlock_refresh_hours * 3600:
                analysis = self.unlock_pipeline.analyze_short_opportunity(
                    cached_profile, current_price
                )
                return -analysis.short_signal, analysis

        # Fetch fresh data
        profile = await self.unlock_pipeline.fetch_token_unlocks(base_token)

        if profile is None:
            return 0.0, None

        # Cache profile
        self._unlock_cache[base_token] = (datetime.utcnow(), profile)

        # Generate analysis
        analysis = self.unlock_pipeline.analyze_short_opportunity(profile, current_price)

        # Convert to signal (negative because unlocks = sell pressure)
        return -analysis.short_signal, analysis

    async def _analyze_buybacks(
        self,
        symbol: str,
        current_price: float
    ) -> Tuple[float, Optional[BuybackAnalysis]]:
        """
        Analyze buyback activity

        Returns:
            Tuple of (signal, analysis)
            Signal: 0 (no buybacks) to 1 (strong buyback support)
        """
        base_token = symbol.upper().replace('/USDT', '').replace('/USDC', '')

        # Only analyze known deflationary tokens
        if not self.deflationary_analyzer.is_deflationary(base_token):
            return 0.0, None

        # Special handling for HYPE
        if base_token == "HYPE":
            profile = await self.buyback_analyzer.fetch_hype_buybacks()
            analysis = self.buyback_analyzer.analyze_for_trading(profile, current_price)
            return analysis.long_signal, analysis

        # Generic deflationary analysis
        analysis = await self.deflationary_analyzer.analyze_deflationary_asset(
            symbol, current_price
        )
        if analysis:
            return analysis.long_signal, analysis

        return 0.0, None

    def _orderbook_to_signal(self, analysis: OrderbookAnalysis) -> float:
        """Convert orderbook imbalance to signal"""
        base_signal = analysis.depth_imbalance

        # Boost signal if strong consensus across exchanges
        if analysis.exchange_consensus:
            buy_count = sum(
                1 for s in analysis.exchange_consensus.values()
                if s in [ImbalanceSignal.BUY, ImbalanceSignal.STRONG_BUY]
            )
            sell_count = sum(
                1 for s in analysis.exchange_consensus.values()
                if s in [ImbalanceSignal.SELL, ImbalanceSignal.STRONG_SELL]
            )
            total = len(analysis.exchange_consensus)

            if buy_count >= total * 0.7:
                base_signal = max(base_signal, 0.5)
            elif sell_count >= total * 0.7:
                base_signal = min(base_signal, -0.5)

        return max(-1, min(1, base_signal))

    def _combine_signals(self, signal: InflationSignal, base_token: str) -> InflationSignal:
        """
        Combine individual signals into final trading recommendation

        Signal combination with weights:
        - Unlock signal: 40% (strongest predictor for shorts)
        - Buyback signal: 35% (strongest predictor for longs)
        - Orderbook signal: 25% (timing/confirmation)
        """
        # Apply weights
        weighted_unlock = signal.unlock_signal * self.config.unlock_weight
        weighted_buyback = signal.buyback_signal * self.config.buyback_weight
        weighted_orderbook = signal.orderbook_signal * self.config.orderbook_weight

        # Combined signal
        combined = weighted_unlock + weighted_buyback + weighted_orderbook
        signal.combined_signal = max(-1, min(1, combined))

        # Calculate confidence based on signal strength and data quality
        data_quality = 0.0
        if signal.unlock_analysis:
            data_quality += 0.4
        if signal.buyback_analysis:
            data_quality += 0.4
        if signal.orderbook_analysis:
            data_quality += 0.2

        signal.confidence = abs(signal.combined_signal) * data_quality

        # Determine inflation type
        signal.inflation_type = self._classify_inflation(signal, base_token)

        # Determine direction
        signal.direction = self._signal_to_direction(signal.combined_signal)

        # Calculate recommended position weight
        signal.recommended_weight = self._calculate_position_weight(signal)

        # Set risk level
        signal.risk_level = self._assess_risk(signal, base_token)

        # Set max position based on risk
        signal.max_position_pct = self._get_max_position(signal)

        return signal

    def _classify_inflation(
        self,
        signal: InflationSignal,
        base_token: str
    ) -> AssetInflationType:
        """Classify token as inflationary or deflationary"""
        # Meme tokens are always considered inflationary
        if base_token in self._meme_tokens:
            return AssetInflationType.HIGHLY_INFLATIONARY

        # Known deflationary tokens
        if base_token in self._deflationary_tokens:
            if signal.buyback_signal >= 0.6:
                return AssetInflationType.HIGHLY_DEFLATIONARY
            elif signal.buyback_signal >= 0.3:
                return AssetInflationType.DEFLATIONARY

        # Based on unlock analysis
        if signal.unlock_analysis:
            unlock_pct_30d = signal.unlock_analysis.unlock_30d_pct

            if unlock_pct_30d >= 5:
                return AssetInflationType.HIGHLY_INFLATIONARY
            elif unlock_pct_30d >= 2:
                return AssetInflationType.INFLATIONARY

        return AssetInflationType.NEUTRAL

    def _signal_to_direction(self, combined_signal: float) -> TradingDirection:
        """Convert numeric signal to trading direction"""
        if combined_signal <= -self.config.strong_signal_threshold:
            return TradingDirection.STRONG_SHORT
        elif combined_signal <= -self.config.min_signal_for_trade:
            return TradingDirection.SHORT
        elif combined_signal >= self.config.strong_signal_threshold:
            return TradingDirection.STRONG_LONG
        elif combined_signal >= self.config.min_signal_for_trade:
            return TradingDirection.LONG
        else:
            return TradingDirection.NEUTRAL

    def _calculate_position_weight(self, signal: InflationSignal) -> float:
        """Calculate recommended position weight (-1 to 1)"""
        if abs(signal.combined_signal) < self.config.min_signal_for_trade:
            return 0.0

        # Scale position by signal strength and confidence
        base_weight = signal.combined_signal * signal.confidence

        # Cap to max single position
        return max(
            -self.config.max_single_position,
            min(self.config.max_single_position, base_weight)
        )

    def _assess_risk(self, signal: InflationSignal, base_token: str) -> str:
        """Assess risk level of the signal"""
        # Meme tokens are always high risk
        if base_token in self._meme_tokens:
            return "high"

        # Strong deflationary with buybacks = lower risk
        if signal.inflation_type == AssetInflationType.HIGHLY_DEFLATIONARY:
            return "low"

        # Large unlock risk
        if signal.unlock_analysis and signal.unlock_analysis.risk_level in [
            UnlockRisk.CRITICAL, UnlockRisk.HIGH
        ]:
            return "high" if signal.direction in [TradingDirection.LONG, TradingDirection.STRONG_LONG] else "medium"

        # Default medium risk
        return "medium"

    def _get_max_position(self, signal: InflationSignal) -> float:
        """Get maximum position size based on risk"""
        risk_limits = {
            "low": self.config.max_single_position,
            "medium": self.config.max_single_position * 0.75,
            "high": self.config.max_single_position * 0.5
        }
        return risk_limits.get(signal.risk_level, self.config.max_single_position * 0.5)

    async def screen_universe(
        self,
        symbols: List[str],
        prices: Dict[str, float],
        orderbook_analyses: Optional[Dict[str, OrderbookAnalysis]] = None
    ) -> Dict[str, InflationSignal]:
        """
        Screen entire asset universe for inflation/deflation signals

        Args:
            symbols: List of symbols to screen
            prices: Current prices
            orderbook_analyses: Optional orderbook data

        Returns:
            Dict of symbol -> InflationSignal
        """
        signals = {}

        for symbol in symbols:
            price = prices.get(symbol, 0)
            if price <= 0:
                continue

            ob_analysis = orderbook_analyses.get(symbol) if orderbook_analyses else None

            try:
                signal = await self.analyze_symbol(symbol, price, ob_analysis)
                signals[symbol] = signal
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")

        return signals

    def get_portfolio_recommendations(
        self,
        signals: Dict[str, InflationSignal]
    ) -> Dict[str, Any]:
        """
        Generate portfolio-level recommendations

        Returns:
            Dict with short/long targets and exposure limits
        """
        shorts = []
        longs = []
        neutral = []

        for symbol, signal in signals.items():
            if signal.direction in [TradingDirection.SHORT, TradingDirection.STRONG_SHORT]:
                shorts.append((symbol, signal))
            elif signal.direction in [TradingDirection.LONG, TradingDirection.STRONG_LONG]:
                longs.append((symbol, signal))
            else:
                neutral.append((symbol, signal))

        # Sort by signal strength
        shorts.sort(key=lambda x: x[1].combined_signal)  # Most negative first
        longs.sort(key=lambda x: x[1].combined_signal, reverse=True)  # Most positive first

        # Calculate total exposures
        total_short = sum(abs(s[1].recommended_weight) for s in shorts)
        total_long = sum(s[1].recommended_weight for s in longs)

        # Scale if exceeding limits
        short_scale = min(1.0, self.config.max_short_exposure / total_short) if total_short > 0 else 1.0
        long_scale = min(1.0, self.config.max_long_exposure / total_long) if total_long > 0 else 1.0

        return {
            "shorts": [
                {
                    "symbol": symbol,
                    "weight": signal.recommended_weight * short_scale,
                    "signal": signal.combined_signal,
                    "confidence": signal.confidence,
                    "inflation_type": signal.inflation_type.value,
                    "risk": signal.risk_level,
                    "notes": signal.analysis_notes[:3]  # Top 3 notes
                }
                for symbol, signal in shorts
            ],
            "longs": [
                {
                    "symbol": symbol,
                    "weight": signal.recommended_weight * long_scale,
                    "signal": signal.combined_signal,
                    "confidence": signal.confidence,
                    "inflation_type": signal.inflation_type.value,
                    "risk": signal.risk_level,
                    "notes": signal.analysis_notes[:3]
                }
                for symbol, signal in longs
            ],
            "summary": {
                "total_short_exposure": total_short * short_scale,
                "total_long_exposure": total_long * long_scale,
                "net_exposure": (total_long * long_scale) - (total_short * short_scale),
                "num_shorts": len(shorts),
                "num_longs": len(longs),
                "num_neutral": len(neutral)
            }
        }

    def get_unlock_calendar(self, days_ahead: int = 14) -> List[Dict]:
        """
        Get upcoming unlock events from cached data

        Returns list of unlock events sorted by date
        """
        events = []

        for token, (_, profile) in self._unlock_cache.items():
            for unlock in profile.upcoming_unlocks:
                if (unlock.unlock_date - datetime.utcnow()).days <= days_ahead:
                    events.append({
                        "token": token,
                        "date": unlock.unlock_date.isoformat(),
                        "amount_pct": unlock.percentage_of_supply,
                        "category": unlock.category.value,
                        "beneficiary": unlock.beneficiary
                    })

        events.sort(key=lambda x: x["date"])
        return events

    def should_short(self, signal: InflationSignal) -> bool:
        """Check if signal recommends shorting"""
        return signal.direction in [TradingDirection.SHORT, TradingDirection.STRONG_SHORT]

    def should_long(self, signal: InflationSignal) -> bool:
        """Check if signal recommends longing"""
        return signal.direction in [TradingDirection.LONG, TradingDirection.STRONG_LONG]

    def get_position_size(
        self,
        signal: InflationSignal,
        portfolio_value: float,
        current_exposure: float
    ) -> float:
        """
        Calculate actual position size in USD

        Args:
            signal: Trading signal
            portfolio_value: Total portfolio value
            current_exposure: Current exposure in same direction

        Returns:
            Position size in USD (negative for shorts)
        """
        # Base position from signal
        base_size = portfolio_value * signal.recommended_weight

        # Adjust for existing exposure
        remaining_capacity = (
            self.config.max_short_exposure if base_size < 0 else self.config.max_long_exposure
        ) * portfolio_value - current_exposure

        if remaining_capacity <= 0:
            return 0.0

        return max(min(base_size, remaining_capacity), -remaining_capacity)
