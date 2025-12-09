"""
Buyback Analyzer Pipeline - Analyzes token buyback programs

Primary focus: HYPE (Hyperliquid) buybacks from Assistance Fund

Data Sources:
- HypurrScan.io (HYPE buybacks)
- Tokenomist.ai (general buyback data)
- On-chain data

Strategy:
- Track buyback activity to identify bullish periods
- Increase long position sizing when buybacks are active
- Reduce risk when buyback activity is low
"""

import asyncio
import aiohttp
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
import json

logger = logging.getLogger(__name__)


class BuybackSentiment(Enum):
    """Buyback activity sentiment"""
    VERY_BULLISH = "very_bullish"   # Large buybacks in last 24h
    BULLISH = "bullish"             # Moderate buybacks in last 24h
    NEUTRAL = "neutral"             # Normal buyback activity
    BEARISH = "bearish"             # No recent buybacks
    VERY_BEARISH = "very_bearish"   # Buybacks stopped


@dataclass
class BuybackEvent:
    """Single buyback event"""
    timestamp: datetime
    token: str
    amount_tokens: float
    amount_usd: float
    price_at_buyback: float
    source: str  # e.g., "Assistance Fund", "Treasury"
    tx_hash: Optional[str] = None


@dataclass
class BuybackProfile:
    """Complete buyback profile for a token"""
    token: str
    has_buyback_program: bool = False

    # Program details
    program_name: str = ""
    fee_percentage_to_buyback: float = 0.0  # e.g., 97% for HYPE
    buyback_frequency: str = ""  # "daily", "weekly", "variable"

    # Historical data
    buyback_events: List[BuybackEvent] = field(default_factory=list)
    total_bought_back_usd: float = 0.0
    total_tokens_bought_back: float = 0.0

    # Recent activity
    buyback_24h_usd: float = 0.0
    buyback_7d_usd: float = 0.0
    buyback_30d_usd: float = 0.0

    # Analysis
    avg_daily_buyback_usd: float = 0.0
    sentiment: BuybackSentiment = BuybackSentiment.NEUTRAL


@dataclass
class BuybackAnalysis:
    """Analysis result for buyback-based trading decisions"""
    token: str
    current_price: float

    # Buyback metrics
    buyback_24h_usd: float = 0.0
    buyback_7d_usd: float = 0.0
    is_buyback_active: bool = False

    # Relative to average
    buyback_vs_avg: float = 0.0  # 1.0 = average, 2.0 = 2x average

    # Trading signals
    sentiment: BuybackSentiment = BuybackSentiment.NEUTRAL
    long_signal: float = 0.0  # 0 to 1, higher = stronger long signal
    position_multiplier: float = 1.0  # Suggested position size multiplier

    # Reasoning
    analysis_notes: List[str] = field(default_factory=list)


class BuybackAnalyzer:
    """
    Analyzer for token buyback programs

    Primary use case: HYPE buybacks from Hyperliquid Assistance Fund
    - 97% of trading fees go to buybacks
    - Analyze buyback activity to optimize long entries
    """

    # HypurrScan endpoints
    HYPURRSCAN_API = "https://hypurrscan.io/api"

    # Tokenomist buyback endpoint
    TOKENOMIST_BUYBACK_API = "https://api.tokenomist.ai/v1"

    # Buyback thresholds for HYPE (in USD)
    HYPE_LARGE_BUYBACK = 1_000_000  # > $1M = large
    HYPE_MEDIUM_BUYBACK = 500_000   # > $500K = medium
    HYPE_SMALL_BUYBACK = 100_000    # > $100K = small

    def __init__(
        self,
        tokenomist_api_key: Optional[str] = None,
        cache_ttl: int = 300  # 5 minute cache for real-time data
    ):
        self.tokenomist_api_key = tokenomist_api_key
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}

    async def fetch_hype_buybacks(self) -> BuybackProfile:
        """
        Fetch HYPE buyback data from HypurrScan and Tokenomist

        Returns:
            BuybackProfile with HYPE buyback data
        """
        cache_key = "hype_buybacks"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        profile = BuybackProfile(
            token="HYPE",
            has_buyback_program=True,
            program_name="Hyperliquid Assistance Fund",
            fee_percentage_to_buyback=97.0,
            buyback_frequency="continuous"
        )

        # Try to fetch from HypurrScan
        hypurrscan_data = await self._fetch_hypurrscan_data()
        if hypurrscan_data:
            profile = self._merge_hypurrscan_data(profile, hypurrscan_data)

        # Try to fetch from Tokenomist
        if self.tokenomist_api_key:
            tokenomist_data = await self._fetch_tokenomist_buyback("HYPE")
            if tokenomist_data:
                profile = self._merge_tokenomist_data(profile, tokenomist_data)

        # If no API data, use simulated data
        if not profile.buyback_events:
            profile = await self._get_simulated_hype_data()

        # Calculate sentiment
        profile = self._calculate_sentiment(profile)

        self._cache[cache_key] = profile
        self._cache_timestamps[cache_key] = datetime.utcnow()

        return profile

    async def _fetch_hypurrscan_data(self) -> Optional[Dict]:
        """Fetch buyback data from HypurrScan dashboard"""
        try:
            async with aiohttp.ClientSession() as session:
                # HypurrScan dashboard data endpoint
                async with session.get(
                    f"{self.HYPURRSCAN_API}/buybacks",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()

                # Alternative: scrape dashboard data
                async with session.get(
                    f"{self.HYPURRSCAN_API}/dashboard/stats",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()

        except Exception as e:
            logger.warning(f"HypurrScan fetch failed: {e}")

        return None

    async def _fetch_tokenomist_buyback(self, token: str) -> Optional[Dict]:
        """Fetch buyback data from Tokenomist"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"x-api-key": self.tokenomist_api_key}

                async with session.get(
                    f"{self.TOKENOMIST_BUYBACK_API}/buyback/{token.lower()}",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()

        except Exception as e:
            logger.warning(f"Tokenomist buyback fetch failed: {e}")

        return None

    def _merge_hypurrscan_data(
        self,
        profile: BuybackProfile,
        data: Dict
    ) -> BuybackProfile:
        """Merge HypurrScan data into profile"""
        now = datetime.utcnow()

        # Parse buyback events
        for event_data in data.get('buybacks', []):
            event = BuybackEvent(
                timestamp=datetime.fromisoformat(event_data.get('timestamp', '')),
                token="HYPE",
                amount_tokens=event_data.get('amount', 0),
                amount_usd=event_data.get('valueUsd', 0),
                price_at_buyback=event_data.get('price', 0),
                source="Assistance Fund",
                tx_hash=event_data.get('txHash')
            )
            profile.buyback_events.append(event)

        # Calculate totals
        for event in profile.buyback_events:
            profile.total_bought_back_usd += event.amount_usd
            profile.total_tokens_bought_back += event.amount_tokens

            # Recent activity
            age_hours = (now - event.timestamp).total_seconds() / 3600
            if age_hours <= 24:
                profile.buyback_24h_usd += event.amount_usd
            if age_hours <= 168:  # 7 days
                profile.buyback_7d_usd += event.amount_usd
            if age_hours <= 720:  # 30 days
                profile.buyback_30d_usd += event.amount_usd

        return profile

    def _merge_tokenomist_data(
        self,
        profile: BuybackProfile,
        data: Dict
    ) -> BuybackProfile:
        """Merge Tokenomist buyback data into profile"""
        # Tokenomist provides summary stats
        profile.total_bought_back_usd = max(
            profile.total_bought_back_usd,
            data.get('totalBuybackUsd', 0)
        )

        # Last 10 buyback events
        for event_data in data.get('recentBuybacks', []):
            event = BuybackEvent(
                timestamp=datetime.fromisoformat(event_data.get('date', '')),
                token=profile.token,
                amount_tokens=event_data.get('amount', 0),
                amount_usd=event_data.get('valueUsd', 0),
                price_at_buyback=event_data.get('price', 0),
                source=event_data.get('source', 'Treasury')
            )
            profile.buyback_events.append(event)

        return profile

    async def _get_simulated_hype_data(self) -> BuybackProfile:
        """Get simulated HYPE buyback data for development"""
        profile = BuybackProfile(
            token="HYPE",
            has_buyback_program=True,
            program_name="Hyperliquid Assistance Fund",
            fee_percentage_to_buyback=97.0,
            buyback_frequency="continuous"
        )

        now = datetime.utcnow()

        # Simulate recent buyback activity (realistic based on known data)
        # February 2025 had a $2.4M buyback event
        simulated_events = [
            {"hours_ago": 6, "usd": 850_000},
            {"hours_ago": 18, "usd": 1_200_000},
            {"hours_ago": 36, "usd": 750_000},
            {"hours_ago": 48, "usd": 2_393_423},  # Based on real Feb 2025 event
            {"hours_ago": 72, "usd": 650_000},
            {"hours_ago": 96, "usd": 900_000},
            {"hours_ago": 120, "usd": 1_100_000},
            {"hours_ago": 144, "usd": 800_000},
            {"hours_ago": 168, "usd": 950_000},
        ]

        for event_data in simulated_events:
            event = BuybackEvent(
                timestamp=now - timedelta(hours=event_data["hours_ago"]),
                token="HYPE",
                amount_tokens=event_data["usd"] / 25,  # Assume ~$25 HYPE price
                amount_usd=event_data["usd"],
                price_at_buyback=25.0,
                source="Assistance Fund"
            )
            profile.buyback_events.append(event)
            profile.total_bought_back_usd += event.amount_usd
            profile.total_tokens_bought_back += event.amount_tokens

            # Recent activity
            if event_data["hours_ago"] <= 24:
                profile.buyback_24h_usd += event.amount_usd
            if event_data["hours_ago"] <= 168:
                profile.buyback_7d_usd += event.amount_usd
            profile.buyback_30d_usd += event.amount_usd

        # Calculate average
        profile.avg_daily_buyback_usd = profile.buyback_30d_usd / 30

        return profile

    def _calculate_sentiment(self, profile: BuybackProfile) -> BuybackProfile:
        """Calculate buyback sentiment based on activity"""
        # Calculate average daily buyback
        if profile.buyback_30d_usd > 0:
            profile.avg_daily_buyback_usd = profile.buyback_30d_usd / 30

        daily_24h = profile.buyback_24h_usd
        avg = profile.avg_daily_buyback_usd

        if avg <= 0:
            profile.sentiment = BuybackSentiment.NEUTRAL
            return profile

        ratio = daily_24h / avg if avg > 0 else 0

        if ratio >= 2.0:  # 2x average
            profile.sentiment = BuybackSentiment.VERY_BULLISH
        elif ratio >= 1.2:  # 20% above average
            profile.sentiment = BuybackSentiment.BULLISH
        elif ratio >= 0.5:  # At least 50% of average
            profile.sentiment = BuybackSentiment.NEUTRAL
        elif ratio > 0:  # Some activity
            profile.sentiment = BuybackSentiment.BEARISH
        else:  # No activity
            profile.sentiment = BuybackSentiment.VERY_BEARISH

        return profile

    def analyze_for_trading(
        self,
        profile: BuybackProfile,
        current_price: float
    ) -> BuybackAnalysis:
        """
        Analyze buyback profile for trading decisions

        Returns position sizing recommendation based on buyback activity
        """
        analysis = BuybackAnalysis(
            token=profile.token,
            current_price=current_price,
            buyback_24h_usd=profile.buyback_24h_usd,
            buyback_7d_usd=profile.buyback_7d_usd,
            is_buyback_active=profile.buyback_24h_usd > 0,
            sentiment=profile.sentiment
        )

        notes = []

        # Calculate buyback vs average
        if profile.avg_daily_buyback_usd > 0:
            analysis.buyback_vs_avg = profile.buyback_24h_usd / profile.avg_daily_buyback_usd
        else:
            analysis.buyback_vs_avg = 1.0

        # Calculate long signal and position multiplier
        if profile.sentiment == BuybackSentiment.VERY_BULLISH:
            analysis.long_signal = 0.8
            analysis.position_multiplier = 1.5  # 50% larger position
            notes.append(f"Very bullish: ${profile.buyback_24h_usd/1e6:.2f}M buyback (2x+ avg)")

        elif profile.sentiment == BuybackSentiment.BULLISH:
            analysis.long_signal = 0.6
            analysis.position_multiplier = 1.25  # 25% larger position
            notes.append(f"Bullish: ${profile.buyback_24h_usd/1e6:.2f}M buyback (above avg)")

        elif profile.sentiment == BuybackSentiment.NEUTRAL:
            analysis.long_signal = 0.3
            analysis.position_multiplier = 1.0  # Normal position
            notes.append(f"Neutral: Normal buyback activity")

        elif profile.sentiment == BuybackSentiment.BEARISH:
            analysis.long_signal = 0.1
            analysis.position_multiplier = 0.75  # 25% smaller position
            notes.append(f"Bearish: Low buyback activity")

        else:  # VERY_BEARISH
            analysis.long_signal = 0.0
            analysis.position_multiplier = 0.5  # 50% smaller position
            notes.append(f"Very bearish: No recent buybacks")

        # Additional context
        if profile.buyback_7d_usd > 0:
            notes.append(f"7d total: ${profile.buyback_7d_usd/1e6:.2f}M")

        if profile.total_bought_back_usd > 0:
            notes.append(f"All-time buybacks: ${profile.total_bought_back_usd/1e6:.2f}M")

        analysis.analysis_notes = notes

        return analysis

    async def get_buyback_summary(self, tokens: List[str]) -> Dict[str, BuybackAnalysis]:
        """
        Get buyback analysis for multiple tokens

        Currently supports:
        - HYPE (Hyperliquid)

        Future:
        - BNB (Binance burns)
        - Other tokens with buyback programs
        """
        results = {}

        for token in tokens:
            token_upper = token.upper().replace('/USDT', '').replace('/USDC', '')

            if token_upper == "HYPE":
                profile = await self.fetch_hype_buybacks()
                # Note: Using placeholder price - should be fetched from market data
                analysis = self.analyze_for_trading(profile, 25.0)
                results[token] = analysis

        return results

    def should_increase_position(
        self,
        analysis: BuybackAnalysis,
        threshold: float = 0.5
    ) -> bool:
        """
        Determine if position should be increased based on buybacks

        Args:
            analysis: Buyback analysis result
            threshold: Minimum long signal to recommend increase

        Returns:
            True if buyback activity supports position increase
        """
        return analysis.long_signal >= threshold and analysis.is_buyback_active

    def get_position_size_adjustment(
        self,
        analysis: BuybackAnalysis,
        base_position_size: float
    ) -> float:
        """
        Calculate adjusted position size based on buyback activity

        Args:
            analysis: Buyback analysis result
            base_position_size: Original position size

        Returns:
            Adjusted position size
        """
        return base_position_size * analysis.position_multiplier

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cache entry is still valid"""
        if key not in self._cache:
            return False
        if key not in self._cache_timestamps:
            return False
        age = (datetime.utcnow() - self._cache_timestamps[key]).total_seconds()
        return age < self.cache_ttl


class DeflationaryAssetAnalyzer:
    """
    Analyzer for deflationary assets (buybacks, burns, etc.)

    Identifies tokens that are reducing supply through:
    - Fee buybacks (HYPE, BNB)
    - Token burns
    - Revenue sharing to holders
    """

    def __init__(self):
        self.buyback_analyzer = BuybackAnalyzer()

        # Known deflationary tokens
        self.deflationary_assets = {
            "HYPE": {
                "mechanism": "fee_buyback",
                "rate": 0.97,  # 97% of fees
                "description": "97% of trading fees used for buybacks"
            },
            "BNB": {
                "mechanism": "quarterly_burn",
                "target": "100M tokens",
                "description": "Quarterly burns until 100M supply"
            },
            "ETH": {
                "mechanism": "eip1559_burn",
                "description": "Base fee burned on every transaction"
            }
        }

    async def analyze_deflationary_asset(
        self,
        symbol: str,
        current_price: float
    ) -> Optional[BuybackAnalysis]:
        """
        Analyze deflationary characteristics of an asset

        Returns trading signal based on deflation mechanism activity
        """
        token = symbol.upper().replace('/USDT', '').replace('/USDC', '')

        if token not in self.deflationary_assets:
            return None

        if token == "HYPE":
            profile = await self.buyback_analyzer.fetch_hype_buybacks()
            return self.buyback_analyzer.analyze_for_trading(profile, current_price)

        # For other tokens, return basic analysis
        return BuybackAnalysis(
            token=symbol,
            current_price=current_price,
            is_buyback_active=True,
            sentiment=BuybackSentiment.NEUTRAL,
            long_signal=0.4,  # Baseline long bias for deflationary assets
            position_multiplier=1.1,
            analysis_notes=[f"Deflationary: {self.deflationary_assets[token]['description']}"]
        )

    def is_deflationary(self, symbol: str) -> bool:
        """Check if a token has deflationary mechanisms"""
        token = symbol.upper().replace('/USDT', '').replace('/USDC', '')
        return token in self.deflationary_assets

    def get_deflationary_tokens(self) -> List[str]:
        """Get list of all known deflationary tokens"""
        return list(self.deflationary_assets.keys())
