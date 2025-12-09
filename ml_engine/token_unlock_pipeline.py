"""
Token Unlock Pipeline - Analyzes token vesting schedules, team/VC unlocks

Data Sources:
- Tokenomist.ai API (primary)
- CryptoRank.io (backup)

Identifies:
- Upcoming token unlocks (team, investors, ecosystem)
- Presale investor prices (seed, series A/B/C)
- Inflation rate calculations
- Short opportunities for high-inflation assets
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


class AllocationCategory(Enum):
    """Token allocation categories"""
    TEAM = "team"
    INVESTORS = "investors"
    SEED = "seed"
    PRIVATE_SALE = "private_sale"
    SERIES_A = "series_a"
    SERIES_B = "series_b"
    SERIES_C = "series_c"
    ECOSYSTEM = "ecosystem"
    TREASURY = "treasury"
    COMMUNITY = "community"
    AIRDROP = "airdrop"
    LIQUIDITY = "liquidity"
    ADVISORS = "advisors"
    FOUNDATION = "foundation"


class UnlockRisk(Enum):
    """Risk level for upcoming unlocks"""
    CRITICAL = "critical"    # > 5% supply unlock within 7 days
    HIGH = "high"            # > 2% supply unlock within 7 days
    MEDIUM = "medium"        # > 1% supply unlock within 14 days
    LOW = "low"              # < 1% unlock within 30 days
    MINIMAL = "minimal"      # No significant unlocks


@dataclass
class TokenUnlock:
    """Single token unlock event"""
    symbol: str
    unlock_date: datetime
    amount: float
    amount_usd: float
    percentage_of_supply: float
    category: AllocationCategory
    beneficiary: str
    cliff_end: bool = False
    vesting_end: bool = False


@dataclass
class InvestorRound:
    """Investment round details"""
    round_name: str
    price_per_token: float
    total_raised: float
    tokens_allocated: float
    unlock_start: datetime
    unlock_end: datetime
    vesting_months: int
    cliff_months: int
    current_roi: float = 0.0  # Based on current price


@dataclass
class TokenomicsProfile:
    """Complete tokenomics profile for a token"""
    symbol: str
    total_supply: float
    circulating_supply: float
    max_supply: Optional[float]

    # Unlock data
    upcoming_unlocks: List[TokenUnlock] = field(default_factory=list)
    total_locked: float = 0.0
    total_unlocked: float = 0.0

    # Investor rounds
    investor_rounds: List[InvestorRound] = field(default_factory=list)

    # Calculated metrics
    inflation_30d: float = 0.0  # 30-day inflation rate
    inflation_90d: float = 0.0  # 90-day inflation rate
    unlock_risk: UnlockRisk = UnlockRisk.MINIMAL

    # Short/Long signal
    recommended_direction: str = "neutral"  # "short", "long", "neutral"
    direction_confidence: float = 0.0


@dataclass
class UnlockAnalysis:
    """Analysis result for unlock-based trading"""
    symbol: str
    current_price: float

    # Unlock pressure
    unlock_7d_pct: float = 0.0
    unlock_30d_pct: float = 0.0
    unlock_90d_pct: float = 0.0

    # Investor pressure
    investor_avg_cost: float = 0.0
    investor_roi: float = 0.0
    underwater_investors: bool = False

    # Risk assessment
    risk_level: UnlockRisk = UnlockRisk.MINIMAL
    short_signal: float = 0.0  # -1 to 1, higher = stronger short signal

    # Reasoning
    analysis_notes: List[str] = field(default_factory=list)


class TokenUnlockPipeline:
    """
    Pipeline for analyzing token unlocks and generating short signals

    Strategy:
    - Short tokens with large upcoming team/VC unlocks
    - Especially if investors are in significant profit
    - Higher confidence if unlock is cliff-end (large one-time unlock)
    """

    # Tokenomist API base URL
    TOKENOMIST_API = "https://api.tokenomist.ai"

    # CryptoRank API base URL (backup)
    CRYPTORANK_API = "https://api.cryptorank.io/v1"

    # Risk thresholds
    CRITICAL_UNLOCK_PCT = 5.0   # > 5% = critical
    HIGH_UNLOCK_PCT = 2.0       # > 2% = high
    MEDIUM_UNLOCK_PCT = 1.0     # > 1% = medium

    def __init__(
        self,
        tokenomist_api_key: Optional[str] = None,
        cryptorank_api_key: Optional[str] = None,
        cache_ttl: int = 3600  # 1 hour cache
    ):
        self.tokenomist_api_key = tokenomist_api_key
        self.cryptorank_api_key = cryptorank_api_key
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}

    async def fetch_token_unlocks(self, symbol: str) -> Optional[TokenomicsProfile]:
        """
        Fetch token unlock data from Tokenomist or CryptoRank

        Args:
            symbol: Token symbol (e.g., 'ARB', 'OP', 'APT')

        Returns:
            TokenomicsProfile with all unlock data
        """
        # Check cache
        cache_key = f"unlocks_{symbol}"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        profile = None

        # Try Tokenomist first
        if self.tokenomist_api_key:
            profile = await self._fetch_from_tokenomist(symbol)

        # Fallback to CryptoRank
        if profile is None and self.cryptorank_api_key:
            profile = await self._fetch_from_cryptorank(symbol)

        # Use simulated data if no API keys (for development)
        if profile is None:
            profile = await self._get_simulated_data(symbol)

        if profile:
            self._cache[cache_key] = profile
            self._cache_timestamps[cache_key] = datetime.utcnow()

        return profile

    async def _fetch_from_tokenomist(self, symbol: str) -> Optional[TokenomicsProfile]:
        """Fetch from Tokenomist.ai API"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"x-api-key": self.tokenomist_api_key}

                # Get token list to find tokenID
                async with session.get(
                    f"{self.TOKENOMIST_API}/v2/tokens",
                    headers=headers
                ) as resp:
                    if resp.status != 200:
                        logger.warning(f"Tokenomist token list failed: {resp.status}")
                        return None
                    tokens = await resp.json()

                # Find token ID
                token_id = None
                for token in tokens.get('data', []):
                    if token.get('symbol', '').upper() == symbol.upper():
                        token_id = token.get('tokenId')
                        break

                if not token_id:
                    logger.warning(f"Token {symbol} not found in Tokenomist")
                    return None

                # Get allocations
                async with session.get(
                    f"{self.TOKENOMIST_API}/v2/allocations/{token_id}",
                    headers=headers
                ) as resp:
                    if resp.status != 200:
                        return None
                    allocations = await resp.json()

                # Get weekly emissions
                async with session.get(
                    f"{self.TOKENOMIST_API}/v2/emissions/{token_id}",
                    headers=headers
                ) as resp:
                    if resp.status != 200:
                        return None
                    emissions = await resp.json()

                return self._parse_tokenomist_data(symbol, allocations, emissions)

        except Exception as e:
            logger.error(f"Tokenomist API error: {e}")
            return None

    async def _fetch_from_cryptorank(self, symbol: str) -> Optional[TokenomicsProfile]:
        """Fetch from CryptoRank.io API"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"api-key": self.cryptorank_api_key}

                # Get token vesting data
                async with session.get(
                    f"{self.CRYPTORANK_API}/currencies/{symbol.lower()}/vesting",
                    headers=headers
                ) as resp:
                    if resp.status != 200:
                        logger.warning(f"CryptoRank vesting failed: {resp.status}")
                        return None
                    vesting = await resp.json()

                # Get fundraising data
                async with session.get(
                    f"{self.CRYPTORANK_API}/currencies/{symbol.lower()}/fundraising",
                    headers=headers
                ) as resp:
                    fundraising = None
                    if resp.status == 200:
                        fundraising = await resp.json()

                return self._parse_cryptorank_data(symbol, vesting, fundraising)

        except Exception as e:
            logger.error(f"CryptoRank API error: {e}")
            return None

    def _parse_tokenomist_data(
        self,
        symbol: str,
        allocations: Dict,
        emissions: Dict
    ) -> TokenomicsProfile:
        """Parse Tokenomist API response into TokenomicsProfile"""
        profile = TokenomicsProfile(
            symbol=symbol,
            total_supply=allocations.get('totalSupply', 0),
            circulating_supply=allocations.get('circulatingSupply', 0),
            max_supply=allocations.get('maxSupply')
        )

        # Parse allocations
        for alloc in allocations.get('allocations', []):
            category = self._map_allocation_category(alloc.get('category', ''))

            # Check for investor rounds
            if category in [
                AllocationCategory.SEED,
                AllocationCategory.PRIVATE_SALE,
                AllocationCategory.SERIES_A,
                AllocationCategory.SERIES_B,
                AllocationCategory.SERIES_C
            ]:
                round_info = InvestorRound(
                    round_name=alloc.get('name', category.value),
                    price_per_token=alloc.get('price', 0),
                    total_raised=alloc.get('raised', 0),
                    tokens_allocated=alloc.get('amount', 0),
                    unlock_start=datetime.fromisoformat(alloc.get('vestingStart', datetime.utcnow().isoformat())),
                    unlock_end=datetime.fromisoformat(alloc.get('vestingEnd', datetime.utcnow().isoformat())),
                    vesting_months=alloc.get('vestingMonths', 0),
                    cliff_months=alloc.get('cliffMonths', 0)
                )
                profile.investor_rounds.append(round_info)

        # Parse upcoming emissions/unlocks
        now = datetime.utcnow()
        for emission in emissions.get('weeklyEmissions', []):
            unlock_date = datetime.fromisoformat(emission.get('date', ''))
            if unlock_date > now:
                unlock = TokenUnlock(
                    symbol=symbol,
                    unlock_date=unlock_date,
                    amount=emission.get('amount', 0),
                    amount_usd=emission.get('valueUsd', 0),
                    percentage_of_supply=emission.get('percentageOfSupply', 0),
                    category=self._map_allocation_category(emission.get('allocation', '')),
                    beneficiary=emission.get('beneficiary', 'Unknown'),
                    cliff_end=emission.get('isCliffEnd', False)
                )
                profile.upcoming_unlocks.append(unlock)

        # Calculate inflation rates
        profile = self._calculate_inflation(profile)

        return profile

    def _parse_cryptorank_data(
        self,
        symbol: str,
        vesting: Dict,
        fundraising: Optional[Dict]
    ) -> TokenomicsProfile:
        """Parse CryptoRank API response into TokenomicsProfile"""
        data = vesting.get('data', {})

        profile = TokenomicsProfile(
            symbol=symbol,
            total_supply=data.get('totalSupply', 0),
            circulating_supply=data.get('circulatingSupply', 0),
            max_supply=data.get('maxSupply')
        )

        # Parse vesting schedule
        now = datetime.utcnow()
        for schedule in data.get('vestingSchedule', []):
            unlock_date = datetime.fromisoformat(schedule.get('date', ''))
            if unlock_date > now:
                unlock = TokenUnlock(
                    symbol=symbol,
                    unlock_date=unlock_date,
                    amount=schedule.get('amount', 0),
                    amount_usd=schedule.get('valueUsd', 0),
                    percentage_of_supply=schedule.get('percent', 0),
                    category=self._map_allocation_category(schedule.get('category', '')),
                    beneficiary=schedule.get('beneficiary', 'Unknown')
                )
                profile.upcoming_unlocks.append(unlock)

        # Parse fundraising rounds if available
        if fundraising:
            for round_data in fundraising.get('data', {}).get('rounds', []):
                round_info = InvestorRound(
                    round_name=round_data.get('name', ''),
                    price_per_token=round_data.get('price', 0),
                    total_raised=round_data.get('raised', 0),
                    tokens_allocated=round_data.get('tokens', 0),
                    unlock_start=datetime.utcnow(),
                    unlock_end=datetime.utcnow() + timedelta(days=365),
                    vesting_months=round_data.get('vestingMonths', 24),
                    cliff_months=round_data.get('cliffMonths', 12)
                )
                profile.investor_rounds.append(round_info)

        profile = self._calculate_inflation(profile)
        return profile

    async def _get_simulated_data(self, symbol: str) -> TokenomicsProfile:
        """Get simulated data for development/testing"""
        # Simulated unlock data for common tokens
        simulated_data = {
            "ARB": {
                "total_supply": 10_000_000_000,
                "circulating_supply": 3_000_000_000,
                "unlocks": [
                    {"days": 7, "pct": 2.5, "category": "investors"},
                    {"days": 30, "pct": 1.5, "category": "team"},
                    {"days": 60, "pct": 3.0, "category": "ecosystem"},
                ],
                "investor_price": 0.03,
            },
            "OP": {
                "total_supply": 4_294_967_296,
                "circulating_supply": 1_000_000_000,
                "unlocks": [
                    {"days": 14, "pct": 1.8, "category": "team"},
                    {"days": 45, "pct": 2.2, "category": "investors"},
                ],
                "investor_price": 0.08,
            },
            "APT": {
                "total_supply": 1_000_000_000,
                "circulating_supply": 400_000_000,
                "unlocks": [
                    {"days": 5, "pct": 4.2, "category": "team"},
                    {"days": 21, "pct": 2.8, "category": "investors"},
                ],
                "investor_price": 0.50,
            },
            "SUI": {
                "total_supply": 10_000_000_000,
                "circulating_supply": 2_500_000_000,
                "unlocks": [
                    {"days": 10, "pct": 3.5, "category": "team"},
                    {"days": 35, "pct": 1.9, "category": "seed"},
                ],
                "investor_price": 0.03,
            },
            "WLD": {
                "total_supply": 10_000_000_000,
                "circulating_supply": 500_000_000,
                "unlocks": [
                    {"days": 3, "pct": 5.5, "category": "team"},
                    {"days": 15, "pct": 3.2, "category": "investors"},
                ],
                "investor_price": 0.10,
            },
            "PEPE": {
                "total_supply": 420_690_000_000_000,
                "circulating_supply": 420_690_000_000_000,
                "unlocks": [],  # No unlocks - meme token
                "investor_price": 0,
            },
            "DOGE": {
                "total_supply": 140_000_000_000,
                "circulating_supply": 140_000_000_000,
                "unlocks": [],  # Inflationary but no vesting
                "investor_price": 0,
            },
        }

        data = simulated_data.get(symbol.upper().replace('/USDT', '').replace('/USDC', ''), {})

        if not data:
            # Default profile for unknown tokens
            return TokenomicsProfile(
                symbol=symbol,
                total_supply=1_000_000_000,
                circulating_supply=500_000_000,
                max_supply=None
            )

        profile = TokenomicsProfile(
            symbol=symbol,
            total_supply=data["total_supply"],
            circulating_supply=data["circulating_supply"],
            max_supply=data["total_supply"]
        )

        now = datetime.utcnow()
        for unlock_data in data.get("unlocks", []):
            unlock = TokenUnlock(
                symbol=symbol,
                unlock_date=now + timedelta(days=unlock_data["days"]),
                amount=data["total_supply"] * unlock_data["pct"] / 100,
                amount_usd=0,
                percentage_of_supply=unlock_data["pct"],
                category=self._map_allocation_category(unlock_data["category"]),
                beneficiary=unlock_data["category"].title()
            )
            profile.upcoming_unlocks.append(unlock)

        if data.get("investor_price", 0) > 0:
            profile.investor_rounds.append(InvestorRound(
                round_name="Seed",
                price_per_token=data["investor_price"],
                total_raised=10_000_000,
                tokens_allocated=data["total_supply"] * 0.1,
                unlock_start=now - timedelta(days=365),
                unlock_end=now + timedelta(days=365),
                vesting_months=24,
                cliff_months=12
            ))

        profile = self._calculate_inflation(profile)
        return profile

    def _map_allocation_category(self, category_str: str) -> AllocationCategory:
        """Map string category to AllocationCategory enum"""
        mapping = {
            "team": AllocationCategory.TEAM,
            "investor": AllocationCategory.INVESTORS,
            "investors": AllocationCategory.INVESTORS,
            "seed": AllocationCategory.SEED,
            "private": AllocationCategory.PRIVATE_SALE,
            "private_sale": AllocationCategory.PRIVATE_SALE,
            "series_a": AllocationCategory.SERIES_A,
            "series a": AllocationCategory.SERIES_A,
            "series_b": AllocationCategory.SERIES_B,
            "series b": AllocationCategory.SERIES_B,
            "series_c": AllocationCategory.SERIES_C,
            "series c": AllocationCategory.SERIES_C,
            "ecosystem": AllocationCategory.ECOSYSTEM,
            "treasury": AllocationCategory.TREASURY,
            "community": AllocationCategory.COMMUNITY,
            "airdrop": AllocationCategory.AIRDROP,
            "liquidity": AllocationCategory.LIQUIDITY,
            "advisors": AllocationCategory.ADVISORS,
            "foundation": AllocationCategory.FOUNDATION,
        }
        return mapping.get(category_str.lower(), AllocationCategory.ECOSYSTEM)

    def _calculate_inflation(self, profile: TokenomicsProfile) -> TokenomicsProfile:
        """Calculate inflation rates from unlock schedule"""
        now = datetime.utcnow()

        unlock_7d = 0.0
        unlock_30d = 0.0
        unlock_90d = 0.0

        for unlock in profile.upcoming_unlocks:
            days_until = (unlock.unlock_date - now).days

            if days_until <= 7:
                unlock_7d += unlock.percentage_of_supply
            if days_until <= 30:
                unlock_30d += unlock.percentage_of_supply
            if days_until <= 90:
                unlock_90d += unlock.percentage_of_supply

        profile.inflation_30d = unlock_30d
        profile.inflation_90d = unlock_90d

        # Determine unlock risk
        if unlock_7d >= self.CRITICAL_UNLOCK_PCT:
            profile.unlock_risk = UnlockRisk.CRITICAL
        elif unlock_7d >= self.HIGH_UNLOCK_PCT:
            profile.unlock_risk = UnlockRisk.HIGH
        elif unlock_30d >= self.MEDIUM_UNLOCK_PCT:
            profile.unlock_risk = UnlockRisk.MEDIUM
        elif unlock_30d > 0:
            profile.unlock_risk = UnlockRisk.LOW
        else:
            profile.unlock_risk = UnlockRisk.MINIMAL

        # Calculate total locked/unlocked
        if profile.total_supply > 0:
            profile.total_unlocked = profile.circulating_supply / profile.total_supply
            profile.total_locked = 1 - profile.total_unlocked

        return profile

    def analyze_short_opportunity(
        self,
        profile: TokenomicsProfile,
        current_price: float
    ) -> UnlockAnalysis:
        """
        Analyze a token for short opportunity based on unlocks

        Args:
            profile: Token's unlock profile
            current_price: Current token price

        Returns:
            UnlockAnalysis with short signal strength
        """
        analysis = UnlockAnalysis(
            symbol=profile.symbol,
            current_price=current_price,
            risk_level=profile.unlock_risk
        )

        now = datetime.utcnow()

        # Calculate unlock percentages by timeframe
        for unlock in profile.upcoming_unlocks:
            days_until = (unlock.unlock_date - now).days

            if days_until <= 7:
                analysis.unlock_7d_pct += unlock.percentage_of_supply
            if days_until <= 30:
                analysis.unlock_30d_pct += unlock.percentage_of_supply
            if days_until <= 90:
                analysis.unlock_90d_pct += unlock.percentage_of_supply

        # Calculate investor metrics
        if profile.investor_rounds:
            total_invested = sum(r.tokens_allocated * r.price_per_token for r in profile.investor_rounds)
            total_tokens = sum(r.tokens_allocated for r in profile.investor_rounds)

            if total_tokens > 0:
                analysis.investor_avg_cost = total_invested / total_tokens
                analysis.investor_roi = (current_price - analysis.investor_avg_cost) / analysis.investor_avg_cost if analysis.investor_avg_cost > 0 else 0
                analysis.underwater_investors = analysis.investor_roi < 0

                # Update investor round ROIs
                for round_info in profile.investor_rounds:
                    if round_info.price_per_token > 0:
                        round_info.current_roi = (current_price - round_info.price_per_token) / round_info.price_per_token

        # Calculate short signal
        short_signal = 0.0
        notes = []

        # Factor 1: Near-term unlock pressure (40% weight)
        if analysis.unlock_7d_pct >= 5:
            short_signal += 0.4
            notes.append(f"CRITICAL: {analysis.unlock_7d_pct:.1f}% unlock in 7 days")
        elif analysis.unlock_7d_pct >= 2:
            short_signal += 0.3
            notes.append(f"HIGH: {analysis.unlock_7d_pct:.1f}% unlock in 7 days")
        elif analysis.unlock_7d_pct >= 1:
            short_signal += 0.2
            notes.append(f"MEDIUM: {analysis.unlock_7d_pct:.1f}% unlock in 7 days")

        # Factor 2: Monthly unlock pressure (25% weight)
        if analysis.unlock_30d_pct >= 10:
            short_signal += 0.25
            notes.append(f"Heavy monthly inflation: {analysis.unlock_30d_pct:.1f}%")
        elif analysis.unlock_30d_pct >= 5:
            short_signal += 0.15
            notes.append(f"Elevated monthly inflation: {analysis.unlock_30d_pct:.1f}%")

        # Factor 3: Investor profit taking risk (25% weight)
        if analysis.investor_roi > 5:  # 500% profit
            short_signal += 0.25
            notes.append(f"Extreme investor ROI: {analysis.investor_roi*100:.0f}%")
        elif analysis.investor_roi > 2:  # 200% profit
            short_signal += 0.15
            notes.append(f"High investor ROI: {analysis.investor_roi*100:.0f}%")
        elif analysis.investor_roi > 0.5:  # 50% profit
            short_signal += 0.05
            notes.append(f"Moderate investor ROI: {analysis.investor_roi*100:.0f}%")
        elif analysis.underwater_investors:
            short_signal -= 0.1  # Reduce short signal if investors underwater
            notes.append("Investors underwater - reduced sell pressure")

        # Factor 4: Team/insider unlocks (10% weight)
        team_unlock_soon = any(
            u.category in [AllocationCategory.TEAM, AllocationCategory.ADVISORS]
            and (u.unlock_date - now).days <= 14
            for u in profile.upcoming_unlocks
        )
        if team_unlock_soon:
            short_signal += 0.1
            notes.append("Team/advisor unlock within 14 days")

        # Cap signal between -1 and 1
        analysis.short_signal = max(-1, min(1, short_signal))
        analysis.analysis_notes = notes

        return analysis

    async def screen_tokens_for_shorts(
        self,
        symbols: List[str],
        prices: Dict[str, float],
        min_signal: float = 0.3
    ) -> List[UnlockAnalysis]:
        """
        Screen multiple tokens for short opportunities

        Args:
            symbols: List of token symbols to screen
            prices: Current prices for tokens
            min_signal: Minimum short signal to include

        Returns:
            List of UnlockAnalysis sorted by short signal strength
        """
        results = []

        for symbol in symbols:
            try:
                profile = await self.fetch_token_unlocks(symbol)
                if profile is None:
                    continue

                price = prices.get(symbol, prices.get(symbol.replace('/USDT', '').replace('/USDC', ''), 0))
                if price <= 0:
                    continue

                analysis = self.analyze_short_opportunity(profile, price)

                if analysis.short_signal >= min_signal:
                    results.append(analysis)

            except Exception as e:
                logger.error(f"Error screening {symbol}: {e}")

        # Sort by short signal strength
        results.sort(key=lambda x: x.short_signal, reverse=True)

        return results

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cache entry is still valid"""
        if key not in self._cache:
            return False
        if key not in self._cache_timestamps:
            return False
        age = (datetime.utcnow() - self._cache_timestamps[key]).total_seconds()
        return age < self.cache_ttl

    def get_unlock_calendar(
        self,
        profiles: List[TokenomicsProfile],
        days_ahead: int = 30
    ) -> List[TokenUnlock]:
        """
        Get combined unlock calendar for multiple tokens

        Returns sorted list of upcoming unlocks
        """
        all_unlocks = []
        cutoff = datetime.utcnow() + timedelta(days=days_ahead)

        for profile in profiles:
            for unlock in profile.upcoming_unlocks:
                if unlock.unlock_date <= cutoff:
                    all_unlocks.append(unlock)

        # Sort by date
        all_unlocks.sort(key=lambda x: x.unlock_date)

        return all_unlocks
