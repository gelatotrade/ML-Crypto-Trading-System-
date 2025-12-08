"""
Regime Detector - Market Regime Detection for Dynamic Beta Management
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    NEUTRAL = "neutral"
    CRISIS = "crisis"
    EUPHORIA = "euphoria"


@dataclass
class RegimeAnalysis:
    """Regime analysis results"""
    regime: MarketRegime
    confidence: float
    indicators: Dict[str, float]
    recommended_beta: float
    description: str


class RegimeDetector:
    """
    Detects market regime for dynamic beta adjustment.

    Features:
    - Multi-factor regime detection
    - IV-based sentiment analysis
    - Trend and momentum regime signals
    - Correlation regime detection
    - Volatility regime clustering
    """

    def __init__(
        self,
        risk_on_beta: float = 0.3,
        risk_off_beta: float = -0.3,
        neutral_beta: float = 0.0,
        config=None
    ):
        """
        Initialize regime detector.

        Args:
            risk_on_beta: Target beta for risk-on regime
            risk_off_beta: Target beta for risk-off regime
            neutral_beta: Target beta for neutral regime
            config: Optional configuration
        """
        self.risk_on_beta = risk_on_beta
        self.risk_off_beta = risk_off_beta
        self.neutral_beta = neutral_beta
        self.config = config

        # Regime history
        self._regime_history: List[Tuple[datetime, MarketRegime]] = []

    def detect_regime(
        self,
        market_data: pd.DataFrame,
        options_analysis: Optional[Dict] = None,
        btc_data: Optional[pd.DataFrame] = None
    ) -> RegimeAnalysis:
        """
        Detect current market regime.

        Args:
            market_data: OHLCV data with indicators
            options_analysis: Optional options flow analysis
            btc_data: Optional BTC data for correlation

        Returns:
            RegimeAnalysis with regime and recommended beta
        """
        indicators = {}

        # Trend indicators
        trend_score = self._analyze_trend(market_data)
        indicators['trend'] = trend_score

        # Volatility regime
        vol_regime = self._analyze_volatility_regime(market_data)
        indicators['volatility'] = vol_regime

        # Momentum score
        momentum = self._analyze_momentum(market_data)
        indicators['momentum'] = momentum

        # Options sentiment (if available)
        if options_analysis:
            options_score = self._analyze_options_sentiment(options_analysis)
            indicators['options_sentiment'] = options_score
        else:
            indicators['options_sentiment'] = 0

        # Correlation regime
        if btc_data is not None:
            corr_regime = self._analyze_correlation_regime(market_data, btc_data)
            indicators['correlation'] = corr_regime
        else:
            indicators['correlation'] = 0

        # Combine indicators
        regime, confidence = self._combine_indicators(indicators)

        # Get recommended beta
        recommended_beta = self._get_recommended_beta(regime, confidence)

        # Store in history
        from datetime import datetime
        self._regime_history.append((datetime.utcnow(), regime))

        return RegimeAnalysis(
            regime=regime,
            confidence=confidence,
            indicators=indicators,
            recommended_beta=recommended_beta,
            description=self._get_regime_description(regime, indicators)
        )

    def _analyze_trend(self, data: pd.DataFrame) -> float:
        """
        Analyze trend strength and direction.
        Returns: -1 (strong downtrend) to +1 (strong uptrend)
        """
        if len(data) < 50:
            return 0

        close = data['close']

        # SMA trend
        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1]
        current_price = close.iloc[-1]

        sma_score = 0
        if current_price > sma_20 > sma_50:
            sma_score = 1
        elif current_price < sma_20 < sma_50:
            sma_score = -1
        elif current_price > sma_20:
            sma_score = 0.5
        elif current_price < sma_20:
            sma_score = -0.5

        # ADX trend strength
        adx = data.get('adx', pd.Series([25])).iloc[-1] if 'adx' in data.columns else 25
        trend_strength = min(1, adx / 40)

        return sma_score * trend_strength

    def _analyze_volatility_regime(self, data: pd.DataFrame) -> float:
        """
        Analyze volatility regime.
        Returns: -1 (high vol/fear) to +1 (low vol/complacency)
        """
        if len(data) < 30:
            return 0

        # Calculate realized volatility
        returns = data['close'].pct_change().dropna()
        current_vol = returns.iloc[-20:].std() * np.sqrt(252 * 24)
        historical_vol = returns.std() * np.sqrt(252 * 24)

        # Volatility percentile
        rolling_vol = returns.rolling(20).std() * np.sqrt(252 * 24)
        vol_percentile = (rolling_vol <= current_vol).mean()

        # High vol = risk-off, low vol = risk-on
        if vol_percentile > 0.8:
            return -0.8  # High vol regime
        elif vol_percentile < 0.2:
            return 0.5  # Low vol regime (but not euphoria)
        else:
            return 0

    def _analyze_momentum(self, data: pd.DataFrame) -> float:
        """
        Analyze momentum indicators.
        Returns: -1 to +1
        """
        if len(data) < 20:
            return 0

        scores = []

        # RSI
        if 'rsi_14' in data.columns:
            rsi = data['rsi_14'].iloc[-1]
            if rsi > 70:
                scores.append(0.5)  # Overbought but still bullish
            elif rsi < 30:
                scores.append(-0.5)  # Oversold
            elif rsi > 50:
                scores.append(0.3)
            else:
                scores.append(-0.3)

        # MACD
        if 'macd' in data.columns and 'macd_signal' in data.columns:
            macd = data['macd'].iloc[-1]
            signal = data['macd_signal'].iloc[-1]
            if macd > signal:
                scores.append(0.5)
            else:
                scores.append(-0.5)

        # Price momentum
        returns_5d = data['close'].pct_change(5).iloc[-1]
        returns_20d = data['close'].pct_change(20).iloc[-1]

        if returns_5d > 0.05 and returns_20d > 0.1:
            scores.append(1.0)
        elif returns_5d < -0.05 and returns_20d < -0.1:
            scores.append(-1.0)
        elif returns_5d > 0:
            scores.append(0.3)
        else:
            scores.append(-0.3)

        return np.mean(scores) if scores else 0

    def _analyze_options_sentiment(self, options_analysis: Dict) -> float:
        """
        Analyze options-based sentiment.
        Returns: -1 (bearish) to +1 (bullish)
        """
        score = 0

        # Put/Call ratio
        pcr = options_analysis.get('put_call_ratio', 1)
        if pcr > 1.5:
            score -= 0.5  # High fear
        elif pcr < 0.7:
            score += 0.5  # Bullish sentiment
        elif pcr > 1.2:
            score -= 0.2
        elif pcr < 0.9:
            score += 0.2

        # IV skew
        iv_skew = options_analysis.get('iv_skew', 0)
        if iv_skew > 0.1:
            score -= 0.3  # Put skew = bearish
        elif iv_skew < -0.05:
            score += 0.3  # Call skew = bullish

        # ATM IV level
        atm_iv = options_analysis.get('atm_iv', 0.5)
        if atm_iv > 0.8:
            score -= 0.4  # High fear
        elif atm_iv < 0.3:
            score += 0.2  # Low fear

        return np.clip(score, -1, 1)

    def _analyze_correlation_regime(
        self,
        asset_data: pd.DataFrame,
        btc_data: pd.DataFrame
    ) -> float:
        """
        Analyze correlation regime with BTC.
        Returns: correlation-based regime signal
        """
        if len(asset_data) < 30 or len(btc_data) < 30:
            return 0

        # Align data
        asset_returns = asset_data['close'].pct_change().dropna()
        btc_returns = btc_data['close'].pct_change().dropna()

        # Calculate rolling correlation
        common_idx = asset_returns.index.intersection(btc_returns.index)
        if len(common_idx) < 20:
            return 0

        corr = asset_returns.loc[common_idx].corr(btc_returns.loc[common_idx])

        # High correlation = risk assets moving together
        # During risk-off, correlations typically increase
        if corr > 0.8:
            return -0.3  # High correlation often means panic
        elif corr < 0.3:
            return 0.2  # Low correlation = diversification working

        return 0

    def _combine_indicators(
        self,
        indicators: Dict[str, float]
    ) -> Tuple[MarketRegime, float]:
        """Combine indicators to determine regime"""
        # Weights
        weights = {
            'trend': 0.25,
            'volatility': 0.25,
            'momentum': 0.2,
            'options_sentiment': 0.2,
            'correlation': 0.1
        }

        # Weighted score
        total_score = sum(
            indicators.get(k, 0) * w
            for k, w in weights.items()
        )

        # Determine regime
        if total_score > 0.4:
            if total_score > 0.7:
                regime = MarketRegime.EUPHORIA
            else:
                regime = MarketRegime.RISK_ON
        elif total_score < -0.4:
            if total_score < -0.7:
                regime = MarketRegime.CRISIS
            else:
                regime = MarketRegime.RISK_OFF
        else:
            regime = MarketRegime.NEUTRAL

        # Confidence based on indicator agreement
        indicator_values = list(indicators.values())
        if indicator_values:
            agreement = 1 - np.std(indicator_values)
            confidence = max(0.3, min(0.95, agreement))
        else:
            confidence = 0.5

        return regime, confidence

    def _get_recommended_beta(
        self,
        regime: MarketRegime,
        confidence: float
    ) -> float:
        """Get recommended portfolio beta for regime"""
        base_betas = {
            MarketRegime.RISK_ON: self.risk_on_beta,
            MarketRegime.RISK_OFF: self.risk_off_beta,
            MarketRegime.NEUTRAL: self.neutral_beta,
            MarketRegime.CRISIS: self.risk_off_beta * 1.5,
            MarketRegime.EUPHORIA: self.risk_on_beta * 0.5  # Be cautious in euphoria
        }

        base_beta = base_betas.get(regime, 0)

        # Scale by confidence
        return base_beta * confidence

    def _get_regime_description(
        self,
        regime: MarketRegime,
        indicators: Dict[str, float]
    ) -> str:
        """Generate human-readable regime description"""
        descriptions = {
            MarketRegime.RISK_ON: "Bullish environment - trending up with positive momentum",
            MarketRegime.RISK_OFF: "Bearish environment - risk aversion increasing",
            MarketRegime.NEUTRAL: "Mixed signals - market directionless",
            MarketRegime.CRISIS: "High stress - significant risk-off move",
            MarketRegime.EUPHORIA: "Extreme bullishness - caution warranted"
        }

        base_desc = descriptions.get(regime, "Unknown regime")

        # Add indicator context
        key_indicators = []
        if abs(indicators.get('trend', 0)) > 0.5:
            key_indicators.append(f"Trend: {'Strong Up' if indicators['trend'] > 0 else 'Strong Down'}")
        if abs(indicators.get('volatility', 0)) > 0.5:
            key_indicators.append(f"Vol: {'High' if indicators['volatility'] < 0 else 'Low'}")

        if key_indicators:
            return f"{base_desc} ({', '.join(key_indicators)})"

        return base_desc

    def get_regime_history(self, periods: int = 100) -> List[Tuple]:
        """Get recent regime history"""
        return self._regime_history[-periods:]

    def get_regime_persistence(self) -> Dict[MarketRegime, int]:
        """Calculate how long each regime has persisted"""
        if not self._regime_history:
            return {}

        persistence = {}
        current_regime = self._regime_history[-1][1]
        count = 0

        for _, regime in reversed(self._regime_history):
            if regime == current_regime:
                count += 1
            else:
                break

        persistence[current_regime] = count
        return persistence
