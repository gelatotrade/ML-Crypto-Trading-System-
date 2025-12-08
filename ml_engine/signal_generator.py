"""
Signal Generator - Trading Signals with Kelly Criterion Position Sizing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

from .ml_models import PredictionResult

logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """Container for trading signal"""
    symbol: str
    direction: str  # 'long', 'short', 'close'
    strength: float  # Signal strength (-1 to 1)
    kelly_fraction: float  # Kelly criterion position size
    confidence: float
    target_weight: float  # Target portfolio weight
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reason: str = ""


class SignalGenerator:
    """
    Generates trading signals from ML predictions with Kelly criterion sizing.

    Features:
    - Kelly criterion position sizing
    - Signal strength normalization
    - Multi-factor signal combination
    - Risk-adjusted sizing
    """

    def __init__(
        self,
        kelly_fraction: float = 0.5,  # Half-Kelly for safety
        min_signal_strength: float = 0.3,
        max_position_size: float = 0.2,
        config=None
    ):
        """
        Initialize signal generator.

        Args:
            kelly_fraction: Fraction of Kelly criterion to use
            min_signal_strength: Minimum signal strength to generate trade
            max_position_size: Maximum position size
            config: Optional configuration object
        """
        self.kelly_fraction = kelly_fraction
        self.min_signal_strength = min_signal_strength
        self.max_position_size = max_position_size
        self.config = config

    def generate_signals(
        self,
        predictions: Dict[str, PredictionResult],
        volatilities: Dict[str, float],
        current_prices: Dict[str, float],
        risk_metrics: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, TradingSignal]:
        """
        Generate trading signals from ML predictions.

        Args:
            predictions: Dict of symbol -> PredictionResult
            volatilities: Dict of symbol -> volatility
            current_prices: Dict of symbol -> current price
            risk_metrics: Optional risk metrics per symbol

        Returns:
            Dict of symbol -> TradingSignal
        """
        signals = {}

        for symbol, prediction in predictions.items():
            if prediction.confidence < (self.config.ml.min_confidence if self.config else 0.6):
                continue

            vol = volatilities.get(symbol, 0.3)
            price = current_prices.get(symbol, 0)

            signal = self._create_signal(prediction, vol, price, risk_metrics)

            if abs(signal.strength) >= self.min_signal_strength:
                signals[symbol] = signal

        return signals

    def _create_signal(
        self,
        prediction: PredictionResult,
        volatility: float,
        current_price: float,
        risk_metrics: Optional[Dict[str, Dict]] = None
    ) -> TradingSignal:
        """Create a trading signal from prediction"""
        # Signal strength from prediction
        strength = np.tanh(prediction.predicted_return / (volatility + 1e-10))
        strength *= prediction.confidence

        # Kelly criterion calculation
        if volatility > 0:
            # Simplified Kelly: f* = (p*b - q) / b
            # where p = win probability, b = win/loss ratio
            win_prob = 0.5 + prediction.confidence * 0.2  # Confidence boost
            win_loss_ratio = abs(prediction.predicted_return) / volatility

            kelly = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
            kelly = max(0, min(1, kelly))  # Bound between 0 and 1
            kelly *= self.kelly_fraction  # Apply fraction
        else:
            kelly = 0

        # Target weight with volatility adjustment
        vol_scalar = 0.15 / (volatility + 0.01)  # Target 15% vol
        target_weight = kelly * vol_scalar * strength
        target_weight = np.clip(target_weight, -self.max_position_size, self.max_position_size)

        # Direction
        if target_weight > 0.01:
            direction = 'long'
        elif target_weight < -0.01:
            direction = 'short'
        else:
            direction = 'close'

        # Stop loss and take profit
        stop_loss = None
        take_profit = None
        if current_price > 0 and direction != 'close':
            atr_mult = 2.0
            if direction == 'long':
                stop_loss = current_price * (1 - atr_mult * volatility)
                take_profit = current_price * (1 + 3 * atr_mult * volatility)
            else:
                stop_loss = current_price * (1 + atr_mult * volatility)
                take_profit = current_price * (1 - 3 * atr_mult * volatility)

        return TradingSignal(
            symbol=prediction.symbol,
            direction=direction,
            strength=strength,
            kelly_fraction=kelly,
            confidence=prediction.confidence,
            target_weight=target_weight,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=f"ML prediction: {prediction.predicted_return:.4f}"
        )

    def combine_signals(
        self,
        ml_signals: Dict[str, TradingSignal],
        technical_signals: Optional[Dict[str, float]] = None,
        sentiment_signals: Optional[Dict[str, float]] = None,
        weights: Tuple[float, float, float] = (0.6, 0.3, 0.1)
    ) -> Dict[str, TradingSignal]:
        """
        Combine signals from multiple sources.

        Args:
            ml_signals: ML-based signals
            technical_signals: Technical indicator signals
            sentiment_signals: Sentiment-based signals
            weights: Weights for (ML, technical, sentiment)

        Returns:
            Combined signals
        """
        combined = {}

        all_symbols = set(ml_signals.keys())
        if technical_signals:
            all_symbols |= set(technical_signals.keys())
        if sentiment_signals:
            all_symbols |= set(sentiment_signals.keys())

        for symbol in all_symbols:
            ml_strength = ml_signals[symbol].strength if symbol in ml_signals else 0
            tech_strength = technical_signals.get(symbol, 0) if technical_signals else 0
            sent_strength = sentiment_signals.get(symbol, 0) if sentiment_signals else 0

            # Weighted combination
            combined_strength = (
                weights[0] * ml_strength +
                weights[1] * tech_strength +
                weights[2] * sent_strength
            )

            if symbol in ml_signals:
                signal = ml_signals[symbol]
                signal.strength = combined_strength
                signal.target_weight = signal.target_weight * (combined_strength / (signal.strength + 1e-10))
                combined[symbol] = signal

        return combined

    def filter_by_correlation(
        self,
        signals: Dict[str, TradingSignal],
        correlation_screener,
        returns_df: pd.DataFrame
    ) -> Dict[str, TradingSignal]:
        """
        Filter signals based on correlation screening.

        Args:
            signals: Trading signals
            correlation_screener: CorrelationScreener instance
            returns_df: Returns DataFrame

        Returns:
            Filtered signals
        """
        if not signals:
            return signals

        # Convert signals to position dict
        positions = {s.symbol: s.target_weight for s in signals.values()}

        # Screen for correlations
        filtered_signals = correlation_screener.filter_correlated_signals(
            positions,
            returns_df,
            keep_strongest=True
        )

        # Return only signals that passed screening
        return {sym: sig for sym, sig in signals.items() if sym in filtered_signals}

    def rank_signals(
        self,
        signals: Dict[str, TradingSignal],
        by: str = 'strength'
    ) -> List[TradingSignal]:
        """
        Rank signals by specified metric.

        Args:
            signals: Trading signals
            by: Ranking metric ('strength', 'confidence', 'kelly')

        Returns:
            Sorted list of signals
        """
        if by == 'strength':
            key = lambda x: abs(x.strength)
        elif by == 'confidence':
            key = lambda x: x.confidence
        elif by == 'kelly':
            key = lambda x: x.kelly_fraction
        else:
            key = lambda x: abs(x.strength)

        return sorted(signals.values(), key=key, reverse=True)
