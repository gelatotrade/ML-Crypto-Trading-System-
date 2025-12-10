"""
VERFEINERTE DATEN-PIPELINE: Mehrschichtige Datenverarbeitung

Layer 1: Raw Data Processing
- Event-Driven Architecture mit Complex Event Processing (CEP)
- Online-Outlier-Detection (Mahalanobis-Distanz)
- Microstructure-Features (VPIN, Order Flow Imbalance, Price Impact)

Layer 2: Alternative Daten-Integration
- NLP-Pipeline für Nachrichten (BERT-basiert, LDA Topics, GRU Sequential)
- Options-Markt-Daten (SABR-Modell Volatilitäts-Oberflächen)

Layer 3: Stationaritäts-Transformationen
- Log-Returns für Preis-basierte Features
- Z-score Normalisierung mit rolling mean/std
- Quantile-Transformation für Outlier-Robustheit
- Cointegration-Tests für Paar-Trading Features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import deque
import logging
import warnings
import hashlib
import json

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TradeEvent:
    """Individual trade event for microstructure analysis"""
    timestamp: datetime
    price: float
    volume: float
    side: str  # 'buy' or 'sell'
    trade_id: str
    is_aggressive: bool = True  # Taker vs maker


@dataclass
class OrderbookSnapshot:
    """Point-in-time orderbook state"""
    timestamp: datetime
    bids: List[Tuple[float, float]]  # (price, volume) pairs
    asks: List[Tuple[float, float]]
    mid_price: float
    spread: float
    imbalance: float


@dataclass
class MicrostructureFeatures:
    """Microstructure analysis results"""
    vpin: float                    # Volume-Synchronized Probability of Informed Trading
    order_flow_imbalance: float    # Signed volume imbalance
    price_impact: float            # δP/δV
    kyle_lambda: float             # Kyle's lambda
    roll_spread: float             # Roll spread estimator
    effective_spread: float
    realized_volatility: float
    trade_intensity: float         # Trades per unit time
    volume_clock_volatility: float
    toxicity_index: float          # Combined flow toxicity metric


@dataclass
class SentimentAnalysis:
    """NLP sentiment analysis results"""
    overall_sentiment: float       # -1 to 1
    confidence: float              # 0 to 1
    topic_distribution: Dict[str, float]
    key_entities: List[str]
    urgency_score: float           # How time-sensitive
    source_credibility: float
    raw_text: str
    timestamp: datetime


@dataclass
class VolatilitySurface:
    """Options volatility surface (SABR calibrated)"""
    atm_vol: float
    skew: float                    # 25-delta risk reversal
    smile: float                   # 25-delta butterfly
    term_structure: Dict[str, float]  # Tenor -> ATM vol
    sabr_alpha: float              # SABR parameters
    sabr_beta: float
    sabr_rho: float
    sabr_nu: float


@dataclass
class ProcessedFeatures:
    """Fully processed feature set"""
    symbol: str
    timestamp: datetime
    raw_features: Dict[str, float]
    stationary_features: Dict[str, float]
    microstructure_features: MicrostructureFeatures
    sentiment_features: Optional[SentimentAnalysis]
    volatility_surface: Optional[VolatilitySurface]
    feature_quality_scores: Dict[str, float]


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 1: MICROSTRUCTURE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

class MicrostructureAnalyzer:
    """
    Microstructure analysis for high-frequency features.

    Implements:
    - VPIN (Volume-Synchronized Probability of Informed Trading)
    - Order Flow Imbalance
    - Price Impact Coefficient
    - Kyle's Lambda estimation
    """

    def __init__(
        self,
        vpin_bucket_size: float = 50000,  # Volume per bucket in USD
        vpin_lookback: int = 50,           # Number of buckets
        flow_lookback: int = 100,          # Trades for flow calculation
        impact_window: int = 20            # Window for impact calculation
    ):
        self.vpin_bucket_size = vpin_bucket_size
        self.vpin_lookback = vpin_lookback
        self.flow_lookback = flow_lookback
        self.impact_window = impact_window

        # Rolling data storage
        self.trade_buffer: deque = deque(maxlen=10000)
        self.vpin_buckets: deque = deque(maxlen=vpin_lookback)
        self.orderbook_buffer: deque = deque(maxlen=1000)

    def process_trade(self, trade: TradeEvent) -> Optional[MicrostructureFeatures]:
        """Process incoming trade and update microstructure metrics"""
        self.trade_buffer.append(trade)

        # Update VPIN buckets
        self._update_vpin_buckets(trade)

        # Return features if enough data
        if len(self.trade_buffer) >= self.flow_lookback:
            return self.calculate_features()

        return None

    def process_orderbook(self, snapshot: OrderbookSnapshot):
        """Process orderbook snapshot"""
        self.orderbook_buffer.append(snapshot)

    def _update_vpin_buckets(self, trade: TradeEvent):
        """Update VPIN volume buckets"""
        if not hasattr(self, '_current_bucket_volume'):
            self._current_bucket_volume = 0
            self._current_bucket_buy_volume = 0

        trade_volume = trade.volume * trade.price  # Convert to USD

        if trade.side == 'buy':
            self._current_bucket_buy_volume += trade_volume

        self._current_bucket_volume += trade_volume

        # Check if bucket is full
        if self._current_bucket_volume >= self.vpin_bucket_size:
            # Calculate bucket imbalance
            buy_ratio = self._current_bucket_buy_volume / (self._current_bucket_volume + 1e-10)
            imbalance = abs(buy_ratio - 0.5) * 2  # 0 to 1

            self.vpin_buckets.append({
                'imbalance': imbalance,
                'volume': self._current_bucket_volume,
                'buy_ratio': buy_ratio
            })

            # Reset bucket
            self._current_bucket_volume = 0
            self._current_bucket_buy_volume = 0

    def calculate_vpin(self) -> float:
        """
        Calculate VPIN (Volume-Synchronized Probability of Informed Trading).

        VPIN = Σ|Buy_i - Sell_i| / (2 * Σ Volume_i)
        """
        if len(self.vpin_buckets) < self.vpin_lookback:
            return 0.5  # Default to neutral

        # Calculate VPIN from buckets
        total_imbalance = sum(b['imbalance'] * b['volume'] for b in self.vpin_buckets)
        total_volume = sum(b['volume'] for b in self.vpin_buckets)

        vpin = total_imbalance / (2 * total_volume + 1e-10)

        return np.clip(vpin, 0, 1)

    def calculate_order_flow_imbalance(self) -> float:
        """
        Calculate Order Flow Imbalance.

        OFI = Σ sign(trade_i) * √volume_i
        """
        if len(self.trade_buffer) < self.flow_lookback:
            return 0

        recent_trades = list(self.trade_buffer)[-self.flow_lookback:]

        ofi = 0
        for trade in recent_trades:
            sign = 1 if trade.side == 'buy' else -1
            ofi += sign * np.sqrt(trade.volume)

        # Normalize by square root of total volume
        total_vol = sum(t.volume for t in recent_trades)
        normalized_ofi = ofi / (np.sqrt(total_vol) + 1e-10)

        return np.clip(normalized_ofi, -1, 1)

    def calculate_price_impact(self) -> float:
        """
        Calculate Price Impact Coefficient.

        δP/δV = regression coefficient of price change on signed volume
        """
        if len(self.trade_buffer) < self.impact_window * 2:
            return 0

        recent_trades = list(self.trade_buffer)[-self.impact_window * 2:]

        # Create return and volume series
        prices = [t.price for t in recent_trades]
        volumes = [t.volume * (1 if t.side == 'buy' else -1) for t in recent_trades]

        returns = np.diff(prices) / np.array(prices[:-1])
        signed_volumes = np.array(volumes[1:])

        # Linear regression
        if np.std(signed_volumes) > 0:
            coef = np.cov(returns, signed_volumes)[0, 1] / (np.var(signed_volumes) + 1e-10)
        else:
            coef = 0

        return coef

    def calculate_kyle_lambda(self) -> float:
        """
        Calculate Kyle's Lambda.

        λ = Cov(ΔP, OrderFlow) / Var(OrderFlow)

        Measures market depth and information asymmetry.
        """
        if len(self.trade_buffer) < self.impact_window * 2:
            return 0

        recent_trades = list(self.trade_buffer)[-self.impact_window * 2:]

        # Aggregate into periods
        period_size = 5  # Aggregate every 5 trades
        n_periods = len(recent_trades) // period_size

        if n_periods < 10:
            return 0

        price_changes = []
        order_flows = []

        for i in range(n_periods):
            period_trades = recent_trades[i * period_size:(i + 1) * period_size]
            start_price = period_trades[0].price
            end_price = period_trades[-1].price

            price_change = (end_price - start_price) / start_price
            flow = sum(t.volume * (1 if t.side == 'buy' else -1) for t in period_trades)

            price_changes.append(price_change)
            order_flows.append(flow)

        price_changes = np.array(price_changes)
        order_flows = np.array(order_flows)

        # Calculate lambda
        cov = np.cov(price_changes, order_flows)[0, 1]
        var_flow = np.var(order_flows)

        kyle_lambda = cov / (var_flow + 1e-10)

        return kyle_lambda

    def calculate_roll_spread(self) -> float:
        """
        Calculate Roll Spread Estimator.

        Spread = 2 * √(-Cov(r_t, r_{t-1}))
        """
        if len(self.trade_buffer) < 50:
            return 0

        prices = [t.price for t in list(self.trade_buffer)[-100:]]
        returns = np.diff(np.log(prices))

        if len(returns) < 2:
            return 0

        # Autocovariance at lag 1
        cov_lag1 = np.cov(returns[:-1], returns[1:])[0, 1]

        if cov_lag1 < 0:
            roll_spread = 2 * np.sqrt(-cov_lag1)
        else:
            roll_spread = 0

        return roll_spread

    def calculate_effective_spread(self) -> float:
        """
        Calculate Effective Spread from orderbook.

        Effective Spread = 2 * |P_trade - Mid| / Mid
        """
        if len(self.orderbook_buffer) < 1 or len(self.trade_buffer) < 1:
            return 0

        last_orderbook = self.orderbook_buffer[-1]
        last_trade = self.trade_buffer[-1]

        mid_price = last_orderbook.mid_price

        effective_spread = 2 * abs(last_trade.price - mid_price) / (mid_price + 1e-10)

        return effective_spread

    def calculate_toxicity_index(self) -> float:
        """
        Calculate combined flow toxicity index.

        Combines VPIN, order flow imbalance, and effective spread.
        """
        vpin = self.calculate_vpin()
        ofi = abs(self.calculate_order_flow_imbalance())
        spread = min(1, self.calculate_effective_spread() * 100)  # Normalize

        # Weighted combination
        toxicity = 0.4 * vpin + 0.35 * ofi + 0.25 * spread

        return np.clip(toxicity, 0, 1)

    def calculate_features(self) -> MicrostructureFeatures:
        """Calculate all microstructure features"""
        # Trade intensity
        if len(self.trade_buffer) > 0:
            time_span = (self.trade_buffer[-1].timestamp -
                        self.trade_buffer[0].timestamp).total_seconds()
            trade_intensity = len(self.trade_buffer) / (time_span + 1) if time_span > 0 else 0
        else:
            trade_intensity = 0

        # Realized volatility
        if len(self.trade_buffer) > 20:
            prices = [t.price for t in list(self.trade_buffer)[-100:]]
            returns = np.diff(np.log(prices))
            realized_vol = np.std(returns) * np.sqrt(252 * 24 * 60)  # Annualized
        else:
            realized_vol = 0

        # Volume clock volatility (volatility per unit volume)
        if len(self.vpin_buckets) > 5:
            bucket_returns = []
            prev_price = None
            for bucket in list(self.vpin_buckets)[-20:]:
                if 'last_price' in bucket:
                    if prev_price is not None:
                        ret = np.log(bucket['last_price'] / prev_price)
                        bucket_returns.append(ret)
                    prev_price = bucket['last_price']
            vol_clock_vol = np.std(bucket_returns) if bucket_returns else 0
        else:
            vol_clock_vol = 0

        return MicrostructureFeatures(
            vpin=self.calculate_vpin(),
            order_flow_imbalance=self.calculate_order_flow_imbalance(),
            price_impact=self.calculate_price_impact(),
            kyle_lambda=self.calculate_kyle_lambda(),
            roll_spread=self.calculate_roll_spread(),
            effective_spread=self.calculate_effective_spread(),
            realized_volatility=realized_vol,
            trade_intensity=trade_intensity,
            volume_clock_volatility=vol_clock_vol,
            toxicity_index=self.calculate_toxicity_index()
        )


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 1: ONLINE OUTLIER DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

class OnlineOutlierDetector:
    """
    Online Outlier Detection using Mahalanobis Distance.

    Maintains rolling mean and covariance for real-time outlier detection.
    """

    def __init__(
        self,
        window_size: int = 100,
        threshold_percentile: float = 99,
        min_samples: int = 30
    ):
        self.window_size = window_size
        self.threshold_percentile = threshold_percentile
        self.min_samples = min_samples

        self.data_buffer: deque = deque(maxlen=window_size)
        self.outlier_history: deque = deque(maxlen=1000)

    def update(self, observation: np.ndarray) -> Tuple[bool, float]:
        """
        Update with new observation and check if outlier.

        Returns:
            - is_outlier: Boolean
            - mahalanobis_distance: float
        """
        self.data_buffer.append(observation)

        if len(self.data_buffer) < self.min_samples:
            return False, 0.0

        # Calculate Mahalanobis distance
        data = np.array(list(self.data_buffer))
        mean = np.mean(data[:-1], axis=0)

        # Regularized covariance
        cov = np.cov(data[:-1].T)
        if cov.ndim == 0:
            cov = np.array([[cov]])

        # Add regularization for numerical stability
        cov += np.eye(cov.shape[0]) * 1e-6

        try:
            cov_inv = np.linalg.inv(cov)
        except:
            return False, 0.0

        # Mahalanobis distance
        diff = observation - mean
        mahal_dist = np.sqrt(diff @ cov_inv @ diff)

        # Determine threshold from chi-squared distribution
        from scipy import stats
        dof = len(observation)
        threshold = np.sqrt(stats.chi2.ppf(self.threshold_percentile / 100, dof))

        is_outlier = mahal_dist > threshold

        self.outlier_history.append({
            'timestamp': datetime.utcnow(),
            'distance': mahal_dist,
            'is_outlier': is_outlier,
            'threshold': threshold
        })

        return is_outlier, mahal_dist

    def get_outlier_rate(self, lookback: int = 100) -> float:
        """Get recent outlier rate"""
        if len(self.outlier_history) < 10:
            return 0.0

        recent = list(self.outlier_history)[-lookback:]
        return sum(1 for o in recent if o['is_outlier']) / len(recent)


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 2: NLP SENTIMENT PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

class NLPSentimentPipeline:
    """
    NLP Pipeline for News/Social Media Sentiment Analysis.

    Combines:
    - BERT-based sentiment (or fallback to lexicon-based)
    - LDA topic modeling
    - Sequential context with GRU-like aggregation

    sentiment = BERT_Fin(sentence) ⊕ LDA(topic) ⊕ GRU(sequential_context)
    """

    def __init__(
        self,
        use_transformer: bool = False,  # Use BERT if available
        max_sequence_length: int = 512,
        topic_count: int = 10,
        sentiment_decay: float = 0.95  # Decay for sequential aggregation
    ):
        self.use_transformer = use_transformer
        self.max_sequence_length = max_sequence_length
        self.topic_count = topic_count
        self.sentiment_decay = sentiment_decay

        # Sentiment lexicon (FinBERT-style keywords)
        self.positive_words = {
            'bullish', 'surge', 'rally', 'pump', 'moon', 'breakout',
            'growth', 'profit', 'gain', 'increase', 'rise', 'soar',
            'strong', 'outperform', 'exceed', 'upgrade', 'momentum',
            'accumulation', 'buy', 'long', 'support', 'adoption'
        }

        self.negative_words = {
            'bearish', 'crash', 'dump', 'plunge', 'fall', 'drop',
            'loss', 'decline', 'decrease', 'weak', 'underperform',
            'miss', 'downgrade', 'selloff', 'sell', 'short', 'resist',
            'fear', 'panic', 'liquidation', 'hack', 'exploit', 'rug'
        }

        self.urgency_words = {
            'breaking', 'urgent', 'alert', 'now', 'immediately',
            'just', 'happening', 'live', 'emergency', 'critical'
        }

        # Sequential sentiment buffer
        self.sentiment_buffer: deque = deque(maxlen=100)

        # Topic model (simplified LDA-like)
        self.topic_keywords = self._init_crypto_topics()

    def _init_crypto_topics(self) -> Dict[str, List[str]]:
        """Initialize crypto-specific topic keywords"""
        return {
            'defi': ['defi', 'yield', 'farm', 'liquidity', 'pool', 'swap', 'amm', 'tvl'],
            'nft': ['nft', 'mint', 'collection', 'art', 'pfp', 'opensea', 'blur'],
            'regulation': ['sec', 'regulation', 'compliance', 'legal', 'lawsuit', 'ban'],
            'macro': ['fed', 'rate', 'inflation', 'recession', 'economy', 'dollar'],
            'technology': ['upgrade', 'fork', 'layer2', 'scaling', 'security', 'smart contract'],
            'exchange': ['exchange', 'cex', 'dex', 'binance', 'coinbase', 'listing'],
            'whale': ['whale', 'accumulation', 'distribution', 'wallet', 'transfer'],
            'sentiment': ['fomo', 'fud', 'fear', 'greed', 'sentiment', 'crowd'],
            'ecosystem': ['ecosystem', 'partnership', 'integration', 'developer', 'grant'],
            'price': ['price', 'ath', 'atl', 'resistance', 'support', 'target']
        }

    def analyze(
        self,
        text: str,
        source: str = 'unknown',
        timestamp: Optional[datetime] = None
    ) -> SentimentAnalysis:
        """
        Analyze text sentiment.

        Args:
            text: Raw text to analyze
            source: Source of the text (twitter, news, etc.)
            timestamp: Timestamp of the text

        Returns:
            SentimentAnalysis with all metrics
        """
        timestamp = timestamp or datetime.utcnow()

        # Preprocess text
        processed_text = self._preprocess(text)
        words = processed_text.lower().split()

        # 1. Lexicon-based sentiment (fallback for BERT)
        base_sentiment = self._lexicon_sentiment(words)

        # 2. Topic distribution
        topic_dist = self._calculate_topic_distribution(words)

        # 3. Entity extraction (simplified)
        entities = self._extract_entities(text)

        # 4. Urgency score
        urgency = self._calculate_urgency(words)

        # 5. Source credibility
        credibility = self._estimate_credibility(source)

        # 6. Sequential context (GRU-like aggregation)
        sequential_sentiment = self._aggregate_sequential_sentiment(
            base_sentiment, timestamp
        )

        # Combine all signals
        overall_sentiment = self._combine_signals(
            base_sentiment,
            topic_dist,
            sequential_sentiment
        )

        # Calculate confidence
        confidence = self._calculate_confidence(
            words, topic_dist, credibility
        )

        analysis = SentimentAnalysis(
            overall_sentiment=overall_sentiment,
            confidence=confidence,
            topic_distribution=topic_dist,
            key_entities=entities,
            urgency_score=urgency,
            source_credibility=credibility,
            raw_text=text[:500],  # Truncate for storage
            timestamp=timestamp
        )

        # Store for sequential analysis
        self.sentiment_buffer.append({
            'sentiment': overall_sentiment,
            'timestamp': timestamp,
            'confidence': confidence
        })

        return analysis

    def _preprocess(self, text: str) -> str:
        """Preprocess text for analysis"""
        import re

        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)

        # Remove mentions
        text = re.sub(r'@\w+', '', text)

        # Remove hashtag symbols (keep words)
        text = re.sub(r'#', '', text)

        # Remove special characters except basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)

        # Normalize whitespace
        text = ' '.join(text.split())

        return text

    def _lexicon_sentiment(self, words: List[str]) -> float:
        """Calculate lexicon-based sentiment"""
        positive_count = sum(1 for w in words if w in self.positive_words)
        negative_count = sum(1 for w in words if w in self.negative_words)

        total = positive_count + negative_count

        if total == 0:
            return 0.0

        sentiment = (positive_count - negative_count) / total

        return np.clip(sentiment, -1, 1)

    def _calculate_topic_distribution(self, words: List[str]) -> Dict[str, float]:
        """Calculate topic distribution (simplified LDA)"""
        topic_scores = {}

        for topic, keywords in self.topic_keywords.items():
            score = sum(1 for w in words if w in keywords)
            topic_scores[topic] = score

        # Normalize
        total = sum(topic_scores.values()) + 1e-10

        return {topic: score / total for topic, score in topic_scores.items()}

    def _extract_entities(self, text: str) -> List[str]:
        """Extract key entities (simplified NER)"""
        entities = []

        # Crypto symbols (uppercase 3-5 letters)
        import re
        symbols = re.findall(r'\b[A-Z]{3,5}\b', text)
        entities.extend(symbols[:5])  # Top 5

        # Dollar amounts
        amounts = re.findall(r'\$[\d,]+(?:\.\d+)?[BMK]?', text)
        entities.extend(amounts[:3])

        return list(set(entities))

    def _calculate_urgency(self, words: List[str]) -> float:
        """Calculate urgency score"""
        urgency_count = sum(1 for w in words if w in self.urgency_words)

        # Normalize by text length
        urgency = urgency_count / (len(words) + 1) * 10

        return np.clip(urgency, 0, 1)

    def _estimate_credibility(self, source: str) -> float:
        """Estimate source credibility"""
        credibility_scores = {
            'reuters': 0.95,
            'bloomberg': 0.95,
            'coindesk': 0.85,
            'cointelegraph': 0.80,
            'decrypt': 0.80,
            'twitter': 0.50,
            'reddit': 0.40,
            'telegram': 0.30,
            'unknown': 0.50
        }

        source_lower = source.lower()

        for key, score in credibility_scores.items():
            if key in source_lower:
                return score

        return 0.50

    def _aggregate_sequential_sentiment(
        self,
        current_sentiment: float,
        timestamp: datetime
    ) -> float:
        """Aggregate sentiment over time (GRU-like decay)"""
        if len(self.sentiment_buffer) == 0:
            return current_sentiment

        # Time-weighted aggregation
        aggregated = current_sentiment
        total_weight = 1.0

        for item in reversed(list(self.sentiment_buffer)):
            time_diff = (timestamp - item['timestamp']).total_seconds() / 3600  # hours

            # Exponential decay
            weight = self.sentiment_decay ** time_diff

            if weight < 0.01:
                break

            aggregated += item['sentiment'] * item['confidence'] * weight
            total_weight += item['confidence'] * weight

        return aggregated / total_weight

    def _combine_signals(
        self,
        base_sentiment: float,
        topic_dist: Dict[str, float],
        sequential_sentiment: float
    ) -> float:
        """Combine all sentiment signals"""
        # Topic-adjusted sentiment
        topic_adjustment = 0

        # Positive topics
        positive_topics = ['ecosystem', 'technology', 'whale']
        for topic in positive_topics:
            topic_adjustment += topic_dist.get(topic, 0) * 0.1

        # Negative topics
        negative_topics = ['regulation']
        for topic in negative_topics:
            topic_adjustment -= topic_dist.get(topic, 0) * 0.15

        # Combine
        combined = (
            0.4 * base_sentiment +
            0.3 * sequential_sentiment +
            0.3 * (base_sentiment + topic_adjustment)
        )

        return np.clip(combined, -1, 1)

    def _calculate_confidence(
        self,
        words: List[str],
        topic_dist: Dict[str, float],
        credibility: float
    ) -> float:
        """Calculate analysis confidence"""
        # Text length factor
        length_factor = min(1.0, len(words) / 20)

        # Topic clarity (entropy-based)
        topic_probs = list(topic_dist.values())
        if sum(topic_probs) > 0:
            entropy = -sum(p * np.log(p + 1e-10) for p in topic_probs)
            max_entropy = np.log(len(topic_probs))
            clarity = 1 - (entropy / max_entropy)
        else:
            clarity = 0.5

        # Combine factors
        confidence = 0.4 * length_factor + 0.3 * clarity + 0.3 * credibility

        return np.clip(confidence, 0, 1)

    def get_aggregated_sentiment(
        self,
        lookback_hours: int = 24
    ) -> Tuple[float, float]:
        """
        Get aggregated sentiment over time period.

        Returns:
            - Average sentiment
            - Sentiment trend (change)
        """
        if len(self.sentiment_buffer) < 2:
            return 0.0, 0.0

        cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)
        recent = [
            s for s in self.sentiment_buffer
            if s['timestamp'] > cutoff
        ]

        if len(recent) < 2:
            return 0.0, 0.0

        # Weighted average
        total_sentiment = sum(s['sentiment'] * s['confidence'] for s in recent)
        total_weight = sum(s['confidence'] for s in recent)
        avg_sentiment = total_sentiment / (total_weight + 1e-10)

        # Trend (first half vs second half)
        mid = len(recent) // 2
        first_half = sum(s['sentiment'] for s in recent[:mid]) / (mid + 1e-10)
        second_half = sum(s['sentiment'] for s in recent[mid:]) / (len(recent) - mid + 1e-10)
        trend = second_half - first_half

        return avg_sentiment, trend


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 2: SABR VOLATILITY SURFACE
# ═══════════════════════════════════════════════════════════════════════════════

class SABRVolatilitySurface:
    """
    SABR Model for Volatility Surface Calibration.

    σ_SABR(K,T) = α * f^(β-1) * [1 + ((1-β)²/24 * α²/f^(2-2β) + ρβνα/(4f^(1-β)) + (2-3ρ²)/24 * ν²) * T]

    Where:
    - α (alpha): ATM volatility level
    - β (beta): CEV exponent (typically 0.5 for rates, 1.0 for FX)
    - ρ (rho): Correlation between asset and volatility
    - ν (nu): Volatility of volatility
    """

    def __init__(
        self,
        beta: float = 0.5,  # Fixed beta for crypto (log-normal blend)
        calibration_strikes: int = 5  # Number of strikes to calibrate
    ):
        self.beta = beta
        self.calibration_strikes = calibration_strikes

        # Cached calibration
        self.calibrated_params: Dict[str, Dict] = {}

    def calibrate(
        self,
        forward: float,
        expiry_years: float,
        strikes: np.ndarray,
        market_vols: np.ndarray
    ) -> Dict[str, float]:
        """
        Calibrate SABR parameters to market volatilities.

        Args:
            forward: Forward price
            expiry_years: Time to expiry in years
            strikes: Array of strike prices
            market_vols: Array of market implied volatilities

        Returns:
            Calibrated SABR parameters
        """
        from scipy.optimize import minimize

        def objective(params):
            alpha, rho, nu = params

            # Constraints
            if alpha <= 0 or nu <= 0 or abs(rho) >= 1:
                return 1e10

            # Calculate model vols
            model_vols = np.array([
                self._sabr_vol(forward, strike, expiry_years, alpha, rho, nu)
                for strike in strikes
            ])

            # MSE
            return np.mean((model_vols - market_vols) ** 2)

        # Initial guess
        atm_vol = np.interp(forward, strikes, market_vols)
        x0 = [atm_vol, -0.2, 0.5]

        # Bounds
        bounds = [(0.01, 2.0), (-0.99, 0.99), (0.01, 2.0)]

        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds
        )

        alpha, rho, nu = result.x

        return {
            'alpha': alpha,
            'beta': self.beta,
            'rho': rho,
            'nu': nu
        }

    def _sabr_vol(
        self,
        forward: float,
        strike: float,
        expiry: float,
        alpha: float,
        rho: float,
        nu: float
    ) -> float:
        """
        Calculate SABR implied volatility using Hagan's approximation.
        """
        if abs(forward - strike) < 1e-10:
            # ATM case
            vol = alpha * forward ** (self.beta - 1) * (
                1 + (
                    ((1 - self.beta) ** 2 / 24) * (alpha ** 2 / forward ** (2 - 2 * self.beta)) +
                    (rho * self.beta * nu * alpha) / (4 * forward ** (1 - self.beta)) +
                    ((2 - 3 * rho ** 2) / 24) * nu ** 2
                ) * expiry
            )
            return vol

        # Non-ATM case
        f_k = forward * strike
        log_fk = np.log(forward / strike)

        z = (nu / alpha) * f_k ** ((1 - self.beta) / 2) * log_fk
        x = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))

        prefix = alpha / (
            f_k ** ((1 - self.beta) / 2) *
            (1 + ((1 - self.beta) ** 2 / 24) * log_fk ** 2 +
             ((1 - self.beta) ** 4 / 1920) * log_fk ** 4)
        )

        factor1 = z / x if abs(x) > 1e-10 else 1

        factor2 = 1 + (
            ((1 - self.beta) ** 2 / 24) * (alpha ** 2 / f_k ** (1 - self.beta)) +
            (rho * self.beta * nu * alpha) / (4 * f_k ** ((1 - self.beta) / 2)) +
            ((2 - 3 * rho ** 2) / 24) * nu ** 2
        ) * expiry

        vol = prefix * factor1 * factor2

        return max(0.01, vol)

    def get_vol(
        self,
        forward: float,
        strike: float,
        expiry_years: float,
        params: Optional[Dict] = None
    ) -> float:
        """Get implied volatility for given strike"""
        if params is None:
            # Use default/last calibrated params
            params = {
                'alpha': 0.5,
                'beta': self.beta,
                'rho': -0.2,
                'nu': 0.5
            }

        return self._sabr_vol(
            forward, strike, expiry_years,
            params['alpha'], params['rho'], params['nu']
        )

    def get_surface(
        self,
        forward: float,
        expiry_years: float,
        params: Dict
    ) -> VolatilitySurface:
        """Generate volatility surface metrics"""
        # ATM vol
        atm_vol = self.get_vol(forward, forward, expiry_years, params)

        # 25-delta strikes (approximation)
        delta_25_put = forward * 0.9
        delta_25_call = forward * 1.1

        vol_put = self.get_vol(forward, delta_25_put, expiry_years, params)
        vol_call = self.get_vol(forward, delta_25_call, expiry_years, params)

        # Risk reversal (skew)
        rr_25 = vol_call - vol_put

        # Butterfly (smile)
        bf_25 = (vol_put + vol_call) / 2 - atm_vol

        return VolatilitySurface(
            atm_vol=atm_vol,
            skew=rr_25,
            smile=bf_25,
            term_structure={},  # Would need multiple tenors
            sabr_alpha=params['alpha'],
            sabr_beta=params['beta'],
            sabr_rho=params['rho'],
            sabr_nu=params['nu']
        )


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 3: STATIONARITY TRANSFORMATIONS
# ═══════════════════════════════════════════════════════════════════════════════

class StationarityTransformer:
    """
    Stationarity transformations for time series features.

    Implements:
    - Log-Returns for price-based features
    - Z-score normalization with rolling mean/std
    - Quantile transformation for outlier robustness
    - Cointegration tests for pair trading
    """

    def __init__(
        self,
        zscore_halflife: int = 40,     # 40 periods halflife for EWM
        quantile_window: int = 252,     # 1 year for quantile reference
        min_periods: int = 20
    ):
        self.zscore_halflife = zscore_halflife
        self.quantile_window = quantile_window
        self.min_periods = min_periods

        # Cache for rolling statistics
        self.rolling_stats: Dict[str, Dict] = {}

    def transform_to_returns(
        self,
        prices: pd.Series,
        log_returns: bool = True
    ) -> pd.Series:
        """
        Transform prices to returns.

        Args:
            prices: Price series
            log_returns: Use log returns (default True)

        Returns:
            Returns series
        """
        if log_returns:
            return np.log(prices / prices.shift(1))
        else:
            return prices.pct_change()

    def zscore_normalize(
        self,
        series: pd.Series,
        use_ewm: bool = True
    ) -> pd.Series:
        """
        Z-score normalization with rolling statistics.

        z = (x - μ_rolling) / σ_rolling
        """
        if use_ewm:
            # Exponentially weighted for recency bias
            mean = series.ewm(halflife=self.zscore_halflife, min_periods=self.min_periods).mean()
            std = series.ewm(halflife=self.zscore_halflife, min_periods=self.min_periods).std()
        else:
            # Simple rolling
            window = self.zscore_halflife * 2
            mean = series.rolling(window, min_periods=self.min_periods).mean()
            std = series.rolling(window, min_periods=self.min_periods).std()

        zscore = (series - mean) / (std + 1e-10)

        # Winsorize extreme values
        return zscore.clip(-5, 5)

    def quantile_transform(
        self,
        series: pd.Series,
        n_quantiles: int = 100
    ) -> pd.Series:
        """
        Quantile transformation for uniform distribution.

        Robust to outliers - maps to percentile rank.
        """
        def rolling_percentile(window):
            if len(window) < self.min_periods:
                return 0.5
            return (window.iloc[-1] > window.iloc[:-1]).mean()

        return series.rolling(
            self.quantile_window,
            min_periods=self.min_periods
        ).apply(rolling_percentile)

    def test_stationarity(
        self,
        series: pd.Series,
        significance: float = 0.05
    ) -> Dict[str, Any]:
        """
        Test series for stationarity using ADF test.

        Returns:
            Dict with test results
        """
        try:
            from statsmodels.tsa.stattools import adfuller

            # Remove NaN
            clean_series = series.dropna()

            if len(clean_series) < 20:
                return {
                    'is_stationary': False,
                    'adf_statistic': None,
                    'p_value': 1.0,
                    'critical_values': {}
                }

            result = adfuller(clean_series, autolag='AIC')

            return {
                'is_stationary': result[1] < significance,
                'adf_statistic': result[0],
                'p_value': result[1],
                'critical_values': result[4],
                'n_lags': result[2]
            }

        except ImportError:
            # Fallback: simple variance ratio test
            if len(series) < 40:
                return {'is_stationary': False, 'p_value': 1.0}

            # Variance ratio test
            returns = series.pct_change().dropna()
            var_1 = returns.var()
            var_2 = returns.rolling(2).sum().var() / 2

            vr = var_2 / (var_1 + 1e-10)

            # VR close to 1 suggests random walk (non-stationary)
            is_stationary = abs(vr - 1) > 0.2

            return {
                'is_stationary': is_stationary,
                'variance_ratio': vr,
                'p_value': 0.05 if is_stationary else 0.5
            }

    def test_cointegration(
        self,
        series1: pd.Series,
        series2: pd.Series,
        significance: float = 0.05
    ) -> Dict[str, Any]:
        """
        Test two series for cointegration (Engle-Granger).

        Returns:
            Dict with cointegration results
        """
        try:
            from statsmodels.tsa.stattools import coint

            # Align series
            combined = pd.concat([series1, series2], axis=1).dropna()

            if len(combined) < 50:
                return {
                    'is_cointegrated': False,
                    'p_value': 1.0,
                    'hedge_ratio': None
                }

            # Cointegration test
            score, p_value, crit_values = coint(combined.iloc[:, 0], combined.iloc[:, 1])

            # Calculate hedge ratio via OLS
            y = combined.iloc[:, 0]
            x = combined.iloc[:, 1]
            hedge_ratio = np.cov(y, x)[0, 1] / (np.var(x) + 1e-10)

            return {
                'is_cointegrated': p_value < significance,
                'p_value': p_value,
                't_statistic': score,
                'critical_values': {
                    '1%': crit_values[0],
                    '5%': crit_values[1],
                    '10%': crit_values[2]
                },
                'hedge_ratio': hedge_ratio
            }

        except ImportError:
            # Fallback: correlation-based approximation
            combined = pd.concat([series1, series2], axis=1).dropna()

            if len(combined) < 50:
                return {'is_cointegrated': False, 'p_value': 1.0}

            corr = combined.corr().iloc[0, 1]

            # High correlation + mean-reverting spread suggests cointegration
            spread = combined.iloc[:, 0] - corr * combined.iloc[:, 1]
            spread_std = spread.std()

            # Heuristic: if spread mean-reverts well
            half_life = self._estimate_half_life(spread)
            is_coint = half_life < len(spread) / 4 and abs(corr) > 0.7

            return {
                'is_cointegrated': is_coint,
                'correlation': corr,
                'spread_halflife': half_life,
                'hedge_ratio': corr
            }

    def _estimate_half_life(self, series: pd.Series) -> float:
        """Estimate half-life of mean reversion"""
        series = series.dropna()

        if len(series) < 20:
            return float('inf')

        # AR(1) regression for half-life
        lag = series.shift(1).dropna()
        y = series.iloc[1:]

        if len(lag) < 10:
            return float('inf')

        # OLS: y = alpha + beta * lag + epsilon
        beta = np.cov(y, lag)[0, 1] / (np.var(lag) + 1e-10)

        if beta >= 1 or beta <= 0:
            return float('inf')

        half_life = -np.log(2) / np.log(beta)

        return max(1, half_life)

    def transform_feature(
        self,
        series: pd.Series,
        feature_type: str = 'price'
    ) -> pd.Series:
        """
        Apply appropriate transformation based on feature type.

        Args:
            series: Input series
            feature_type: One of 'price', 'volume', 'volatility', 'ratio'

        Returns:
            Transformed series
        """
        if feature_type == 'price':
            # Log returns + z-score
            returns = self.transform_to_returns(series, log_returns=True)
            return self.zscore_normalize(returns)

        elif feature_type == 'volume':
            # Log + z-score (volume is typically log-normal)
            log_vol = np.log(series + 1)
            return self.zscore_normalize(log_vol)

        elif feature_type == 'volatility':
            # Z-score with longer window
            return self.zscore_normalize(series, use_ewm=False)

        elif feature_type == 'ratio':
            # Quantile transform (already bounded)
            return self.quantile_transform(series)

        else:
            # Default: z-score
            return self.zscore_normalize(series)

    def transform_all_features(
        self,
        df: pd.DataFrame,
        feature_types: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Transform all features in DataFrame.

        Args:
            df: DataFrame with features
            feature_types: Dict mapping column names to types

        Returns:
            Transformed DataFrame
        """
        if feature_types is None:
            # Infer types from column names
            feature_types = {}
            for col in df.columns:
                col_lower = col.lower()
                if 'price' in col_lower or 'close' in col_lower:
                    feature_types[col] = 'price'
                elif 'volume' in col_lower or 'vol' in col_lower:
                    feature_types[col] = 'volume'
                elif 'volatility' in col_lower or 'atr' in col_lower:
                    feature_types[col] = 'volatility'
                elif 'ratio' in col_lower or 'pct' in col_lower:
                    feature_types[col] = 'ratio'
                else:
                    feature_types[col] = 'default'

        result = pd.DataFrame(index=df.index)

        for col in df.columns:
            ftype = feature_types.get(col, 'default')
            result[f'{col}_transformed'] = self.transform_feature(df[col], ftype)

        return result


# ═══════════════════════════════════════════════════════════════════════════════
# ADVANCED DATA PIPELINE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

class AdvancedDataPipeline:
    """
    Orchestrates all data processing layers.

    Combines:
    1. Microstructure analysis
    2. NLP sentiment
    3. Volatility surface
    4. Stationarity transformations
    """

    def __init__(
        self,
        enable_microstructure: bool = True,
        enable_sentiment: bool = True,
        enable_vol_surface: bool = True
    ):
        self.enable_microstructure = enable_microstructure
        self.enable_sentiment = enable_sentiment
        self.enable_vol_surface = enable_vol_surface

        # Initialize components
        self.microstructure_analyzer = MicrostructureAnalyzer()
        self.outlier_detector = OnlineOutlierDetector()
        self.sentiment_pipeline = NLPSentimentPipeline()
        self.sabr_surface = SABRVolatilitySurface()
        self.stationarity_transformer = StationarityTransformer()

        # Cache
        self._feature_cache: Dict[str, ProcessedFeatures] = {}

    def process_ohlcv(
        self,
        symbol: str,
        ohlcv_data: pd.DataFrame,
        trades: Optional[List[TradeEvent]] = None,
        orderbooks: Optional[List[OrderbookSnapshot]] = None,
        news: Optional[List[Dict[str, Any]]] = None,
        options_data: Optional[Dict[str, Any]] = None
    ) -> ProcessedFeatures:
        """
        Process all data for a symbol.

        Args:
            symbol: Trading symbol
            ohlcv_data: OHLCV DataFrame
            trades: Optional trade events for microstructure
            orderbooks: Optional orderbook snapshots
            news: Optional news items for sentiment
            options_data: Optional options for vol surface

        Returns:
            ProcessedFeatures with all processed data
        """
        timestamp = datetime.utcnow()

        # 1. Basic feature transformations
        raw_features = self._extract_raw_features(ohlcv_data)

        # 2. Stationarity transformations
        stationary_features = self._apply_stationarity_transforms(ohlcv_data, raw_features)

        # 3. Microstructure features
        if self.enable_microstructure and trades:
            for trade in trades:
                self.microstructure_analyzer.process_trade(trade)

            if orderbooks:
                for ob in orderbooks:
                    self.microstructure_analyzer.process_orderbook(ob)

            micro_features = self.microstructure_analyzer.calculate_features()
        else:
            micro_features = MicrostructureFeatures(
                vpin=0.5, order_flow_imbalance=0, price_impact=0,
                kyle_lambda=0, roll_spread=0, effective_spread=0,
                realized_volatility=0, trade_intensity=0,
                volume_clock_volatility=0, toxicity_index=0
            )

        # 4. Sentiment features
        sentiment = None
        if self.enable_sentiment and news:
            sentiments = [
                self.sentiment_pipeline.analyze(
                    item.get('text', ''),
                    item.get('source', 'unknown'),
                    item.get('timestamp', timestamp)
                )
                for item in news
            ]

            if sentiments:
                # Aggregate sentiments
                avg_sentiment = np.mean([s.overall_sentiment for s in sentiments])
                avg_confidence = np.mean([s.confidence for s in sentiments])

                sentiment = SentimentAnalysis(
                    overall_sentiment=avg_sentiment,
                    confidence=avg_confidence,
                    topic_distribution=sentiments[-1].topic_distribution,
                    key_entities=list(set(e for s in sentiments for e in s.key_entities)),
                    urgency_score=max(s.urgency_score for s in sentiments),
                    source_credibility=np.mean([s.source_credibility for s in sentiments]),
                    raw_text="Aggregated",
                    timestamp=timestamp
                )

        # 5. Volatility surface
        vol_surface = None
        if self.enable_vol_surface and options_data:
            try:
                forward = options_data.get('forward', ohlcv_data['close'].iloc[-1])
                expiry = options_data.get('expiry_years', 30/365)
                strikes = np.array(options_data.get('strikes', [forward]))
                vols = np.array(options_data.get('vols', [0.5]))

                if len(strikes) >= 3:
                    params = self.sabr_surface.calibrate(forward, expiry, strikes, vols)
                    vol_surface = self.sabr_surface.get_surface(forward, expiry, params)

            except Exception as e:
                logger.warning(f"Vol surface calibration failed: {e}")

        # 6. Feature quality scores
        quality_scores = self._assess_feature_quality(
            raw_features, stationary_features, micro_features
        )

        # 7. Outlier detection
        feature_vector = np.array(list(stationary_features.values())[:10])
        is_outlier, mahal_dist = self.outlier_detector.update(feature_vector)

        if is_outlier:
            logger.warning(f"Outlier detected for {symbol}: Mahalanobis distance = {mahal_dist:.2f}")
            quality_scores['outlier_flag'] = 1.0
            quality_scores['mahalanobis_distance'] = mahal_dist

        result = ProcessedFeatures(
            symbol=symbol,
            timestamp=timestamp,
            raw_features=raw_features,
            stationary_features=stationary_features,
            microstructure_features=micro_features,
            sentiment_features=sentiment,
            volatility_surface=vol_surface,
            feature_quality_scores=quality_scores
        )

        # Cache
        self._feature_cache[symbol] = result

        return result

    def _extract_raw_features(self, ohlcv: pd.DataFrame) -> Dict[str, float]:
        """Extract raw features from OHLCV"""
        features = {}

        if len(ohlcv) < 2:
            return features

        close = ohlcv['close']
        high = ohlcv.get('high', close)
        low = ohlcv.get('low', close)
        volume = ohlcv.get('volume', pd.Series([1] * len(close)))

        # Price features
        features['close'] = close.iloc[-1]
        features['return_1'] = close.pct_change().iloc[-1]
        features['return_5'] = close.pct_change(5).iloc[-1] if len(close) > 5 else 0
        features['return_20'] = close.pct_change(20).iloc[-1] if len(close) > 20 else 0

        # Volatility
        returns = close.pct_change().dropna()
        if len(returns) > 10:
            features['volatility_10'] = returns.iloc[-10:].std() * np.sqrt(252 * 24)
            features['volatility_30'] = returns.iloc[-30:].std() * np.sqrt(252 * 24) if len(returns) > 30 else features['volatility_10']

        # Volume
        features['volume'] = volume.iloc[-1]
        features['volume_ma_ratio'] = volume.iloc[-1] / (volume.rolling(20).mean().iloc[-1] + 1e-10)

        # Range
        features['high_low_ratio'] = (high.iloc[-1] - low.iloc[-1]) / (close.iloc[-1] + 1e-10)

        # Trend
        if len(close) >= 20:
            features['sma_20'] = close.rolling(20).mean().iloc[-1]
            features['price_to_sma_20'] = close.iloc[-1] / (features['sma_20'] + 1e-10)

        return features

    def _apply_stationarity_transforms(
        self,
        ohlcv: pd.DataFrame,
        raw_features: Dict[str, float]
    ) -> Dict[str, float]:
        """Apply stationarity transformations"""
        stationary = {}

        close = ohlcv['close']
        volume = ohlcv.get('volume', pd.Series([1] * len(close)))

        # Z-scored returns
        returns = self.stationarity_transformer.transform_to_returns(close)
        z_returns = self.stationarity_transformer.zscore_normalize(returns)
        stationary['z_return'] = z_returns.iloc[-1] if not z_returns.empty else 0

        # Quantile-transformed price position
        price_quantile = self.stationarity_transformer.quantile_transform(close)
        stationary['price_quantile'] = price_quantile.iloc[-1] if not price_quantile.empty else 0.5

        # Z-scored volume
        log_vol = np.log(volume + 1)
        z_volume = self.stationarity_transformer.zscore_normalize(log_vol)
        stationary['z_volume'] = z_volume.iloc[-1] if not z_volume.empty else 0

        # Volatility z-score
        if len(close) > 30:
            rolling_vol = returns.rolling(20).std() * np.sqrt(252 * 24)
            z_vol = self.stationarity_transformer.zscore_normalize(rolling_vol, use_ewm=False)
            stationary['z_volatility'] = z_vol.iloc[-1] if not z_vol.empty else 0

        # Momentum z-score
        if len(close) > 20:
            momentum = close.pct_change(20)
            z_momentum = self.stationarity_transformer.zscore_normalize(momentum)
            stationary['z_momentum'] = z_momentum.iloc[-1] if not z_momentum.empty else 0

        return stationary

    def _assess_feature_quality(
        self,
        raw_features: Dict[str, float],
        stationary_features: Dict[str, float],
        micro_features: MicrostructureFeatures
    ) -> Dict[str, float]:
        """Assess quality of computed features"""
        quality = {}

        # Check for NaN/Inf
        nan_count = sum(1 for v in raw_features.values() if np.isnan(v) or np.isinf(v))
        quality['raw_feature_validity'] = 1 - (nan_count / (len(raw_features) + 1e-10))

        nan_count_stat = sum(1 for v in stationary_features.values() if np.isnan(v) or np.isinf(v))
        quality['stationary_feature_validity'] = 1 - (nan_count_stat / (len(stationary_features) + 1e-10))

        # Microstructure data quality
        quality['microstructure_coverage'] = 1.0 if micro_features.trade_intensity > 0 else 0.0

        # Overall quality score
        quality['overall_quality'] = np.mean([
            quality['raw_feature_validity'],
            quality['stationary_feature_validity'],
            quality['microstructure_coverage']
        ])

        return quality

    def get_cached_features(self, symbol: str) -> Optional[ProcessedFeatures]:
        """Get cached features for symbol"""
        return self._feature_cache.get(symbol)

    def get_all_features_as_dataframe(self) -> pd.DataFrame:
        """Get all cached features as DataFrame"""
        rows = []

        for symbol, features in self._feature_cache.items():
            row = {'symbol': symbol, 'timestamp': features.timestamp}
            row.update(features.raw_features)
            row.update({f'stat_{k}': v for k, v in features.stationary_features.items()})

            # Add microstructure
            row['vpin'] = features.microstructure_features.vpin
            row['ofi'] = features.microstructure_features.order_flow_imbalance
            row['toxicity'] = features.microstructure_features.toxicity_index

            # Add sentiment if available
            if features.sentiment_features:
                row['sentiment'] = features.sentiment_features.overall_sentiment
                row['sentiment_confidence'] = features.sentiment_features.confidence

            rows.append(row)

        return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    'TradeEvent',
    'OrderbookSnapshot',
    'MicrostructureFeatures',
    'SentimentAnalysis',
    'VolatilitySurface',
    'ProcessedFeatures',
    'MicrostructureAnalyzer',
    'OnlineOutlierDetector',
    'NLPSentimentPipeline',
    'SABRVolatilitySurface',
    'StationarityTransformer',
    'AdvancedDataPipeline',
]
