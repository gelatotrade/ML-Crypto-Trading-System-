"""
Drift Protection Pipeline - Comprehensive drift detection and mitigation

This pipeline protects against model drift through:
1. Statistical drift detection (KS-Test, PSI, CUSUM)
2. Performance monitoring (Sharpe, Drawdown, Hit-Rate)
3. Rolling window retraining
4. Regime-specific model ensemble
5. Online learning capabilities
6. Automatic risk controls when drift detected

Drift Types:
- Data Drift: Input feature distributions change
- Concept Drift: Relationship between features and target changes
- Performance Drift: Model accuracy degrades over time
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
from collections import deque
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class DriftType(Enum):
    """Types of drift detected"""
    DATA_DRIFT = "data_drift"           # Feature distribution changed
    CONCEPT_DRIFT = "concept_drift"     # Feature-target relationship changed
    PERFORMANCE_DRIFT = "performance_drift"  # Model accuracy degraded
    REGIME_CHANGE = "regime_change"     # Market regime shifted


class DriftSeverity(Enum):
    """Severity level of detected drift"""
    NONE = "none"           # No drift detected
    LOW = "low"             # Minor drift, continue monitoring
    MEDIUM = "medium"       # Significant drift, consider retraining
    HIGH = "high"           # Severe drift, trigger retraining
    CRITICAL = "critical"   # Emergency, reduce positions immediately


class ModelAction(Enum):
    """Recommended action based on drift"""
    CONTINUE = "continue"           # No action needed
    MONITOR = "monitor"             # Increase monitoring frequency
    RETRAIN = "retrain"             # Retrain model with recent data
    SWITCH_REGIME = "switch_regime" # Switch to different regime model
    REDUCE_EXPOSURE = "reduce_exposure"  # Reduce position sizes
    HALT_TRADING = "halt_trading"   # Stop trading until resolved


@dataclass
class DriftAlert:
    """Alert for detected drift"""
    timestamp: datetime
    drift_type: DriftType
    severity: DriftSeverity
    feature: Optional[str]  # Which feature drifted (if applicable)
    metric_value: float     # The drift metric value
    threshold: float        # The threshold that was exceeded
    recommended_action: ModelAction
    description: str


@dataclass
class PerformanceMetrics:
    """Real-time performance tracking"""
    timestamp: datetime
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    hit_rate: float = 0.0          # % of profitable trades
    profit_factor: float = 0.0     # Gross profit / gross loss
    avg_win: float = 0.0
    avg_loss: float = 0.0
    win_loss_ratio: float = 0.0
    num_trades: int = 0
    pnl_today: float = 0.0
    pnl_week: float = 0.0
    pnl_month: float = 0.0


@dataclass
class DriftReport:
    """Comprehensive drift analysis report"""
    timestamp: datetime
    overall_severity: DriftSeverity
    recommended_action: ModelAction

    # Individual drift scores
    data_drift_score: float = 0.0
    concept_drift_score: float = 0.0
    performance_drift_score: float = 0.0

    # Detailed alerts
    alerts: List[DriftAlert] = field(default_factory=list)

    # Feature-level drift
    drifted_features: List[str] = field(default_factory=list)
    stable_features: List[str] = field(default_factory=list)

    # Performance summary
    current_performance: Optional[PerformanceMetrics] = None
    baseline_performance: Optional[PerformanceMetrics] = None

    # Recommendations
    recommendations: List[str] = field(default_factory=list)


@dataclass
class DriftConfig:
    """Configuration for drift detection"""
    # Statistical test thresholds
    ks_test_threshold: float = 0.1          # KS statistic threshold
    psi_threshold: float = 0.2              # Population Stability Index threshold
    psi_warning_threshold: float = 0.1      # PSI warning level

    # Performance thresholds
    sharpe_min_threshold: float = 0.5       # Minimum acceptable Sharpe
    sharpe_drop_threshold: float = 0.5      # Sharpe drop from baseline
    drawdown_threshold: float = 0.10        # Max drawdown trigger (10%)
    hit_rate_min: float = 0.45              # Minimum hit rate

    # Monitoring windows
    baseline_window_days: int = 30          # Baseline period for comparison
    detection_window_days: int = 7          # Recent window for drift detection
    retraining_frequency_days: int = 7      # How often to retrain

    # Risk controls
    max_position_reduction: float = 0.5     # Reduce positions by 50% on high drift
    halt_on_critical: bool = True           # Halt trading on critical drift

    # Online learning
    online_learning_enabled: bool = True
    online_learning_rate: float = 0.01


class StatisticalDriftDetector:
    """
    Statistical tests for drift detection

    Implements:
    - Kolmogorov-Smirnov Test (KS-Test)
    - Population Stability Index (PSI)
    - CUSUM (Cumulative Sum) for change detection
    - Page-Hinkley Test for concept drift
    """

    def __init__(self, config: DriftConfig):
        self.config = config

    def ks_test(
        self,
        baseline: np.ndarray,
        current: np.ndarray
    ) -> Tuple[float, float, bool]:
        """
        Kolmogorov-Smirnov test for distribution comparison

        Returns:
            Tuple of (statistic, p_value, is_drifted)
        """
        if len(baseline) < 10 or len(current) < 10:
            return 0.0, 1.0, False

        statistic, p_value = stats.ks_2samp(baseline, current)
        is_drifted = statistic > self.config.ks_test_threshold

        return statistic, p_value, is_drifted

    def calculate_psi(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10
    ) -> Tuple[float, str]:
        """
        Population Stability Index (PSI)

        PSI < 0.1: No significant change
        0.1 <= PSI < 0.2: Moderate change, monitor
        PSI >= 0.2: Significant change, action needed

        Returns:
            Tuple of (psi_value, interpretation)
        """
        if len(baseline) < 10 or len(current) < 10:
            return 0.0, "insufficient_data"

        # Create bins from baseline
        try:
            bins = np.percentile(baseline, np.linspace(0, 100, n_bins + 1))
            bins[0] = -np.inf
            bins[-1] = np.inf

            # Count frequencies
            baseline_counts = np.histogram(baseline, bins=bins)[0]
            current_counts = np.histogram(current, bins=bins)[0]

            # Convert to proportions (add small epsilon to avoid division by zero)
            epsilon = 1e-10
            baseline_props = (baseline_counts + epsilon) / (len(baseline) + epsilon * n_bins)
            current_props = (current_counts + epsilon) / (len(current) + epsilon * n_bins)

            # Calculate PSI
            psi = np.sum((current_props - baseline_props) * np.log(current_props / baseline_props))

            # Interpretation
            if psi < self.config.psi_warning_threshold:
                interpretation = "stable"
            elif psi < self.config.psi_threshold:
                interpretation = "moderate_drift"
            else:
                interpretation = "significant_drift"

            return float(psi), interpretation

        except Exception as e:
            logger.warning(f"PSI calculation failed: {e}")
            return 0.0, "error"

    def cusum_test(
        self,
        data: np.ndarray,
        threshold: float = 5.0,
        drift_magnitude: float = 1.0
    ) -> Tuple[List[int], float]:
        """
        CUSUM (Cumulative Sum) test for change point detection

        Returns:
            Tuple of (change_points, max_cusum_value)
        """
        if len(data) < 20:
            return [], 0.0

        # Normalize data
        mean = np.mean(data[:len(data)//2])  # Use first half as reference
        std = np.std(data[:len(data)//2]) + 1e-10

        normalized = (data - mean) / std

        # Calculate CUSUM
        cusum_pos = np.zeros(len(data))
        cusum_neg = np.zeros(len(data))
        change_points = []

        for i in range(1, len(data)):
            cusum_pos[i] = max(0, cusum_pos[i-1] + normalized[i] - drift_magnitude)
            cusum_neg[i] = max(0, cusum_neg[i-1] - normalized[i] - drift_magnitude)

            if cusum_pos[i] > threshold or cusum_neg[i] > threshold:
                change_points.append(i)
                # Reset CUSUM after detection
                cusum_pos[i] = 0
                cusum_neg[i] = 0

        max_cusum = max(np.max(cusum_pos), np.max(cusum_neg))

        return change_points, float(max_cusum)

    def page_hinkley_test(
        self,
        data: np.ndarray,
        delta: float = 0.005,
        lambda_threshold: float = 50.0
    ) -> Tuple[bool, int, float]:
        """
        Page-Hinkley test for concept drift detection

        Good for detecting gradual changes in mean

        Returns:
            Tuple of (drift_detected, change_point, test_statistic)
        """
        if len(data) < 20:
            return False, -1, 0.0

        n = len(data)
        cumsum = 0.0
        min_cumsum = 0.0
        change_point = -1

        for i in range(n):
            cumsum += data[i] - np.mean(data[:i+1]) - delta
            min_cumsum = min(min_cumsum, cumsum)

            if cumsum - min_cumsum > lambda_threshold:
                return True, i, cumsum - min_cumsum

        return False, -1, cumsum - min_cumsum


class PerformanceMonitor:
    """
    Real-time performance monitoring

    Tracks:
    - Rolling Sharpe ratio
    - Drawdown analysis
    - Hit rate and profit factor
    - Trade-level statistics
    """

    def __init__(
        self,
        config: DriftConfig,
        risk_free_rate: float = 0.05
    ):
        self.config = config
        self.risk_free_rate = risk_free_rate / 252  # Daily

        # Historical data
        self._returns: deque = deque(maxlen=252 * 2)  # 2 years
        self._trades: List[Dict] = []
        self._equity_curve: deque = deque(maxlen=252 * 2)
        self._peak_equity: float = 0.0

        # Baseline metrics
        self.baseline_metrics: Optional[PerformanceMetrics] = None
        self.baseline_set_date: Optional[datetime] = None

    def add_return(self, return_value: float, timestamp: datetime):
        """Add a return observation"""
        self._returns.append((timestamp, return_value))

    def add_trade(self, trade: Dict):
        """
        Add a completed trade

        trade = {
            'timestamp': datetime,
            'symbol': str,
            'side': 'long' or 'short',
            'pnl': float,
            'pnl_pct': float,
            'duration': timedelta
        }
        """
        self._trades.append(trade)

    def add_equity(self, equity: float, timestamp: datetime):
        """Add equity point"""
        self._equity_curve.append((timestamp, equity))
        self._peak_equity = max(self._peak_equity, equity)

    def calculate_metrics(
        self,
        window_days: Optional[int] = None
    ) -> PerformanceMetrics:
        """Calculate performance metrics for given window"""
        now = datetime.utcnow()

        # Filter by window
        if window_days:
            cutoff = now - timedelta(days=window_days)
            returns = [r for t, r in self._returns if t >= cutoff]
            trades = [t for t in self._trades if t['timestamp'] >= cutoff]
        else:
            returns = [r for _, r in self._returns]
            trades = self._trades

        metrics = PerformanceMetrics(timestamp=now)

        if not returns:
            return metrics

        returns_arr = np.array(returns)

        # Sharpe Ratio (annualized)
        if len(returns_arr) > 1 and np.std(returns_arr) > 0:
            excess_returns = returns_arr - self.risk_free_rate
            metrics.sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

        # Sortino Ratio
        downside_returns = returns_arr[returns_arr < 0]
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns)
            if downside_std > 0:
                metrics.sortino_ratio = (np.mean(returns_arr) - self.risk_free_rate) / downside_std * np.sqrt(252)

        # Drawdown
        if self._equity_curve:
            equities = [e for _, e in self._equity_curve]
            if window_days:
                cutoff_idx = max(0, len(equities) - window_days)
                equities = equities[cutoff_idx:]

            if equities:
                peak = equities[0]
                max_dd = 0.0
                for eq in equities:
                    peak = max(peak, eq)
                    dd = (peak - eq) / peak if peak > 0 else 0
                    max_dd = max(max_dd, dd)
                metrics.max_drawdown = max_dd

                # Current drawdown
                if self._peak_equity > 0:
                    metrics.current_drawdown = (self._peak_equity - equities[-1]) / self._peak_equity

        # Trade statistics
        if trades:
            metrics.num_trades = len(trades)
            pnls = [t['pnl'] for t in trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]

            metrics.hit_rate = len(wins) / len(pnls) if pnls else 0

            if wins:
                metrics.avg_win = np.mean(wins)
            if losses:
                metrics.avg_loss = abs(np.mean(losses))

            if metrics.avg_loss > 0:
                metrics.win_loss_ratio = metrics.avg_win / metrics.avg_loss

            gross_profit = sum(wins) if wins else 0
            gross_loss = abs(sum(losses)) if losses else 0
            if gross_loss > 0:
                metrics.profit_factor = gross_profit / gross_loss

            # PnL periods
            today = now.date()
            week_ago = now - timedelta(days=7)
            month_ago = now - timedelta(days=30)

            metrics.pnl_today = sum(t['pnl'] for t in trades if t['timestamp'].date() == today)
            metrics.pnl_week = sum(t['pnl'] for t in trades if t['timestamp'] >= week_ago)
            metrics.pnl_month = sum(t['pnl'] for t in trades if t['timestamp'] >= month_ago)

        return metrics

    def set_baseline(self):
        """Set current performance as baseline"""
        self.baseline_metrics = self.calculate_metrics(
            window_days=self.config.baseline_window_days
        )
        self.baseline_set_date = datetime.utcnow()
        logger.info(f"Baseline set - Sharpe: {self.baseline_metrics.sharpe_ratio:.2f}")

    def check_performance_drift(self) -> Tuple[DriftSeverity, List[str]]:
        """
        Check for performance drift vs baseline

        Returns:
            Tuple of (severity, list of issues)
        """
        if not self.baseline_metrics:
            return DriftSeverity.NONE, ["No baseline set"]

        current = self.calculate_metrics(window_days=self.config.detection_window_days)
        issues = []
        severity = DriftSeverity.NONE

        # Check Sharpe ratio
        sharpe_drop = self.baseline_metrics.sharpe_ratio - current.sharpe_ratio
        if current.sharpe_ratio < self.config.sharpe_min_threshold:
            issues.append(f"Sharpe below minimum: {current.sharpe_ratio:.2f} < {self.config.sharpe_min_threshold}")
            severity = max(severity, DriftSeverity.HIGH)
        elif sharpe_drop > self.config.sharpe_drop_threshold:
            issues.append(f"Sharpe dropped: {sharpe_drop:.2f} from baseline")
            severity = max(severity, DriftSeverity.MEDIUM)

        # Check drawdown
        if current.max_drawdown > self.config.drawdown_threshold:
            issues.append(f"Max drawdown exceeded: {current.max_drawdown:.1%}")
            severity = max(severity, DriftSeverity.HIGH)

        if current.current_drawdown > self.config.drawdown_threshold * 0.7:
            issues.append(f"Current drawdown high: {current.current_drawdown:.1%}")
            severity = max(severity, DriftSeverity.MEDIUM)

        # Check hit rate
        if current.num_trades >= 20 and current.hit_rate < self.config.hit_rate_min:
            issues.append(f"Hit rate low: {current.hit_rate:.1%}")
            severity = max(severity, DriftSeverity.MEDIUM)

        return severity, issues


class RobustFeatureEngineer:
    """
    Drift-resistant feature engineering

    Creates features that are more stable across market regimes:
    - Risk-adjusted returns instead of absolute
    - Normalized indicators
    - Cross-sectional features (relative to universe)
    """

    def __init__(self):
        self._feature_stats: Dict[str, Dict] = {}

    def create_robust_features(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> pd.DataFrame:
        """
        Create drift-resistant features

        Args:
            df: OHLCV DataFrame
            symbol: Trading symbol

        Returns:
            DataFrame with robust features
        """
        result = df.copy()

        # Risk-adjusted returns (more stable than absolute)
        returns = df['close'].pct_change()
        volatility = returns.rolling(20).std()

        # Sharpe-like return (risk-adjusted)
        result['risk_adj_return'] = returns / (volatility + 1e-10)

        # Normalized returns (z-score)
        result['return_zscore'] = (returns - returns.rolling(60).mean()) / (returns.rolling(60).std() + 1e-10)

        # Percentile rank (0-1, very stable across time)
        result['return_percentile'] = returns.rolling(252).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100 if len(x) > 20 else 0.5,
            raw=False
        )

        # Volume-adjusted price change
        if 'volume' in df.columns:
            avg_volume = df['volume'].rolling(20).mean()
            volume_ratio = df['volume'] / (avg_volume + 1e-10)
            result['volume_adj_return'] = returns * np.log1p(volume_ratio)

        # Normalized volatility (percentile of historical)
        result['vol_percentile'] = volatility.rolling(252).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100 if len(x) > 20 else 0.5,
            raw=False
        )

        # Mean reversion indicator (normalized)
        sma_20 = df['close'].rolling(20).mean()
        sma_60 = df['close'].rolling(60).mean()
        result['mean_reversion_score'] = (df['close'] - sma_20) / (sma_60 + 1e-10)

        # Momentum (risk-adjusted)
        mom_20 = df['close'].pct_change(20)
        mom_vol = returns.rolling(20).std() * np.sqrt(20)
        result['risk_adj_momentum'] = mom_20 / (mom_vol + 1e-10)

        # Trend strength (normalized ADX-like)
        high_low = df['high'] - df['low']
        atr = high_low.rolling(14).mean()
        result['normalized_range'] = high_low / (atr + 1e-10)

        # RSI as probability (already 0-100, normalize to 0-1)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        result['rsi_normalized'] = rs / (1 + rs)

        return result

    def track_feature_stability(
        self,
        feature_name: str,
        values: np.ndarray
    ):
        """Track feature statistics for drift monitoring"""
        if feature_name not in self._feature_stats:
            self._feature_stats[feature_name] = {
                'baseline_mean': np.mean(values),
                'baseline_std': np.std(values),
                'baseline_values': values.copy(),
                'timestamp': datetime.utcnow()
            }
        else:
            # Update with exponential moving average
            alpha = 0.1
            self._feature_stats[feature_name]['baseline_mean'] = (
                alpha * np.mean(values) +
                (1 - alpha) * self._feature_stats[feature_name]['baseline_mean']
            )
            self._feature_stats[feature_name]['baseline_std'] = (
                alpha * np.std(values) +
                (1 - alpha) * self._feature_stats[feature_name]['baseline_std']
            )


class OnlineLearningAdapter:
    """
    Online learning capabilities for incremental model updates

    Supports:
    - Incremental updates with new data
    - Learning rate decay
    - Concept drift adaptation
    """

    def __init__(
        self,
        config: DriftConfig,
        base_model: Any = None
    ):
        self.config = config
        self.base_model = base_model
        self._update_count = 0
        self._last_update = None

        # Sliding window for online learning
        self._feature_buffer: deque = deque(maxlen=1000)
        self._target_buffer: deque = deque(maxlen=1000)

    def add_observation(
        self,
        features: np.ndarray,
        target: float,
        timestamp: datetime
    ):
        """Add new observation for online learning"""
        self._feature_buffer.append((timestamp, features))
        self._target_buffer.append((timestamp, target))

    def get_effective_learning_rate(self) -> float:
        """Get learning rate with decay"""
        # Decay learning rate over time
        decay_factor = 1.0 / (1.0 + 0.01 * self._update_count)
        return self.config.online_learning_rate * decay_factor

    def should_update(self, current_time: datetime) -> bool:
        """Check if model should be updated"""
        if not self.config.online_learning_enabled:
            return False

        if self._last_update is None:
            return True

        # Update at least once per day
        hours_since_update = (current_time - self._last_update).total_seconds() / 3600
        return hours_since_update >= 24

    def get_recent_data(
        self,
        window_hours: int = 168  # 1 week
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get recent data for retraining"""
        cutoff = datetime.utcnow() - timedelta(hours=window_hours)

        features = [f for t, f in self._feature_buffer if t >= cutoff]
        targets = [t for ts, t in self._target_buffer if ts >= cutoff]

        if not features:
            return np.array([]), np.array([])

        return np.array(features), np.array(targets)


class DriftProtectionPipeline:
    """
    Main drift protection pipeline

    Integrates all drift detection and mitigation components:
    1. Statistical drift detection
    2. Performance monitoring
    3. Robust feature engineering
    4. Online learning
    5. Automatic risk controls
    """

    def __init__(
        self,
        config: Optional[DriftConfig] = None
    ):
        self.config = config or DriftConfig()

        # Components
        self.statistical_detector = StatisticalDriftDetector(self.config)
        self.performance_monitor = PerformanceMonitor(self.config)
        self.feature_engineer = RobustFeatureEngineer()
        self.online_adapter = OnlineLearningAdapter(self.config)

        # State
        self._baseline_features: Dict[str, np.ndarray] = {}
        self._current_features: Dict[str, np.ndarray] = {}
        self._alerts: List[DriftAlert] = []
        self._last_check = None
        self._retraining_scheduled = False
        self._position_multiplier = 1.0  # Reduce when drift detected

        # Regime-specific models
        self._regime_models: Dict[str, Any] = {}
        self._current_regime: str = "neutral"

    def set_feature_baseline(
        self,
        feature_name: str,
        values: np.ndarray
    ):
        """Set baseline distribution for a feature"""
        self._baseline_features[feature_name] = values.copy()
        logger.info(f"Set baseline for feature: {feature_name} (n={len(values)})")

    def update_current_features(
        self,
        feature_name: str,
        values: np.ndarray
    ):
        """Update current feature values for drift comparison"""
        self._current_features[feature_name] = values.copy()

    def check_all_drift(self) -> DriftReport:
        """
        Run comprehensive drift check

        Returns:
            DriftReport with all drift analysis results
        """
        now = datetime.utcnow()
        report = DriftReport(
            timestamp=now,
            overall_severity=DriftSeverity.NONE,
            recommended_action=ModelAction.CONTINUE
        )

        # 1. Check data drift for each feature
        data_drift_scores = []
        for feature_name, baseline in self._baseline_features.items():
            if feature_name not in self._current_features:
                continue

            current = self._current_features[feature_name]

            # KS Test
            ks_stat, ks_pval, ks_drifted = self.statistical_detector.ks_test(baseline, current)

            # PSI
            psi_value, psi_interp = self.statistical_detector.calculate_psi(baseline, current)

            # Combined score
            drift_score = (ks_stat + psi_value) / 2
            data_drift_scores.append(drift_score)

            if ks_drifted or psi_interp == "significant_drift":
                report.drifted_features.append(feature_name)

                severity = DriftSeverity.HIGH if psi_value > 0.25 else DriftSeverity.MEDIUM
                alert = DriftAlert(
                    timestamp=now,
                    drift_type=DriftType.DATA_DRIFT,
                    severity=severity,
                    feature=feature_name,
                    metric_value=psi_value,
                    threshold=self.config.psi_threshold,
                    recommended_action=ModelAction.RETRAIN,
                    description=f"Feature '{feature_name}' drifted: PSI={psi_value:.3f}, KS={ks_stat:.3f}"
                )
                report.alerts.append(alert)
            else:
                report.stable_features.append(feature_name)

        if data_drift_scores:
            report.data_drift_score = np.mean(data_drift_scores)

        # 2. Check performance drift
        perf_severity, perf_issues = self.performance_monitor.check_performance_drift()
        report.performance_drift_score = perf_severity.value if isinstance(perf_severity.value, (int, float)) else 0

        if perf_severity != DriftSeverity.NONE:
            for issue in perf_issues:
                alert = DriftAlert(
                    timestamp=now,
                    drift_type=DriftType.PERFORMANCE_DRIFT,
                    severity=perf_severity,
                    feature=None,
                    metric_value=0,
                    threshold=0,
                    recommended_action=ModelAction.REDUCE_EXPOSURE if perf_severity == DriftSeverity.HIGH else ModelAction.MONITOR,
                    description=issue
                )
                report.alerts.append(alert)

        # 3. Get current and baseline performance
        report.current_performance = self.performance_monitor.calculate_metrics(
            window_days=self.config.detection_window_days
        )
        report.baseline_performance = self.performance_monitor.baseline_metrics

        # 4. Determine overall severity and action
        max_severity = DriftSeverity.NONE
        for alert in report.alerts:
            if alert.severity.value > max_severity.value if hasattr(max_severity, 'value') else 0:
                max_severity = alert.severity

        report.overall_severity = max_severity

        # 5. Determine recommended action
        if max_severity == DriftSeverity.CRITICAL:
            report.recommended_action = ModelAction.HALT_TRADING
            report.recommendations.append("CRITICAL: Halt trading immediately")
            self._position_multiplier = 0.0
        elif max_severity == DriftSeverity.HIGH:
            report.recommended_action = ModelAction.REDUCE_EXPOSURE
            report.recommendations.append("HIGH: Reduce position sizes by 50%")
            report.recommendations.append("Schedule immediate model retraining")
            self._position_multiplier = 0.5
            self._retraining_scheduled = True
        elif max_severity == DriftSeverity.MEDIUM:
            report.recommended_action = ModelAction.RETRAIN
            report.recommendations.append("MEDIUM: Schedule model retraining")
            report.recommendations.append("Increase monitoring frequency")
            self._position_multiplier = 0.75
            self._retraining_scheduled = True
        elif max_severity == DriftSeverity.LOW:
            report.recommended_action = ModelAction.MONITOR
            report.recommendations.append("LOW: Continue monitoring")
            self._position_multiplier = 1.0
        else:
            report.recommended_action = ModelAction.CONTINUE
            self._position_multiplier = 1.0

        # Add feature-specific recommendations
        if report.drifted_features:
            report.recommendations.append(
                f"Re-evaluate features: {', '.join(report.drifted_features[:5])}"
            )

        self._last_check = now
        self._alerts.extend(report.alerts)

        return report

    def get_position_multiplier(self) -> float:
        """
        Get position size multiplier based on drift status

        Returns value between 0 and 1 to reduce positions when drift detected
        """
        return self._position_multiplier

    def should_retrain(self) -> bool:
        """Check if model retraining is needed"""
        if self._retraining_scheduled:
            return True

        if self.performance_monitor.baseline_set_date:
            days_since = (datetime.utcnow() - self.performance_monitor.baseline_set_date).days
            if days_since >= self.config.retraining_frequency_days:
                return True

        return False

    def on_retraining_complete(self):
        """Called after model retraining"""
        self._retraining_scheduled = False
        self.performance_monitor.set_baseline()
        self._position_multiplier = 1.0
        logger.info("Retraining complete, baseline reset")

    def register_regime_model(self, regime_name: str, model: Any):
        """Register a model for a specific market regime"""
        self._regime_models[regime_name] = model
        logger.info(f"Registered model for regime: {regime_name}")

    def get_regime_model(self, regime: str) -> Optional[Any]:
        """Get model for current regime"""
        return self._regime_models.get(regime)

    def switch_regime(self, new_regime: str):
        """Switch to a different regime model"""
        if new_regime in self._regime_models:
            self._current_regime = new_regime
            logger.info(f"Switched to regime model: {new_regime}")

            # Reset position multiplier on regime change
            self._position_multiplier = 0.8  # Slight reduction during transition

            # Create alert
            alert = DriftAlert(
                timestamp=datetime.utcnow(),
                drift_type=DriftType.REGIME_CHANGE,
                severity=DriftSeverity.MEDIUM,
                feature=None,
                metric_value=0,
                threshold=0,
                recommended_action=ModelAction.SWITCH_REGIME,
                description=f"Regime changed to: {new_regime}"
            )
            self._alerts.append(alert)

    def get_recent_alerts(
        self,
        hours: int = 24,
        min_severity: DriftSeverity = DriftSeverity.LOW
    ) -> List[DriftAlert]:
        """Get recent alerts above minimum severity"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [
            a for a in self._alerts
            if a.timestamp >= cutoff and
            self._severity_to_int(a.severity) >= self._severity_to_int(min_severity)
        ]

    def _severity_to_int(self, severity: DriftSeverity) -> int:
        """Convert severity to integer for comparison"""
        mapping = {
            DriftSeverity.NONE: 0,
            DriftSeverity.LOW: 1,
            DriftSeverity.MEDIUM: 2,
            DriftSeverity.HIGH: 3,
            DriftSeverity.CRITICAL: 4
        }
        return mapping.get(severity, 0)

    def create_monitoring_summary(self) -> Dict[str, Any]:
        """Create summary for logging/display"""
        report = self.check_all_drift()

        return {
            "timestamp": report.timestamp.isoformat(),
            "overall_severity": report.overall_severity.value,
            "recommended_action": report.recommended_action.value,
            "position_multiplier": self._position_multiplier,
            "data_drift_score": report.data_drift_score,
            "drifted_features_count": len(report.drifted_features),
            "stable_features_count": len(report.stable_features),
            "alerts_count": len(report.alerts),
            "current_sharpe": report.current_performance.sharpe_ratio if report.current_performance else None,
            "current_drawdown": report.current_performance.current_drawdown if report.current_performance else None,
            "retraining_scheduled": self._retraining_scheduled,
            "current_regime": self._current_regime,
            "recommendations": report.recommendations[:3]  # Top 3
        }
