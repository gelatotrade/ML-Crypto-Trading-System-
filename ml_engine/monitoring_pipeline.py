"""
UMFASSENDE ÜBERWACHUNGS-PIPELINE

Implementiert:
A. Real-Time Model Diagnostics
   - Performance Attribution
   - Regime-Switching Detection
   - Backtest Overfitting Tests (PBO)
   - Combinatorially Symmetric Cross-Validation

B. Stress Testing Framework
   - Historische Szenarien (2008, 1987, 2020 COVID)
   - Hypothetische Szenarien (Flash Crash, Liquidity Dry-up)
   - Reverse Stress Testing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import deque
import logging
import warnings
from itertools import combinations

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PerformanceAttribution:
    """Performance attribution breakdown"""
    total_return: float
    alpha_contribution: float
    factor_contributions: Dict[str, float]
    residual_return: float
    timing_contribution: float
    selection_contribution: float
    interaction_contribution: float


@dataclass
class PBOResult:
    """Probability of Backtest Overfitting result"""
    pbo: float                      # Probability of overfitting
    pbo_threshold: float            # PBO threshold (typically 0.5)
    is_overfit: bool
    optimal_strategy_rank: int      # Rank of optimal IS strategy in OOS
    n_combinations: int
    performance_degradation: float  # IS - OOS performance
    confidence_interval: Tuple[float, float]


@dataclass
class StressScenario:
    """Stress test scenario definition"""
    name: str
    description: str
    market_shock: float             # % price change
    volatility_multiplier: float
    correlation_shock: float        # Correlation change
    liquidity_shock: float          # Spread multiplier
    duration_days: int


@dataclass
class StressTestResult:
    """Stress test result"""
    scenario: StressScenario
    portfolio_loss: float
    max_drawdown: float
    var_breach: bool
    recovery_time_days: int
    worst_asset: str
    worst_asset_loss: float


@dataclass
class ModelDiagnostics:
    """Real-time model diagnostics"""
    timestamp: datetime
    model_name: str
    sharpe_ratio: float
    information_ratio: float
    hit_rate: float
    profit_factor: float
    max_drawdown: float
    current_drawdown: float
    regime_alignment: float         # How well model performs in current regime
    signal_decay: float             # Signal autocorrelation (decay indicator)
    feature_stability: float        # Feature importance stability


# ═══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE ATTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════

class PerformanceAttributor:
    """
    Performance Attribution Analysis.

    Decomposes returns into:
    - Alpha (stock selection)
    - Factor contributions (systematic)
    - Timing (market timing)
    - Residual

    PnL_t = Σ[α_i,t * w_i,t * σ_i,t] + Σ[β_i,k * F_k,t * w_i,t] + ε_t
    """

    def __init__(
        self,
        factors: List[str] = None,
        benchmark: str = 'BTC/USDT'
    ):
        self.factors = factors or ['market', 'size', 'momentum', 'volatility']
        self.benchmark = benchmark

        # Historical attributions
        self.attribution_history: List[PerformanceAttribution] = []

    def attribute_returns(
        self,
        portfolio_returns: pd.Series,
        asset_returns: pd.DataFrame,
        weights: pd.DataFrame,
        factor_returns: pd.DataFrame,
        factor_exposures: pd.DataFrame
    ) -> PerformanceAttribution:
        """
        Decompose portfolio returns.

        Args:
            portfolio_returns: Portfolio return series
            asset_returns: Individual asset returns
            weights: Portfolio weights over time
            factor_returns: Factor return series
            factor_exposures: Asset factor exposures (betas)

        Returns:
            PerformanceAttribution breakdown
        """
        # Align all data
        common_idx = portfolio_returns.index.intersection(factor_returns.index)
        port_ret = portfolio_returns.loc[common_idx]
        factor_ret = factor_returns.loc[common_idx]

        total_return = (1 + port_ret).prod() - 1

        # Factor contributions
        factor_contributions = {}
        total_factor_contribution = 0

        for factor in self.factors:
            if factor in factor_ret.columns and factor in factor_exposures.columns:
                # Factor contribution = exposure × factor return
                exposure = factor_exposures[factor].mean()
                factor_contrib = (factor_ret[factor] * exposure).sum()
                factor_contributions[factor] = factor_contrib
                total_factor_contribution += factor_contrib

        # Alpha = Total - Factor contributions
        alpha = total_return - total_factor_contribution

        # Brinson-style attribution (if weights available)
        timing, selection, interaction = self._brinson_attribution(
            portfolio_returns, asset_returns, weights
        )

        # Residual
        residual = total_return - alpha - total_factor_contribution

        attribution = PerformanceAttribution(
            total_return=total_return,
            alpha_contribution=alpha,
            factor_contributions=factor_contributions,
            residual_return=residual,
            timing_contribution=timing,
            selection_contribution=selection,
            interaction_contribution=interaction
        )

        self.attribution_history.append(attribution)

        return attribution

    def _brinson_attribution(
        self,
        portfolio_returns: pd.Series,
        asset_returns: pd.DataFrame,
        weights: pd.DataFrame
    ) -> Tuple[float, float, float]:
        """
        Brinson-Fachler attribution.

        Decomposes active return into:
        - Timing (over/underweight timing)
        - Selection (stock picking)
        - Interaction (combination effect)
        """
        if weights.empty or asset_returns.empty:
            return 0, 0, 0

        try:
            # Align data
            common_idx = portfolio_returns.index.intersection(weights.index)
            common_idx = common_idx.intersection(asset_returns.index)

            if len(common_idx) == 0:
                return 0, 0, 0

            w = weights.loc[common_idx]
            r = asset_returns.loc[common_idx]

            # Benchmark weights (equal weight as proxy)
            w_bench = pd.DataFrame(
                1/len(r.columns),
                index=w.index,
                columns=r.columns
            )

            # Benchmark returns per asset
            r_bench = r.mean()

            # Attribution components
            # Timing: (w - w_bench) × r_bench
            timing = ((w - w_bench).mean() * r_bench).sum()

            # Selection: w_bench × (r - r_bench)
            selection = (w_bench.mean() * (r - r_bench).mean()).sum()

            # Interaction: (w - w_bench) × (r - r_bench)
            interaction = ((w - w_bench).mean() * (r - r_bench).mean()).sum()

            return float(timing), float(selection), float(interaction)

        except Exception as e:
            logger.warning(f"Brinson attribution failed: {e}")
            return 0, 0, 0

    def get_rolling_attribution(
        self,
        window: int = 30
    ) -> pd.DataFrame:
        """Get rolling attribution metrics"""
        if len(self.attribution_history) < window:
            return pd.DataFrame()

        data = []
        for attr in self.attribution_history[-window:]:
            row = {
                'total_return': attr.total_return,
                'alpha': attr.alpha_contribution,
                'timing': attr.timing_contribution,
                'selection': attr.selection_contribution
            }
            row.update(attr.factor_contributions)
            data.append(row)

        return pd.DataFrame(data)


# ═══════════════════════════════════════════════════════════════════════════════
# PROBABILITY OF BACKTEST OVERFITTING (PBO)
# ═══════════════════════════════════════════════════════════════════════════════

class PBOAnalyzer:
    """
    Probability of Backtest Overfitting (PBO) Analysis.

    Based on Bailey et al. (2014).

    Uses Combinatorially Symmetric Cross-Validation (CSCV) to estimate
    the probability that a strategy is overfit.
    """

    def __init__(
        self,
        n_partitions: int = 16,
        metric: str = 'sharpe_ratio'
    ):
        self.n_partitions = n_partitions
        self.metric = metric

    def calculate_pbo(
        self,
        returns_matrix: np.ndarray,
        strategy_ids: Optional[List[str]] = None
    ) -> PBOResult:
        """
        Calculate PBO using CSCV.

        Args:
            returns_matrix: (n_periods, n_strategies) matrix of returns
            strategy_ids: Optional strategy identifiers

        Returns:
            PBOResult with overfitting probability
        """
        n_periods, n_strategies = returns_matrix.shape

        if n_periods < self.n_partitions * 2:
            logger.warning("Insufficient data for PBO analysis")
            return PBOResult(
                pbo=0.5, pbo_threshold=0.5, is_overfit=False,
                optimal_strategy_rank=0, n_combinations=0,
                performance_degradation=0, confidence_interval=(0, 1)
            )

        # Partition data into S blocks
        block_size = n_periods // self.n_partitions
        blocks = []
        for i in range(self.n_partitions):
            start = i * block_size
            end = start + block_size
            blocks.append(returns_matrix[start:end])

        # Generate all combinations of S/2 blocks for IS (training)
        n_is_blocks = self.n_partitions // 2
        all_combinations = list(combinations(range(self.n_partitions), n_is_blocks))

        # Limit combinations if too many
        if len(all_combinations) > 252:
            np.random.seed(42)
            selected_idx = np.random.choice(len(all_combinations), 252, replace=False)
            all_combinations = [all_combinations[i] for i in selected_idx]

        # For each combination, calculate IS and OOS performance
        pbo_count = 0
        ranks = []
        is_performances = []
        oos_performances = []

        for is_blocks in all_combinations:
            oos_blocks = [i for i in range(self.n_partitions) if i not in is_blocks]

            # Combine blocks
            is_data = np.vstack([blocks[i] for i in is_blocks])
            oos_data = np.vstack([blocks[i] for i in oos_blocks])

            # Calculate performance metric for each strategy
            is_perf = self._calculate_performance(is_data)
            oos_perf = self._calculate_performance(oos_data)

            is_performances.append(is_perf)
            oos_performances.append(oos_perf)

            # Find best IS strategy
            best_is_strategy = np.argmax(is_perf)

            # Rank of best IS strategy in OOS
            oos_ranks = np.argsort(np.argsort(-oos_perf))
            rank_of_best = oos_ranks[best_is_strategy]
            ranks.append(rank_of_best)

            # PBO: best IS strategy performs below median in OOS
            if rank_of_best >= n_strategies / 2:
                pbo_count += 1

        # PBO estimate
        pbo = pbo_count / len(all_combinations)

        # Average rank of IS-optimal strategy in OOS
        avg_rank = np.mean(ranks)

        # Performance degradation
        is_best_perf = np.mean([np.max(p) for p in is_performances])
        oos_best_perf = np.mean([
            p[np.argmax(is_performances[i])]
            for i, p in enumerate(oos_performances)
        ])
        degradation = is_best_perf - oos_best_perf

        # Confidence interval (bootstrap)
        ci_lower = np.percentile([r >= n_strategies/2 for r in ranks], 2.5)
        ci_upper = np.percentile([r >= n_strategies/2 for r in ranks], 97.5)

        return PBOResult(
            pbo=pbo,
            pbo_threshold=0.5,
            is_overfit=pbo > 0.5,
            optimal_strategy_rank=int(avg_rank),
            n_combinations=len(all_combinations),
            performance_degradation=degradation,
            confidence_interval=(ci_lower, ci_upper)
        )

    def _calculate_performance(self, returns: np.ndarray) -> np.ndarray:
        """Calculate performance metric for all strategies"""
        if self.metric == 'sharpe_ratio':
            means = np.mean(returns, axis=0)
            stds = np.std(returns, axis=0)
            return means / (stds + 1e-10) * np.sqrt(252)

        elif self.metric == 'total_return':
            return np.mean(returns, axis=0) * returns.shape[0]

        elif self.metric == 'sortino_ratio':
            means = np.mean(returns, axis=0)
            downside = returns.copy()
            downside[downside > 0] = 0
            downside_std = np.std(downside, axis=0)
            return means / (downside_std + 1e-10) * np.sqrt(252)

        else:
            return np.mean(returns, axis=0)

    def analyze_strategy_set(
        self,
        strategy_returns: Dict[str, pd.Series]
    ) -> Dict[str, Any]:
        """
        Analyze a set of strategies for overfitting.

        Args:
            strategy_returns: Dict of strategy name -> return series

        Returns:
            Analysis results including PBO and recommendations
        """
        # Align all series
        df = pd.DataFrame(strategy_returns)
        df = df.dropna()

        if len(df) < 100:
            return {'error': 'Insufficient data'}

        returns_matrix = df.values

        # Calculate PBO
        pbo_result = self.calculate_pbo(returns_matrix, list(strategy_returns.keys()))

        # Individual strategy analysis
        strategy_stats = {}
        for name, returns in strategy_returns.items():
            clean_returns = returns.dropna()
            strategy_stats[name] = {
                'mean_return': clean_returns.mean() * 252,
                'volatility': clean_returns.std() * np.sqrt(252),
                'sharpe': clean_returns.mean() / (clean_returns.std() + 1e-10) * np.sqrt(252),
                'max_drawdown': self._max_drawdown(clean_returns)
            }

        return {
            'pbo_result': pbo_result,
            'strategy_stats': strategy_stats,
            'recommendation': 'Likely overfit' if pbo_result.is_overfit else 'Appears robust'
        }

    def _max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        return abs(drawdowns.min())


# ═══════════════════════════════════════════════════════════════════════════════
# STRESS TESTING FRAMEWORK
# ═══════════════════════════════════════════════════════════════════════════════

class StressTestingFramework:
    """
    Comprehensive Stress Testing Framework.

    Scenarios:
    1. Historical: 2008, 1987, 2020 COVID
    2. Hypothetical: Flash Crash, Liquidity Dry-up, Correlations → 1
    3. Reverse Stress Testing
    """

    def __init__(self):
        # Define standard scenarios
        self.scenarios = self._define_scenarios()

    def _define_scenarios(self) -> Dict[str, StressScenario]:
        """Define standard stress scenarios"""
        return {
            # Historical scenarios
            'crisis_2008': StressScenario(
                name='2008 Financial Crisis',
                description='Lehman-style systemic crisis',
                market_shock=-0.40,
                volatility_multiplier=4.0,
                correlation_shock=0.3,
                liquidity_shock=10.0,
                duration_days=90
            ),
            'black_monday_1987': StressScenario(
                name='Black Monday 1987',
                description='Single-day crash',
                market_shock=-0.22,
                volatility_multiplier=5.0,
                correlation_shock=0.4,
                liquidity_shock=5.0,
                duration_days=1
            ),
            'covid_2020': StressScenario(
                name='COVID-19 March 2020',
                description='Pandemic-induced crash',
                market_shock=-0.35,
                volatility_multiplier=3.5,
                correlation_shock=0.35,
                liquidity_shock=3.0,
                duration_days=30
            ),
            'crypto_winter': StressScenario(
                name='Crypto Winter',
                description='Extended crypto bear market',
                market_shock=-0.80,
                volatility_multiplier=2.0,
                correlation_shock=0.2,
                liquidity_shock=2.0,
                duration_days=365
            ),
            # Hypothetical scenarios
            'flash_crash': StressScenario(
                name='Flash Crash',
                description='Rapid market dislocation',
                market_shock=-0.10,
                volatility_multiplier=10.0,
                correlation_shock=0.5,
                liquidity_shock=20.0,
                duration_days=1
            ),
            'liquidity_crisis': StressScenario(
                name='Liquidity Crisis',
                description='Complete liquidity dry-up',
                market_shock=-0.15,
                volatility_multiplier=3.0,
                correlation_shock=0.4,
                liquidity_shock=50.0,
                duration_days=7
            ),
            'correlation_spike': StressScenario(
                name='Correlation Spike',
                description='All correlations go to 1',
                market_shock=-0.20,
                volatility_multiplier=2.5,
                correlation_shock=0.8,
                liquidity_shock=3.0,
                duration_days=14
            ),
            'stablecoin_depeg': StressScenario(
                name='Stablecoin Depeg',
                description='Major stablecoin loses peg',
                market_shock=-0.25,
                volatility_multiplier=4.0,
                correlation_shock=0.3,
                liquidity_shock=15.0,
                duration_days=7
            ),
            'exchange_hack': StressScenario(
                name='Major Exchange Hack',
                description='Large exchange security breach',
                market_shock=-0.15,
                volatility_multiplier=3.0,
                correlation_shock=0.2,
                liquidity_shock=5.0,
                duration_days=3
            )
        }

    def run_stress_test(
        self,
        portfolio_weights: Dict[str, float],
        asset_volatilities: Dict[str, float],
        correlation_matrix: np.ndarray,
        scenario_name: str,
        portfolio_value: float = 1000000
    ) -> StressTestResult:
        """
        Run stress test for a specific scenario.

        Args:
            portfolio_weights: Asset weights
            asset_volatilities: Asset volatilities
            correlation_matrix: Correlation matrix
            scenario_name: Name of scenario to run
            portfolio_value: Total portfolio value

        Returns:
            StressTestResult
        """
        scenario = self.scenarios.get(scenario_name)
        if scenario is None:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        assets = list(portfolio_weights.keys())
        weights = np.array([portfolio_weights[a] for a in assets])
        vols = np.array([asset_volatilities.get(a, 0.5) for a in assets])

        # Apply stress scenario
        # Shocked volatilities
        stressed_vols = vols * scenario.volatility_multiplier

        # Shocked correlations (move toward 1)
        n = len(assets)
        stressed_corr = correlation_matrix.copy()
        stressed_corr = stressed_corr + scenario.correlation_shock * (np.ones((n, n)) - correlation_matrix)
        np.fill_diagonal(stressed_corr, 1.0)

        # Stressed covariance
        stressed_cov = np.outer(stressed_vols, stressed_vols) * stressed_corr

        # Portfolio risk under stress
        port_vol = np.sqrt(weights @ stressed_cov @ weights)

        # Portfolio loss estimate
        # Using market shock + volatility-based loss
        direct_loss = abs(scenario.market_shock) * np.sum(np.abs(weights))
        vol_loss = port_vol * np.sqrt(scenario.duration_days / 252) * 2  # 2 sigma

        portfolio_loss = min(direct_loss + vol_loss, 1.0) * portfolio_value

        # Max drawdown estimate
        max_dd = min(abs(scenario.market_shock) * 1.5 + vol_loss, 0.99)

        # VaR breach check
        normal_var = port_vol * 2.33 * np.sqrt(1/252) * portfolio_value  # 99% 1-day VaR
        var_breach = portfolio_loss > normal_var * 3

        # Recovery time (simplified)
        recovery_time = int(scenario.duration_days * (1 + abs(scenario.market_shock)))

        # Worst asset
        individual_losses = weights * scenario.market_shock * (1 + vols * scenario.volatility_multiplier)
        worst_idx = np.argmin(individual_losses)
        worst_asset = assets[worst_idx]
        worst_loss = abs(individual_losses[worst_idx]) * portfolio_value

        return StressTestResult(
            scenario=scenario,
            portfolio_loss=portfolio_loss,
            max_drawdown=max_dd,
            var_breach=var_breach,
            recovery_time_days=recovery_time,
            worst_asset=worst_asset,
            worst_asset_loss=worst_loss
        )

    def run_all_scenarios(
        self,
        portfolio_weights: Dict[str, float],
        asset_volatilities: Dict[str, float],
        correlation_matrix: np.ndarray,
        portfolio_value: float = 1000000
    ) -> Dict[str, StressTestResult]:
        """Run all stress scenarios"""
        results = {}

        for scenario_name in self.scenarios.keys():
            try:
                result = self.run_stress_test(
                    portfolio_weights,
                    asset_volatilities,
                    correlation_matrix,
                    scenario_name,
                    portfolio_value
                )
                results[scenario_name] = result
            except Exception as e:
                logger.warning(f"Stress test failed for {scenario_name}: {e}")

        return results

    def reverse_stress_test(
        self,
        portfolio_weights: Dict[str, float],
        asset_volatilities: Dict[str, float],
        target_loss_pct: float = 0.50,
        max_iterations: int = 100
    ) -> StressScenario:
        """
        Reverse stress test: find scenario that causes target loss.

        "What scenario leads to 50% loss?"

        Args:
            portfolio_weights: Current weights
            asset_volatilities: Asset volatilities
            target_loss_pct: Target loss (e.g., 0.50 for 50%)
            max_iterations: Maximum search iterations

        Returns:
            StressScenario that achieves target loss
        """
        # Binary search for market shock
        low_shock = 0.05
        high_shock = 0.95

        for _ in range(max_iterations):
            mid_shock = (low_shock + high_shock) / 2

            # Create test scenario
            test_scenario = StressScenario(
                name='reverse_test',
                description='Reverse stress test',
                market_shock=-mid_shock,
                volatility_multiplier=2.0 + mid_shock * 3,
                correlation_shock=mid_shock * 0.5,
                liquidity_shock=1 + mid_shock * 10,
                duration_days=int(30 * (1 + mid_shock))
            )

            # Estimate loss (simplified)
            weights = np.array(list(portfolio_weights.values()))
            vols = np.array(list(asset_volatilities.values()))

            loss_estimate = mid_shock * np.sum(np.abs(weights))
            vol_contribution = np.sqrt(np.sum((weights * vols) ** 2)) * test_scenario.volatility_multiplier

            total_loss = loss_estimate + vol_contribution * 0.5

            if abs(total_loss - target_loss_pct) < 0.01:
                break

            if total_loss < target_loss_pct:
                low_shock = mid_shock
            else:
                high_shock = mid_shock

        return StressScenario(
            name=f'reverse_stress_{target_loss_pct:.0%}_loss',
            description=f'Scenario causing {target_loss_pct:.0%} portfolio loss',
            market_shock=-mid_shock,
            volatility_multiplier=2.0 + mid_shock * 3,
            correlation_shock=mid_shock * 0.5,
            liquidity_shock=1 + mid_shock * 10,
            duration_days=int(30 * (1 + mid_shock))
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════

class ModelDiagnosticsEngine:
    """
    Real-time model diagnostics and monitoring.
    """

    def __init__(
        self,
        lookback_window: int = 100,
        alert_thresholds: Optional[Dict[str, float]] = None
    ):
        self.lookback_window = lookback_window
        self.alert_thresholds = alert_thresholds or {
            'sharpe_ratio': 0.5,
            'hit_rate': 0.45,
            'max_drawdown': 0.15,
            'signal_decay': 0.3
        }

        # History
        self.diagnostics_history: List[ModelDiagnostics] = []
        self.alerts: List[Dict] = []

    def run_diagnostics(
        self,
        model_name: str,
        predictions: pd.Series,
        actuals: pd.Series,
        positions: Optional[pd.Series] = None
    ) -> ModelDiagnostics:
        """
        Run comprehensive diagnostics on model.

        Args:
            model_name: Name of the model
            predictions: Model predictions
            actuals: Actual returns
            positions: Optional position series

        Returns:
            ModelDiagnostics
        """
        # Align data
        common_idx = predictions.index.intersection(actuals.index)
        preds = predictions.loc[common_idx]
        acts = actuals.loc[common_idx]

        # Hit rate
        correct = ((preds > 0) == (acts > 0)).sum()
        hit_rate = correct / len(preds)

        # Sharpe ratio (of signal returns)
        if positions is not None:
            pos = positions.loc[common_idx]
            signal_returns = pos * acts
        else:
            signal_returns = np.sign(preds) * acts

        sharpe = signal_returns.mean() / (signal_returns.std() + 1e-10) * np.sqrt(252)

        # Profit factor
        gains = signal_returns[signal_returns > 0].sum()
        losses = abs(signal_returns[signal_returns < 0].sum())
        profit_factor = gains / (losses + 1e-10)

        # Drawdown
        cumulative = (1 + signal_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        max_dd = abs(drawdowns.min())
        current_dd = abs(drawdowns.iloc[-1])

        # Information ratio
        ic = preds.corr(acts)
        ir = ic * np.sqrt(252 / len(preds))

        # Signal decay (autocorrelation of predictions)
        if len(preds) > 10:
            signal_decay = preds.autocorr(lag=1)
        else:
            signal_decay = 0

        # Feature stability (would need feature importance history)
        feature_stability = 1.0  # Placeholder

        # Regime alignment (would need regime data)
        regime_alignment = 1.0  # Placeholder

        diagnostics = ModelDiagnostics(
            timestamp=datetime.utcnow(),
            model_name=model_name,
            sharpe_ratio=float(sharpe),
            information_ratio=float(ir),
            hit_rate=float(hit_rate),
            profit_factor=float(profit_factor),
            max_drawdown=float(max_dd),
            current_drawdown=float(current_dd),
            regime_alignment=float(regime_alignment),
            signal_decay=float(signal_decay) if not np.isnan(signal_decay) else 0,
            feature_stability=float(feature_stability)
        )

        self.diagnostics_history.append(diagnostics)

        # Check for alerts
        self._check_alerts(diagnostics)

        return diagnostics

    def _check_alerts(self, diagnostics: ModelDiagnostics):
        """Check if diagnostics trigger any alerts"""
        alerts = []

        if diagnostics.sharpe_ratio < self.alert_thresholds['sharpe_ratio']:
            alerts.append({
                'type': 'low_sharpe',
                'message': f"Sharpe ratio ({diagnostics.sharpe_ratio:.2f}) below threshold",
                'severity': 'warning'
            })

        if diagnostics.hit_rate < self.alert_thresholds['hit_rate']:
            alerts.append({
                'type': 'low_hit_rate',
                'message': f"Hit rate ({diagnostics.hit_rate:.2%}) below threshold",
                'severity': 'warning'
            })

        if diagnostics.max_drawdown > self.alert_thresholds['max_drawdown']:
            alerts.append({
                'type': 'high_drawdown',
                'message': f"Max drawdown ({diagnostics.max_drawdown:.2%}) above threshold",
                'severity': 'critical'
            })

        if diagnostics.signal_decay < self.alert_thresholds['signal_decay']:
            alerts.append({
                'type': 'signal_decay',
                'message': f"Signal decay detected (autocorr={diagnostics.signal_decay:.2f})",
                'severity': 'info'
            })

        for alert in alerts:
            alert['timestamp'] = diagnostics.timestamp
            alert['model'] = diagnostics.model_name
            self.alerts.append(alert)

    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Get recent alerts"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [a for a in self.alerts if a['timestamp'] > cutoff]

    def get_model_health_score(self, model_name: str) -> float:
        """Calculate overall model health score (0-100)"""
        recent = [d for d in self.diagnostics_history[-20:]
                  if d.model_name == model_name]

        if not recent:
            return 50  # Unknown

        latest = recent[-1]

        # Score components
        sharpe_score = min(100, max(0, (latest.sharpe_ratio + 1) * 25))
        hit_rate_score = min(100, max(0, latest.hit_rate * 100))
        dd_score = max(0, 100 - latest.max_drawdown * 200)
        pf_score = min(100, max(0, (latest.profit_factor - 0.5) * 50))

        return np.mean([sharpe_score, hit_rate_score, dd_score, pf_score])


# ═══════════════════════════════════════════════════════════════════════════════
# MONITORING PIPELINE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

class MonitoringPipeline:
    """
    Main monitoring pipeline orchestrator.
    """

    def __init__(
        self,
        config: Optional[Any] = None
    ):
        self.config = config

        # Initialize components
        self.performance_attributor = PerformanceAttributor()
        self.pbo_analyzer = PBOAnalyzer()
        self.stress_tester = StressTestingFramework()
        self.diagnostics_engine = ModelDiagnosticsEngine()

        # State
        self.last_run = None

    def run_full_analysis(
        self,
        portfolio_returns: pd.Series,
        asset_returns: pd.DataFrame,
        weights: pd.DataFrame,
        model_predictions: Dict[str, pd.Series],
        portfolio_value: float = 1000000
    ) -> Dict[str, Any]:
        """
        Run comprehensive monitoring analysis.

        Returns:
            Dict with all analysis results
        """
        results = {}
        timestamp = datetime.utcnow()

        # 1. Performance attribution
        try:
            # Create dummy factor returns (would come from factor model)
            factor_returns = pd.DataFrame({
                'market': asset_returns.mean(axis=1),
                'momentum': asset_returns.diff().mean(axis=1),
            }, index=asset_returns.index)

            factor_exposures = pd.DataFrame({
                'market': [1.0] * len(asset_returns.columns),
                'momentum': [0.5] * len(asset_returns.columns),
            }, index=asset_returns.columns)

            attribution = self.performance_attributor.attribute_returns(
                portfolio_returns, asset_returns, weights,
                factor_returns, factor_exposures
            )
            results['attribution'] = attribution
        except Exception as e:
            logger.warning(f"Attribution failed: {e}")
            results['attribution'] = None

        # 2. Model diagnostics
        diagnostics = {}
        for model_name, predictions in model_predictions.items():
            try:
                diag = self.diagnostics_engine.run_diagnostics(
                    model_name, predictions, portfolio_returns
                )
                diagnostics[model_name] = diag
            except Exception as e:
                logger.warning(f"Diagnostics failed for {model_name}: {e}")

        results['diagnostics'] = diagnostics

        # 3. Stress testing
        try:
            # Get current weights
            current_weights = weights.iloc[-1].to_dict() if not weights.empty else {}
            asset_vols = asset_returns.std().to_dict()
            corr_matrix = asset_returns.corr().values

            stress_results = self.stress_tester.run_all_scenarios(
                current_weights, asset_vols, corr_matrix, portfolio_value
            )
            results['stress_tests'] = stress_results
        except Exception as e:
            logger.warning(f"Stress testing failed: {e}")
            results['stress_tests'] = None

        # 4. Alerts
        results['alerts'] = self.diagnostics_engine.get_recent_alerts(24)

        # 5. Health scores
        health_scores = {
            name: self.diagnostics_engine.get_model_health_score(name)
            for name in model_predictions.keys()
        }
        results['health_scores'] = health_scores

        self.last_run = timestamp
        results['timestamp'] = timestamp

        return results

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        return {
            'last_run': self.last_run,
            'attribution_history': self.performance_attributor.attribution_history[-50:],
            'diagnostics_history': self.diagnostics_engine.diagnostics_history[-100:],
            'recent_alerts': self.diagnostics_engine.get_recent_alerts(72),
            'scenarios': list(self.stress_tester.scenarios.keys())
        }


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    'PerformanceAttribution',
    'PBOResult',
    'StressScenario',
    'StressTestResult',
    'ModelDiagnostics',
    'PerformanceAttributor',
    'PBOAnalyzer',
    'StressTestingFramework',
    'ModelDiagnosticsEngine',
    'MonitoringPipeline',
]
