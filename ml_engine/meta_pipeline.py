"""
META-PIPELINE: System-Steuerung & Adaptivität

Dynamische Anpassung aller Pipelines an Marktbedingungen basierend auf:
- Hidden Markov Modelle (HMM) mit 5-7 Zuständen für Regime-Erkennung
- Bayesian Optimization mit Multi-Armed Bandits für Hyperparameter
- Kapitalallokation zwischen Strategien (Kelly-Kriterium + Risk-Parity)

Theoretische Grundlage: Adaptive Markthypothese (AMH)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
from collections import deque
import warnings

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS & DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class HMMRegimeState(Enum):
    """Hidden Markov Model Regime States (7-state model)"""
    CRISIS = 0              # Extreme risk-off, high volatility, correlations → 1
    HIGH_VOLATILITY = 1     # Elevated volatility, uncertain direction
    RISK_OFF = 2            # Moderate bearish, defensive positioning
    NEUTRAL = 3             # Range-bound, mean-reverting
    RISK_ON = 4             # Moderate bullish, trend-following
    LOW_VOLATILITY = 5      # Compressed volatility, potential breakout
    EUPHORIA = 6            # Extreme risk-on, momentum-driven


@dataclass
class RegimeTransitionProbability:
    """Transition probability matrix for regime changes"""
    from_state: HMMRegimeState
    to_state: HMMRegimeState
    probability: float
    historical_frequency: float
    avg_duration_hours: float


@dataclass
class StrategyPerformance:
    """Performance metrics for a single strategy"""
    strategy_name: str
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    current_drawdown: float
    avg_return: float
    volatility: float
    information_ratio: float
    kelly_fraction: float
    regime_performance: Dict[HMMRegimeState, float] = field(default_factory=dict)


@dataclass
class CapitalAllocation:
    """Capital allocation recommendation"""
    strategy_name: str
    kelly_weight: float           # Kelly criterion weight
    risk_parity_weight: float     # Risk parity weight
    final_weight: float           # Combined weight
    max_leverage: float           # Maximum allowed leverage
    current_regime: HMMRegimeState
    regime_adjustment: float      # Regime-based adjustment factor


@dataclass
class HyperparameterConfig:
    """Hyperparameter configuration with bounds"""
    name: str
    current_value: float
    min_value: float
    max_value: float
    step_size: float
    last_updated: datetime
    performance_history: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class MetaPipelineState:
    """Complete state of the meta-pipeline"""
    current_regime: HMMRegimeState
    regime_confidence: float
    regime_duration: int          # Hours in current regime
    transition_probabilities: Dict[HMMRegimeState, float]
    strategy_allocations: Dict[str, CapitalAllocation]
    hyperparameter_configs: Dict[str, HyperparameterConfig]
    global_risk_multiplier: float
    timestamp: datetime


# ═══════════════════════════════════════════════════════════════════════════════
# HIDDEN MARKOV MODEL REGIME DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class HMMRegimeDetector:
    """
    Hidden Markov Model für Regime-Erkennung mit 7 Zuständen.

    Implementiert Baum-Welch Algorithmus für Parameter-Schätzung und
    Viterbi Algorithmus für optimale Zustandssequenz.

    P(Regime_t | Regime_{t-1}, Market_Features)
    """

    def __init__(
        self,
        n_states: int = 7,
        n_features: int = 10,
        lookback_periods: int = 168,  # 1 week hourly
        min_regime_duration: int = 6   # Minimum hours in regime
    ):
        self.n_states = n_states
        self.n_features = n_features
        self.lookback_periods = lookback_periods
        self.min_regime_duration = min_regime_duration

        # Initialize HMM parameters
        self._init_hmm_parameters()

        # Regime history
        self.regime_history: deque = deque(maxlen=1000)
        self.current_regime = HMMRegimeState.NEUTRAL
        self.regime_start_time = datetime.utcnow()

    def _init_hmm_parameters(self):
        """Initialize HMM parameters with reasonable priors"""
        # Transition matrix (A) - prior based on market behavior
        # Rows = from state, Cols = to state
        self.transition_matrix = np.array([
            # CRISIS  HI_VOL  RISK_OFF NEUTRAL RISK_ON LO_VOL  EUPHORIA
            [0.60,    0.25,   0.10,    0.04,   0.01,   0.00,   0.00],  # CRISIS
            [0.15,    0.50,   0.20,    0.10,   0.04,   0.01,   0.00],  # HIGH_VOL
            [0.05,    0.15,   0.50,    0.20,   0.08,   0.02,   0.00],  # RISK_OFF
            [0.02,    0.08,   0.15,    0.50,   0.15,   0.08,   0.02],  # NEUTRAL
            [0.00,    0.02,   0.08,    0.20,   0.50,   0.15,   0.05],  # RISK_ON
            [0.00,    0.01,   0.04,    0.10,   0.20,   0.50,   0.15],  # LOW_VOL
            [0.00,    0.00,   0.01,    0.04,   0.10,   0.25,   0.60],  # EUPHORIA
        ])

        # Emission parameters (Gaussian for each feature per state)
        # Mean and variance for each feature in each state
        self.emission_means = np.zeros((self.n_states, self.n_features))
        self.emission_vars = np.ones((self.n_states, self.n_features))

        # Initial state probabilities
        self.initial_probs = np.array([0.05, 0.10, 0.15, 0.40, 0.15, 0.10, 0.05])

        # Feature extraction weights
        self.feature_weights = {
            'volatility': 0.20,
            'trend': 0.15,
            'momentum': 0.15,
            'volume': 0.10,
            'correlation': 0.10,
            'spread': 0.10,
            'orderbook_imbalance': 0.10,
            'options_iv': 0.05,
            'funding_rate': 0.05
        }

    def extract_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """
        Extract features for HMM from market data.

        Returns normalized feature vector for regime classification.
        """
        features = []

        if len(market_data) < 20:
            return np.zeros(self.n_features)

        close = market_data['close']
        returns = close.pct_change().dropna()

        # 1. Volatility features
        realized_vol = returns.rolling(20).std() * np.sqrt(252 * 24)
        vol_percentile = self._rolling_percentile(realized_vol, 100)
        features.append(vol_percentile.iloc[-1] if len(vol_percentile) > 0 else 0.5)

        # 2. Volatility of volatility
        vol_of_vol = realized_vol.pct_change().rolling(10).std()
        features.append(self._normalize(vol_of_vol.iloc[-1] if len(vol_of_vol) > 0 else 0))

        # 3. Trend strength (ADX-like)
        trend_strength = self._calculate_trend_strength(market_data)
        features.append(trend_strength)

        # 4. Trend direction
        trend_direction = self._calculate_trend_direction(close)
        features.append(trend_direction)

        # 5. Momentum
        momentum = self._calculate_momentum(returns)
        features.append(momentum)

        # 6. Volume trend
        if 'volume' in market_data.columns:
            vol_trend = self._calculate_volume_trend(market_data['volume'])
            features.append(vol_trend)
        else:
            features.append(0.5)

        # 7. Mean reversion indicator
        mean_reversion = self._calculate_mean_reversion(close)
        features.append(mean_reversion)

        # 8. Tail risk indicator (kurtosis)
        kurtosis = returns.rolling(50).apply(lambda x: self._safe_kurtosis(x))
        features.append(self._normalize(kurtosis.iloc[-1] if len(kurtosis) > 0 else 0))

        # 9. Skewness
        skewness = returns.rolling(50).apply(lambda x: self._safe_skewness(x))
        features.append(self._normalize(skewness.iloc[-1] if len(skewness) > 0 else 0))

        # 10. Autocorrelation (regime persistence)
        autocorr = returns.rolling(30).apply(lambda x: x.autocorr(lag=1) if len(x) > 1 else 0)
        features.append(self._normalize(autocorr.iloc[-1] if len(autocorr) > 0 else 0))

        return np.array(features[:self.n_features])

    def _rolling_percentile(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling percentile"""
        def percentile_rank(x):
            if len(x) < 2:
                return 0.5
            return (x.iloc[-1] > x.iloc[:-1]).mean()
        return series.rolling(window, min_periods=10).apply(percentile_rank)

    def _normalize(self, value: float, min_val: float = -3, max_val: float = 3) -> float:
        """Normalize value to [0, 1] range"""
        if np.isnan(value) or np.isinf(value):
            return 0.5
        clipped = np.clip(value, min_val, max_val)
        return (clipped - min_val) / (max_val - min_val)

    def _safe_kurtosis(self, x: pd.Series) -> float:
        """Safe kurtosis calculation"""
        try:
            from scipy.stats import kurtosis
            return kurtosis(x, nan_policy='omit')
        except:
            return 0

    def _safe_skewness(self, x: pd.Series) -> float:
        """Safe skewness calculation"""
        try:
            from scipy.stats import skew
            return skew(x, nan_policy='omit')
        except:
            return 0

    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength (0-1)"""
        if len(data) < 30:
            return 0.5

        close = data['close']
        high = data.get('high', close)
        low = data.get('low', close)

        # Simplified ADX calculation
        tr = np.maximum(high - low,
                       np.abs(high - close.shift(1)),
                       np.abs(low - close.shift(1)))
        atr = tr.rolling(14).mean()

        plus_dm = (high - high.shift(1)).clip(lower=0)
        minus_dm = (low.shift(1) - low).clip(lower=0)

        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)

        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(14).mean()

        # Normalize to 0-1
        return np.clip(adx.iloc[-1] / 100, 0, 1) if not np.isnan(adx.iloc[-1]) else 0.5

    def _calculate_trend_direction(self, close: pd.Series) -> float:
        """Calculate trend direction (-1 to 1)"""
        if len(close) < 50:
            return 0

        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1]
        current = close.iloc[-1]

        # Score based on price position relative to MAs
        score = 0
        if current > sma_20:
            score += 0.3
        else:
            score -= 0.3

        if current > sma_50:
            score += 0.3
        else:
            score -= 0.3

        if sma_20 > sma_50:
            score += 0.4
        else:
            score -= 0.4

        return (score + 1) / 2  # Normalize to 0-1

    def _calculate_momentum(self, returns: pd.Series) -> float:
        """Calculate momentum indicator"""
        if len(returns) < 20:
            return 0.5

        # Multiple timeframe momentum
        mom_5 = returns.iloc[-5:].sum()
        mom_10 = returns.iloc[-10:].sum()
        mom_20 = returns.iloc[-20:].sum()

        # Weighted average
        combined = 0.5 * mom_5 + 0.3 * mom_10 + 0.2 * mom_20

        # Normalize
        return self._normalize(combined * 100, -20, 20)

    def _calculate_volume_trend(self, volume: pd.Series) -> float:
        """Calculate volume trend"""
        if len(volume) < 20:
            return 0.5

        vol_sma = volume.rolling(20).mean()
        current_ratio = volume.iloc[-1] / vol_sma.iloc[-1] if vol_sma.iloc[-1] > 0 else 1

        return self._normalize(current_ratio, 0.5, 2.0)

    def _calculate_mean_reversion(self, close: pd.Series) -> float:
        """Calculate mean reversion potential"""
        if len(close) < 50:
            return 0.5

        # Z-score from 50-period mean
        mean = close.rolling(50).mean().iloc[-1]
        std = close.rolling(50).std().iloc[-1]

        if std > 0:
            z_score = (close.iloc[-1] - mean) / std
            # High absolute z-score = high mean reversion potential
            return self._normalize(np.abs(z_score), 0, 3)

        return 0.5

    def forward_algorithm(self, observations: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Forward algorithm für HMM.

        Berechnet P(O|λ) und forward probabilities α.
        """
        T = len(observations)
        alpha = np.zeros((T, self.n_states))

        # Initialize
        alpha[0] = self.initial_probs * self._emission_prob(observations[0])
        alpha[0] /= alpha[0].sum() + 1e-10

        # Forward pass
        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = np.sum(alpha[t-1] * self.transition_matrix[:, j]) * \
                             self._emission_prob_single(observations[t], j)
            alpha[t] /= alpha[t].sum() + 1e-10

        # Log likelihood
        log_likelihood = np.log(alpha[-1].sum() + 1e-10)

        return alpha, log_likelihood

    def backward_algorithm(self, observations: np.ndarray) -> np.ndarray:
        """
        Backward algorithm für HMM.
        """
        T = len(observations)
        beta = np.zeros((T, self.n_states))

        # Initialize
        beta[-1] = 1

        # Backward pass
        for t in range(T - 2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(
                    self.transition_matrix[i] *
                    self._emission_prob(observations[t + 1]) *
                    beta[t + 1]
                )
            beta[t] /= beta[t].sum() + 1e-10

        return beta

    def viterbi_algorithm(self, observations: np.ndarray) -> List[int]:
        """
        Viterbi algorithm für optimale Zustandssequenz.
        """
        T = len(observations)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)

        # Initialize
        delta[0] = np.log(self.initial_probs + 1e-10) + \
                   np.log(self._emission_prob(observations[0]) + 1e-10)

        # Forward pass
        for t in range(1, T):
            for j in range(self.n_states):
                temp = delta[t-1] + np.log(self.transition_matrix[:, j] + 1e-10)
                psi[t, j] = np.argmax(temp)
                delta[t, j] = np.max(temp) + \
                             np.log(self._emission_prob_single(observations[t], j) + 1e-10)

        # Backtrack
        path = np.zeros(T, dtype=int)
        path[-1] = np.argmax(delta[-1])

        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]

        return path.tolist()

    def _emission_prob(self, observation: np.ndarray) -> np.ndarray:
        """Calculate emission probabilities for all states"""
        probs = np.zeros(self.n_states)
        for state in range(self.n_states):
            probs[state] = self._emission_prob_single(observation, state)
        return probs

    def _emission_prob_single(self, observation: np.ndarray, state: int) -> float:
        """Calculate emission probability for single state (Gaussian)"""
        mean = self.emission_means[state]
        var = self.emission_vars[state]

        # Multivariate Gaussian (diagonal covariance)
        diff = observation - mean
        exponent = -0.5 * np.sum(diff ** 2 / (var + 1e-10))
        norm_const = np.prod(np.sqrt(2 * np.pi * (var + 1e-10)))

        return np.exp(exponent) / (norm_const + 1e-10)

    def baum_welch_update(
        self,
        observations: np.ndarray,
        n_iterations: int = 10,
        convergence_threshold: float = 1e-4
    ):
        """
        Baum-Welch Algorithmus für HMM Parameter-Schätzung (EM).
        """
        T = len(observations)
        prev_log_likelihood = float('-inf')

        for iteration in range(n_iterations):
            # E-step: Forward-Backward
            alpha, log_likelihood = self.forward_algorithm(observations)
            beta = self.backward_algorithm(observations)

            # Check convergence
            if abs(log_likelihood - prev_log_likelihood) < convergence_threshold:
                logger.info(f"Baum-Welch converged after {iteration + 1} iterations")
                break
            prev_log_likelihood = log_likelihood

            # Compute gamma and xi
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True) + 1e-10

            xi = np.zeros((T - 1, self.n_states, self.n_states))
            for t in range(T - 1):
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        xi[t, i, j] = alpha[t, i] * self.transition_matrix[i, j] * \
                                     self._emission_prob_single(observations[t + 1], j) * \
                                     beta[t + 1, j]
                xi[t] /= xi[t].sum() + 1e-10

            # M-step: Update parameters
            # Update initial probabilities
            self.initial_probs = gamma[0]

            # Update transition matrix
            for i in range(self.n_states):
                for j in range(self.n_states):
                    self.transition_matrix[i, j] = xi[:, i, j].sum() / (gamma[:-1, i].sum() + 1e-10)

            # Normalize rows
            self.transition_matrix /= self.transition_matrix.sum(axis=1, keepdims=True) + 1e-10

            # Update emission parameters
            for j in range(self.n_states):
                gamma_sum = gamma[:, j].sum() + 1e-10
                self.emission_means[j] = (gamma[:, j].reshape(-1, 1) * observations).sum(axis=0) / gamma_sum
                diff = observations - self.emission_means[j]
                self.emission_vars[j] = (gamma[:, j].reshape(-1, 1) * diff ** 2).sum(axis=0) / gamma_sum
                self.emission_vars[j] = np.maximum(self.emission_vars[j], 1e-4)  # Minimum variance

    def detect_regime(self, market_data: pd.DataFrame) -> Tuple[HMMRegimeState, float, Dict[HMMRegimeState, float]]:
        """
        Detect current market regime using HMM.

        Returns:
            - Current regime state
            - Confidence (posterior probability)
            - Transition probabilities to all states
        """
        # Extract features
        features = self.extract_features(market_data)

        # Get state probabilities using forward algorithm on recent data
        if len(market_data) >= self.lookback_periods:
            # Extract feature sequence
            feature_sequence = []
            for i in range(max(0, len(market_data) - self.lookback_periods), len(market_data)):
                if i >= 20:  # Minimum data for feature extraction
                    feat = self.extract_features(market_data.iloc[:i+1])
                    feature_sequence.append(feat)

            if len(feature_sequence) > 0:
                observations = np.array(feature_sequence)
                alpha, _ = self.forward_algorithm(observations)
                state_probs = alpha[-1] / (alpha[-1].sum() + 1e-10)
            else:
                state_probs = self._get_prior_probs(features)
        else:
            state_probs = self._get_prior_probs(features)

        # Get most likely state
        current_state_idx = np.argmax(state_probs)
        confidence = state_probs[current_state_idx]

        # Apply minimum duration filter
        new_regime = HMMRegimeState(current_state_idx)
        hours_in_regime = (datetime.utcnow() - self.regime_start_time).total_seconds() / 3600

        if new_regime != self.current_regime:
            if hours_in_regime >= self.min_regime_duration:
                self.current_regime = new_regime
                self.regime_start_time = datetime.utcnow()
                self.regime_history.append((datetime.utcnow(), new_regime, confidence))
            else:
                # Stay in current regime
                new_regime = self.current_regime

        # Calculate transition probabilities
        transition_probs = {
            HMMRegimeState(i): self.transition_matrix[current_state_idx, i]
            for i in range(self.n_states)
        }

        return new_regime, confidence, transition_probs

    def _get_prior_probs(self, features: np.ndarray) -> np.ndarray:
        """Get state probabilities from features using emission model"""
        emission_probs = self._emission_prob(features)
        probs = self.initial_probs * emission_probs
        return probs / (probs.sum() + 1e-10)


# ═══════════════════════════════════════════════════════════════════════════════
# BAYESIAN HYPERPARAMETER OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════

class BayesianHyperparameterOptimizer:
    """
    Bayesian Optimization mit Multi-Armed Bandits für Hyperparameter.

    θ* = argmax_θ E[f(θ)|D] - λ * Var[f(θ)|D]

    Verwendet Gaussian Process für Surrogatmodell und UCB/Thompson Sampling.
    """

    def __init__(
        self,
        exploration_weight: float = 2.0,
        decay_factor: float = 0.95,
        min_samples: int = 5
    ):
        self.exploration_weight = exploration_weight
        self.decay_factor = decay_factor
        self.min_samples = min_samples

        # Store observations
        self.observations: Dict[str, List[Tuple[np.ndarray, float]]] = {}

        # Multi-armed bandit state
        self.arm_rewards: Dict[str, deque] = {}
        self.arm_pulls: Dict[str, int] = {}

    def suggest_hyperparameters(
        self,
        param_configs: Dict[str, HyperparameterConfig],
        objective_name: str = "sharpe_ratio"
    ) -> Dict[str, float]:
        """
        Suggest next hyperparameter configuration using Bayesian optimization.
        """
        suggestions = {}

        for param_name, config in param_configs.items():
            if param_name not in self.observations:
                self.observations[param_name] = []
                self.arm_rewards[param_name] = deque(maxlen=100)
                self.arm_pulls[param_name] = 0

            obs = self.observations[param_name]

            if len(obs) < self.min_samples:
                # Exploration phase: random sampling
                suggestions[param_name] = np.random.uniform(
                    config.min_value,
                    config.max_value
                )
            else:
                # Exploitation phase: GP-UCB
                suggestions[param_name] = self._gp_ucb_suggest(
                    config, obs
                )

        return suggestions

    def _gp_ucb_suggest(
        self,
        config: HyperparameterConfig,
        observations: List[Tuple[np.ndarray, float]]
    ) -> float:
        """
        Gaussian Process Upper Confidence Bound suggestion.
        """
        # Extract X and y from observations
        X = np.array([obs[0] for obs in observations])
        y = np.array([obs[1] for obs in observations])

        # Create candidate points
        n_candidates = 100
        candidates = np.linspace(config.min_value, config.max_value, n_candidates)

        # Compute GP predictions (simplified RBF kernel)
        ucb_values = []

        for candidate in candidates:
            mean, std = self._gp_predict(X, y, np.array([candidate]))
            ucb = mean + self.exploration_weight * std
            ucb_values.append(ucb)

        # Return candidate with highest UCB
        best_idx = np.argmax(ucb_values)
        return candidates[best_idx]

    def _gp_predict(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        length_scale: float = 1.0,
        noise: float = 0.1
    ) -> Tuple[float, float]:
        """
        Simplified Gaussian Process prediction with RBF kernel.

        k(x,x') = σ² * exp(-||x-x'||²/(2l²))
        """
        # RBF Kernel
        def rbf_kernel(x1, x2, l=length_scale):
            return np.exp(-np.sum((x1 - x2) ** 2) / (2 * l ** 2))

        n = len(X_train)

        # Compute kernel matrices
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = rbf_kernel(X_train[i], X_train[j])

        K += noise * np.eye(n)

        # Compute k_star
        k_star = np.array([rbf_kernel(X_test, X_train[i]) for i in range(n)])

        # GP predictions
        try:
            K_inv = np.linalg.inv(K)
            mean = k_star @ K_inv @ y_train
            var = rbf_kernel(X_test, X_test) - k_star @ K_inv @ k_star
            std = np.sqrt(max(0, var))
        except:
            mean = np.mean(y_train)
            std = np.std(y_train)

        return float(mean), float(std)

    def update_observations(
        self,
        param_name: str,
        param_value: float,
        performance: float
    ):
        """Update observations with new result"""
        if param_name not in self.observations:
            self.observations[param_name] = []
            self.arm_rewards[param_name] = deque(maxlen=100)
            self.arm_pulls[param_name] = 0

        self.observations[param_name].append((np.array([param_value]), performance))
        self.arm_rewards[param_name].append(performance)
        self.arm_pulls[param_name] += 1

    def thompson_sampling_select(
        self,
        strategies: List[str],
        regime: HMMRegimeState
    ) -> str:
        """
        Thompson Sampling für Strategy Selection.

        Wählt Strategie basierend auf posterior sampling.
        """
        samples = {}

        for strategy in strategies:
            key = f"{strategy}_{regime.name}"

            if key not in self.arm_rewards or len(self.arm_rewards[key]) < 2:
                # Prior: Beta(1, 1) for new strategies
                samples[strategy] = np.random.beta(1, 1)
            else:
                # Posterior based on observed rewards
                rewards = np.array(list(self.arm_rewards[key]))

                # Normalize rewards to [0, 1]
                min_r, max_r = rewards.min(), rewards.max()
                if max_r > min_r:
                    norm_rewards = (rewards - min_r) / (max_r - min_r)
                else:
                    norm_rewards = np.ones_like(rewards) * 0.5

                # Estimate Beta parameters
                alpha = norm_rewards.sum() + 1
                beta_param = len(norm_rewards) - norm_rewards.sum() + 1

                samples[strategy] = np.random.beta(alpha, beta_param)

        # Select strategy with highest sample
        return max(samples, key=samples.get)


# ═══════════════════════════════════════════════════════════════════════════════
# CAPITAL ALLOCATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class CapitalAllocationEngine:
    """
    Kapitalallokation zwischen Strategien basierend auf:
    - Kelly-Kriterium für einzelne Strategien
    - Risk-Parity für Portfolio der Strategien

    Kombiniert beide Ansätze mit regime-abhängiger Gewichtung.
    """

    def __init__(
        self,
        max_leverage: float = 3.0,
        kelly_fraction: float = 0.5,  # Half-Kelly for safety
        risk_parity_weight: float = 0.4,
        kelly_weight: float = 0.6,
        min_allocation: float = 0.05,
        max_allocation: float = 0.40
    ):
        self.max_leverage = max_leverage
        self.kelly_fraction = kelly_fraction
        self.risk_parity_weight = risk_parity_weight
        self.kelly_weight = kelly_weight
        self.min_allocation = min_allocation
        self.max_allocation = max_allocation

        # Performance tracking
        self.strategy_returns: Dict[str, deque] = {}

    def calculate_kelly_weights(
        self,
        strategy_performances: Dict[str, StrategyPerformance]
    ) -> Dict[str, float]:
        """
        Calculate Kelly criterion weights for each strategy.

        Kelly fraction f* = (μ - r) / σ² for continuous returns
        or f* = p - q/b for discrete outcomes
        """
        kelly_weights = {}

        for name, perf in strategy_performances.items():
            if perf.volatility <= 0:
                kelly_weights[name] = 0
                continue

            # Continuous Kelly
            excess_return = perf.avg_return - 0.05 / 252  # Risk-free rate
            kelly = excess_return / (perf.volatility ** 2 + 1e-10)

            # Alternative: Discrete Kelly using win rate
            if perf.profit_factor > 0:
                p = perf.win_rate
                b = perf.profit_factor - 1  # Odds
                q = 1 - p
                discrete_kelly = (p * b - q) / (b + 1e-10) if b > 0 else 0
            else:
                discrete_kelly = 0

            # Average of both methods
            kelly_raw = 0.7 * kelly + 0.3 * discrete_kelly

            # Apply fraction and bounds
            kelly_weights[name] = np.clip(
                kelly_raw * self.kelly_fraction,
                -self.max_allocation,
                self.max_allocation
            )

        return kelly_weights

    def calculate_risk_parity_weights(
        self,
        strategy_performances: Dict[str, StrategyPerformance],
        correlation_matrix: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate Risk Parity weights.

        Each strategy contributes equally to total portfolio risk.
        """
        strategies = list(strategy_performances.keys())
        n = len(strategies)

        if n == 0:
            return {}

        # Get volatilities
        vols = np.array([
            strategy_performances[s].volatility
            for s in strategies
        ])

        # Handle zero volatilities
        vols = np.maximum(vols, 1e-6)

        # Simple inverse volatility weighting (approximation of risk parity)
        inv_vol = 1 / vols
        weights = inv_vol / inv_vol.sum()

        # If correlation matrix provided, use it for more accurate risk parity
        if correlation_matrix is not None and correlation_matrix.shape[0] == n:
            # Covariance matrix
            cov = np.outer(vols, vols) * correlation_matrix

            # Iterative risk parity (simplified Newton-Raphson)
            weights = self._iterative_risk_parity(cov, weights)

        return {strategies[i]: weights[i] for i in range(n)}

    def _iterative_risk_parity(
        self,
        cov_matrix: np.ndarray,
        initial_weights: np.ndarray,
        n_iterations: int = 50,
        tol: float = 1e-6
    ) -> np.ndarray:
        """
        Iterative algorithm for risk parity weights.
        """
        n = len(initial_weights)
        weights = initial_weights.copy()

        for _ in range(n_iterations):
            # Portfolio variance
            port_var = weights @ cov_matrix @ weights
            port_vol = np.sqrt(port_var + 1e-10)

            # Marginal risk contribution
            mrc = (cov_matrix @ weights) / (port_vol + 1e-10)

            # Risk contribution
            rc = weights * mrc

            # Target: equal risk contribution
            target_rc = port_vol / n

            # Update weights
            new_weights = weights * (target_rc / (rc + 1e-10))
            new_weights = new_weights / new_weights.sum()

            # Check convergence
            if np.max(np.abs(new_weights - weights)) < tol:
                break

            weights = new_weights

        return weights

    def calculate_allocations(
        self,
        strategy_performances: Dict[str, StrategyPerformance],
        current_regime: HMMRegimeState,
        regime_confidence: float
    ) -> Dict[str, CapitalAllocation]:
        """
        Calculate final capital allocations combining Kelly and Risk Parity.
        """
        if not strategy_performances:
            return {}

        # Calculate individual weights
        kelly_weights = self.calculate_kelly_weights(strategy_performances)
        risk_parity_weights = self.calculate_risk_parity_weights(strategy_performances)

        # Regime adjustments
        regime_multipliers = self._get_regime_multipliers(current_regime)

        allocations = {}

        for name, perf in strategy_performances.items():
            kelly_w = kelly_weights.get(name, 0)
            rp_w = risk_parity_weights.get(name, 0)

            # Combine weights
            combined_weight = (
                self.kelly_weight * kelly_w +
                self.risk_parity_weight * rp_w
            )

            # Apply regime adjustment
            regime_perf = perf.regime_performance.get(current_regime, 1.0)
            regime_adj = regime_multipliers.get(name, 1.0) * regime_perf

            # Scale by confidence
            final_weight = combined_weight * regime_adj * regime_confidence

            # Apply bounds
            final_weight = np.clip(
                final_weight,
                -self.max_allocation,
                self.max_allocation
            )

            allocations[name] = CapitalAllocation(
                strategy_name=name,
                kelly_weight=kelly_w,
                risk_parity_weight=rp_w,
                final_weight=final_weight,
                max_leverage=self._get_strategy_leverage(name, current_regime),
                current_regime=current_regime,
                regime_adjustment=regime_adj
            )

        # Normalize if total exceeds leverage limit
        total_weight = sum(abs(a.final_weight) for a in allocations.values())
        if total_weight > self.max_leverage:
            scale = self.max_leverage / total_weight
            for alloc in allocations.values():
                alloc.final_weight *= scale

        return allocations

    def _get_regime_multipliers(
        self,
        regime: HMMRegimeState
    ) -> Dict[str, float]:
        """Get strategy multipliers based on regime"""
        # Default multipliers by regime
        base_multipliers = {
            HMMRegimeState.CRISIS: 0.3,
            HMMRegimeState.HIGH_VOLATILITY: 0.5,
            HMMRegimeState.RISK_OFF: 0.7,
            HMMRegimeState.NEUTRAL: 1.0,
            HMMRegimeState.RISK_ON: 1.2,
            HMMRegimeState.LOW_VOLATILITY: 0.8,  # Caution before potential breakout
            HMMRegimeState.EUPHORIA: 0.6,  # Reduce risk in euphoria
        }

        return {'default': base_multipliers.get(regime, 1.0)}

    def _get_strategy_leverage(
        self,
        strategy_name: str,
        regime: HMMRegimeState
    ) -> float:
        """Get maximum leverage for strategy in current regime"""
        base_leverage = self.max_leverage

        # Reduce leverage in high-risk regimes
        leverage_adjustments = {
            HMMRegimeState.CRISIS: 0.3,
            HMMRegimeState.HIGH_VOLATILITY: 0.5,
            HMMRegimeState.RISK_OFF: 0.7,
            HMMRegimeState.NEUTRAL: 1.0,
            HMMRegimeState.RISK_ON: 1.0,
            HMMRegimeState.LOW_VOLATILITY: 0.9,
            HMMRegimeState.EUPHORIA: 0.5,
        }

        return base_leverage * leverage_adjustments.get(regime, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# META-PIPELINE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

class MetaPipeline:
    """
    Meta-Pipeline: Zentrale Steuerung und Adaptivität des Trading-Systems.

    Koordiniert:
    1. Regime-Erkennung (HMM)
    2. Hyperparameter-Optimierung (Bayesian + MAB)
    3. Kapitalallokation (Kelly + Risk-Parity)
    4. System-Adaptivität
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        n_regime_states: int = 7,
        max_leverage: float = 3.0
    ):
        self.config = config

        # Initialize components
        self.regime_detector = HMMRegimeDetector(n_states=n_regime_states)
        self.hyperparam_optimizer = BayesianHyperparameterOptimizer()
        self.capital_allocator = CapitalAllocationEngine(max_leverage=max_leverage)

        # State tracking
        self.current_state: Optional[MetaPipelineState] = None
        self.state_history: deque = deque(maxlen=1000)

        # Strategy registry
        self.registered_strategies: Dict[str, Dict] = {}
        self.strategy_performances: Dict[str, StrategyPerformance] = {}

        # Hyperparameter configurations
        self.hyperparameter_configs: Dict[str, HyperparameterConfig] = self._init_hyperparams()

        logger.info("MetaPipeline initialized with %d regime states", n_regime_states)

    def _init_hyperparams(self) -> Dict[str, HyperparameterConfig]:
        """Initialize default hyperparameter configurations"""
        now = datetime.utcnow()

        return {
            'lookback_period': HyperparameterConfig(
                name='lookback_period',
                current_value=168,
                min_value=24,
                max_value=720,
                step_size=24,
                last_updated=now
            ),
            'signal_threshold': HyperparameterConfig(
                name='signal_threshold',
                current_value=0.6,
                min_value=0.3,
                max_value=0.9,
                step_size=0.05,
                last_updated=now
            ),
            'position_size_multiplier': HyperparameterConfig(
                name='position_size_multiplier',
                current_value=1.0,
                min_value=0.2,
                max_value=2.0,
                step_size=0.1,
                last_updated=now
            ),
            'stop_loss_atr_multiplier': HyperparameterConfig(
                name='stop_loss_atr_multiplier',
                current_value=2.0,
                min_value=1.0,
                max_value=4.0,
                step_size=0.25,
                last_updated=now
            ),
            'take_profit_atr_multiplier': HyperparameterConfig(
                name='take_profit_atr_multiplier',
                current_value=3.0,
                min_value=1.5,
                max_value=6.0,
                step_size=0.25,
                last_updated=now
            ),
            'kelly_fraction': HyperparameterConfig(
                name='kelly_fraction',
                current_value=0.5,
                min_value=0.1,
                max_value=1.0,
                step_size=0.05,
                last_updated=now
            ),
            'regime_sensitivity': HyperparameterConfig(
                name='regime_sensitivity',
                current_value=1.0,
                min_value=0.5,
                max_value=2.0,
                step_size=0.1,
                last_updated=now
            ),
        }

    def register_strategy(
        self,
        name: str,
        strategy_type: str,
        initial_performance: Optional[StrategyPerformance] = None
    ):
        """Register a strategy with the meta-pipeline"""
        self.registered_strategies[name] = {
            'type': strategy_type,
            'registered_at': datetime.utcnow(),
            'active': True
        }

        if initial_performance:
            self.strategy_performances[name] = initial_performance
        else:
            self.strategy_performances[name] = StrategyPerformance(
                strategy_name=name,
                sharpe_ratio=0,
                sortino_ratio=0,
                calmar_ratio=0,
                win_rate=0.5,
                profit_factor=1.0,
                max_drawdown=0,
                current_drawdown=0,
                avg_return=0,
                volatility=0.1,
                information_ratio=0,
                kelly_fraction=0.1
            )

        logger.info(f"Registered strategy: {name}")

    def update_strategy_performance(
        self,
        name: str,
        performance: StrategyPerformance
    ):
        """Update strategy performance metrics"""
        if name in self.registered_strategies:
            self.strategy_performances[name] = performance

            # Update hyperparameter optimizer with new observation
            self.hyperparam_optimizer.update_observations(
                f"strategy_{name}",
                performance.sharpe_ratio,
                performance.sharpe_ratio
            )

    def process(
        self,
        market_data: Dict[str, pd.DataFrame],
        strategy_returns: Optional[Dict[str, pd.Series]] = None
    ) -> MetaPipelineState:
        """
        Main processing function.

        Args:
            market_data: Dict of symbol -> OHLCV DataFrames
            strategy_returns: Optional dict of strategy -> returns Series

        Returns:
            MetaPipelineState with recommendations
        """
        # Use BTC as primary market indicator
        primary_data = market_data.get('BTC/USDT', list(market_data.values())[0])

        # 1. Detect regime
        regime, confidence, transition_probs = self.regime_detector.detect_regime(primary_data)

        # 2. Train HMM if enough data
        if len(primary_data) >= self.regime_detector.lookback_periods:
            self._update_hmm(primary_data)

        # 3. Update strategy performances from returns
        if strategy_returns:
            self._update_performances_from_returns(strategy_returns)

        # 4. Calculate capital allocations
        allocations = self.capital_allocator.calculate_allocations(
            self.strategy_performances,
            regime,
            confidence
        )

        # 5. Suggest hyperparameter updates
        suggested_params = self.hyperparam_optimizer.suggest_hyperparameters(
            self.hyperparameter_configs
        )

        # Update hyperparameter configs with suggestions
        for param_name, value in suggested_params.items():
            if param_name in self.hyperparameter_configs:
                self.hyperparameter_configs[param_name].current_value = value
                self.hyperparameter_configs[param_name].last_updated = datetime.utcnow()

        # 6. Calculate global risk multiplier
        global_risk_mult = self._calculate_global_risk_multiplier(
            regime, confidence, primary_data
        )

        # 7. Calculate regime duration
        regime_duration = int(
            (datetime.utcnow() - self.regime_detector.regime_start_time).total_seconds() / 3600
        )

        # Create state
        state = MetaPipelineState(
            current_regime=regime,
            regime_confidence=confidence,
            regime_duration=regime_duration,
            transition_probabilities=transition_probs,
            strategy_allocations=allocations,
            hyperparameter_configs=self.hyperparameter_configs.copy(),
            global_risk_multiplier=global_risk_mult,
            timestamp=datetime.utcnow()
        )

        self.current_state = state
        self.state_history.append(state)

        logger.info(
            f"MetaPipeline processed: Regime={regime.name}, "
            f"Confidence={confidence:.2f}, Risk Mult={global_risk_mult:.2f}"
        )

        return state

    def _update_hmm(self, market_data: pd.DataFrame):
        """Update HMM parameters with recent data"""
        # Extract feature sequence
        feature_sequence = []
        for i in range(max(20, len(market_data) - 168), len(market_data)):
            feat = self.regime_detector.extract_features(market_data.iloc[:i+1])
            feature_sequence.append(feat)

        if len(feature_sequence) > 30:
            observations = np.array(feature_sequence)
            self.regime_detector.baum_welch_update(observations, n_iterations=5)

    def _update_performances_from_returns(
        self,
        strategy_returns: Dict[str, pd.Series]
    ):
        """Update strategy performances from return series"""
        for name, returns in strategy_returns.items():
            if len(returns) < 10:
                continue

            # Calculate performance metrics
            avg_return = returns.mean()
            volatility = returns.std() * np.sqrt(252 * 24)

            # Sharpe
            sharpe = (avg_return * 252 * 24 - 0.05) / (volatility + 1e-10)

            # Sortino
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() * np.sqrt(252 * 24) if len(downside_returns) > 0 else volatility
            sortino = (avg_return * 252 * 24 - 0.05) / (downside_std + 1e-10)

            # Win rate
            win_rate = (returns > 0).mean()

            # Profit factor
            gains = returns[returns > 0].sum()
            losses = abs(returns[returns < 0].sum())
            profit_factor = gains / (losses + 1e-10)

            # Drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdowns = (cumulative - running_max) / running_max
            max_drawdown = abs(drawdowns.min())
            current_drawdown = abs(drawdowns.iloc[-1])

            # Calmar
            calmar = (avg_return * 252 * 24) / (max_drawdown + 1e-10)

            # Kelly
            kelly = (avg_return / (volatility ** 2 + 1e-10)) if volatility > 0 else 0

            perf = StrategyPerformance(
                strategy_name=name,
                sharpe_ratio=sharpe,
                sortino_ratio=sortino,
                calmar_ratio=calmar,
                win_rate=win_rate,
                profit_factor=profit_factor,
                max_drawdown=max_drawdown,
                current_drawdown=current_drawdown,
                avg_return=avg_return,
                volatility=volatility,
                information_ratio=sharpe,  # Simplified
                kelly_fraction=kelly
            )

            self.update_strategy_performance(name, perf)

    def _calculate_global_risk_multiplier(
        self,
        regime: HMMRegimeState,
        confidence: float,
        market_data: pd.DataFrame
    ) -> float:
        """Calculate global risk multiplier based on regime and market conditions"""
        # Base multiplier from regime
        regime_multipliers = {
            HMMRegimeState.CRISIS: 0.2,
            HMMRegimeState.HIGH_VOLATILITY: 0.4,
            HMMRegimeState.RISK_OFF: 0.6,
            HMMRegimeState.NEUTRAL: 1.0,
            HMMRegimeState.RISK_ON: 1.2,
            HMMRegimeState.LOW_VOLATILITY: 0.9,
            HMMRegimeState.EUPHORIA: 0.5,
        }

        base_mult = regime_multipliers.get(regime, 1.0)

        # Adjust by confidence
        confidence_adj = 0.5 + 0.5 * confidence

        # Volatility adjustment
        if len(market_data) > 20:
            returns = market_data['close'].pct_change().dropna()
            current_vol = returns.iloc[-20:].std() * np.sqrt(252 * 24)
            historical_vol = returns.std() * np.sqrt(252 * 24)

            vol_ratio = current_vol / (historical_vol + 1e-10)
            vol_adj = 1 / (1 + max(0, vol_ratio - 1))  # Reduce risk when vol is elevated
        else:
            vol_adj = 1.0

        final_mult = base_mult * confidence_adj * vol_adj

        return np.clip(final_mult, 0.1, 1.5)

    def get_recommended_hyperparameters(self) -> Dict[str, float]:
        """Get current recommended hyperparameters"""
        return {
            name: config.current_value
            for name, config in self.hyperparameter_configs.items()
        }

    def get_strategy_allocations(self) -> Dict[str, float]:
        """Get current strategy allocations"""
        if self.current_state is None:
            return {}

        return {
            name: alloc.final_weight
            for name, alloc in self.current_state.strategy_allocations.items()
        }

    def get_regime_info(self) -> Dict[str, Any]:
        """Get current regime information"""
        if self.current_state is None:
            return {}

        return {
            'regime': self.current_state.current_regime.name,
            'confidence': self.current_state.regime_confidence,
            'duration_hours': self.current_state.regime_duration,
            'transition_probs': {
                k.name: v for k, v in self.current_state.transition_probabilities.items()
            },
            'risk_multiplier': self.current_state.global_risk_multiplier
        }

    def should_reduce_risk(self) -> Tuple[bool, str]:
        """Check if risk should be reduced based on meta-analysis"""
        if self.current_state is None:
            return False, "No state available"

        # High-risk regimes
        high_risk_regimes = {
            HMMRegimeState.CRISIS,
            HMMRegimeState.HIGH_VOLATILITY,
            HMMRegimeState.EUPHORIA
        }

        if self.current_state.current_regime in high_risk_regimes:
            return True, f"High-risk regime: {self.current_state.current_regime.name}"

        # Low confidence
        if self.current_state.regime_confidence < 0.5:
            return True, f"Low regime confidence: {self.current_state.regime_confidence:.2f}"

        # Regime transition likely
        crisis_prob = self.current_state.transition_probabilities.get(HMMRegimeState.CRISIS, 0)
        if crisis_prob > 0.3:
            return True, f"High crisis transition probability: {crisis_prob:.2f}"

        return False, "No risk reduction needed"

    def get_trading_guidance(self) -> Dict[str, Any]:
        """Get comprehensive trading guidance from meta-pipeline"""
        if self.current_state is None:
            return {'error': 'Meta-pipeline not initialized'}

        should_reduce, reason = self.should_reduce_risk()

        return {
            'regime': self.current_state.current_regime.name,
            'regime_confidence': self.current_state.regime_confidence,
            'risk_multiplier': self.current_state.global_risk_multiplier,
            'reduce_risk': should_reduce,
            'risk_reason': reason,
            'strategy_weights': self.get_strategy_allocations(),
            'hyperparameters': self.get_recommended_hyperparameters(),
            'guidance': self._generate_guidance()
        }

    def _generate_guidance(self) -> List[str]:
        """Generate trading guidance based on current state"""
        guidance = []

        if self.current_state is None:
            return ["Initialize meta-pipeline first"]

        regime = self.current_state.current_regime

        regime_guidance = {
            HMMRegimeState.CRISIS: [
                "Reduce all positions to minimum",
                "Focus on capital preservation",
                "Consider increasing cash allocation",
                "Avoid new entries"
            ],
            HMMRegimeState.HIGH_VOLATILITY: [
                "Reduce position sizes",
                "Widen stop-losses",
                "Favor shorter holding periods",
                "Consider volatility strategies"
            ],
            HMMRegimeState.RISK_OFF: [
                "Bias towards defensive positions",
                "Reduce leverage",
                "Focus on quality assets",
                "Consider hedging strategies"
            ],
            HMMRegimeState.NEUTRAL: [
                "Standard position sizing",
                "Balance long/short exposure",
                "Focus on alpha generation",
                "Monitor for regime change"
            ],
            HMMRegimeState.RISK_ON: [
                "Can increase position sizes",
                "Bias towards momentum strategies",
                "Consider higher leverage if appropriate",
                "Watch for euphoria signals"
            ],
            HMMRegimeState.LOW_VOLATILITY: [
                "Prepare for potential breakout",
                "Consider options strategies",
                "Moderate position sizes",
                "Set tight stop-losses"
            ],
            HMMRegimeState.EUPHORIA: [
                "Begin reducing positions",
                "Take profits on winners",
                "Tighten stop-losses",
                "Avoid FOMO entries"
            ],
        }

        guidance.extend(regime_guidance.get(regime, ["Standard trading"]))

        # Add confidence-based guidance
        if self.current_state.regime_confidence < 0.6:
            guidance.append("Low confidence - reduce position sizes")

        return guidance


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    'HMMRegimeState',
    'RegimeTransitionProbability',
    'StrategyPerformance',
    'CapitalAllocation',
    'HyperparameterConfig',
    'MetaPipelineState',
    'HMMRegimeDetector',
    'BayesianHyperparameterOptimizer',
    'CapitalAllocationEngine',
    'MetaPipeline',
]
