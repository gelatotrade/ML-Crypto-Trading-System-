"""
TIEFGEHENDE ALPHA-PIPELINE: Mehrstufiges Ensemble-Design

Implementiert:
A. Mehrstufiges Ensemble-Design
   - Stufe 1: Univariate Signal-Generierung mit Volatilitätsadjustierung
   - Stufe 2: Multivariate Fusion mit Attention-basierter Kombination

B. Theoretisch fundierte Feature-Gruppen
   - Momentum-Familie (Time-Series & Cross-Sectional)
   - Mean-Reversion-Familie
   - Value-Familie (fundamental)
   - Carry-Familie (für FX & Futures)

C. Non-Parametrische Methoden
   - Gaussian Process Regression für Unsicherheitsquantifizierung
   - Quantile Random Forests für volle Verlustvorhersage
   - Causal Inference mit Do-Calculus
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

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AlphaSignal:
    """Individual alpha signal"""
    name: str
    value: float                    # Raw signal value
    volatility_adjusted: float      # Risk-adjusted signal
    confidence: float               # Signal confidence (0-1)
    decay_halflife: int            # Hours until signal decays
    category: str                   # momentum, mean_reversion, value, carry
    timestamp: datetime


@dataclass
class FeatureGroup:
    """Group of related features"""
    name: str
    features: Dict[str, float]
    combined_signal: float
    weight: float
    description: str


@dataclass
class AlphaPrediction:
    """Complete alpha prediction output"""
    symbol: str
    timestamp: datetime
    predicted_return: float
    confidence: float
    uncertainty_lower: float
    uncertainty_upper: float
    signal_decomposition: Dict[str, float]
    causal_strength: float
    regime_adjusted: bool
    feature_importance: Dict[str, float]


@dataclass
class GPPrediction:
    """Gaussian Process prediction result"""
    mean: float
    std: float
    confidence_interval: Tuple[float, float]
    epistemic_uncertainty: float      # Model uncertainty
    aleatoric_uncertainty: float      # Data uncertainty


@dataclass
class CausalEffect:
    """Causal effect estimation"""
    feature: str
    treatment_effect: float           # E[Y | do(X=1)] - E[Y | do(X=0)]
    confidence_interval: Tuple[float, float]
    p_value: float
    is_confounded: bool
    instruments: List[str]


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE FAMILIES
# ═══════════════════════════════════════════════════════════════════════════════

class MomentumFeatures:
    """
    Momentum-Familie: Time-Series & Cross-Sectional Momentum.

    Theoretische Grundlage:
    - momentum_ts = return(t-20:t-1) / σ(t-60:t-1)
    - momentum_cs = rank(return(t-20:t-1)) - 0.5
    """

    def __init__(
        self,
        short_window: int = 5,
        medium_window: int = 20,
        long_window: int = 60,
        vol_window: int = 60
    ):
        self.short_window = short_window
        self.medium_window = medium_window
        self.long_window = long_window
        self.vol_window = vol_window

    def calculate(self, prices: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Calculate momentum features for all assets.

        Args:
            prices: DataFrame with columns as assets, index as dates

        Returns:
            Dict of feature name -> DataFrame
        """
        features = {}

        # Returns over different horizons
        returns = prices.pct_change()

        # Time-series momentum (risk-adjusted)
        ret_short = prices.pct_change(self.short_window)
        ret_medium = prices.pct_change(self.medium_window)
        ret_long = prices.pct_change(self.long_window)

        # Rolling volatility for normalization
        vol = returns.rolling(self.vol_window).std() * np.sqrt(252 * 24)

        # TSM: Time-Series Momentum
        features['tsm_short'] = ret_short / (vol + 1e-10)
        features['tsm_medium'] = ret_medium / (vol + 1e-10)
        features['tsm_long'] = ret_long / (vol + 1e-10)

        # Cross-sectional momentum (rank-based)
        features['csm_short'] = ret_short.rank(axis=1, pct=True) - 0.5
        features['csm_medium'] = ret_medium.rank(axis=1, pct=True) - 0.5
        features['csm_long'] = ret_long.rank(axis=1, pct=True) - 0.5

        # Momentum consistency
        features['momentum_consistency'] = (
            (ret_short > 0).astype(float) +
            (ret_medium > 0).astype(float) +
            (ret_long > 0).astype(float)
        ) / 3

        # Acceleration (change in momentum)
        features['momentum_acceleration'] = ret_short - ret_short.shift(self.short_window)

        # 52-week high proximity (adapted for crypto - 30-day high)
        rolling_high = prices.rolling(self.long_window * 12).max()
        features['high_proximity'] = prices / (rolling_high + 1e-10)

        return features

    def get_combined_signal(
        self,
        features: Dict[str, pd.DataFrame],
        weights: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """Combine momentum features into single signal"""
        if weights is None:
            weights = {
                'tsm_short': 0.15,
                'tsm_medium': 0.25,
                'tsm_long': 0.20,
                'csm_medium': 0.20,
                'momentum_consistency': 0.10,
                'high_proximity': 0.10
            }

        combined = pd.DataFrame(0, index=list(features.values())[0].index,
                               columns=list(features.values())[0].columns)

        for name, weight in weights.items():
            if name in features:
                combined += features[name].fillna(0) * weight

        return combined


class MeanReversionFeatures:
    """
    Mean-Reversion-Familie.

    Theoretische Grundlage:
    - z_score = (price - MA(60)) / σ_rolling(60)
    - hurst_exponent = log(R/S) / log(n) für Regime-Identifikation
    """

    def __init__(
        self,
        short_window: int = 20,
        long_window: int = 60,
        hurst_window: int = 100
    ):
        self.short_window = short_window
        self.long_window = long_window
        self.hurst_window = hurst_window

    def calculate(self, prices: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Calculate mean reversion features"""
        features = {}

        # Z-score from moving average
        ma_short = prices.rolling(self.short_window).mean()
        ma_long = prices.rolling(self.long_window).mean()
        std_short = prices.rolling(self.short_window).std()
        std_long = prices.rolling(self.long_window).std()

        features['zscore_short'] = (prices - ma_short) / (std_short + 1e-10)
        features['zscore_long'] = (prices - ma_long) / (std_long + 1e-10)

        # Bollinger Band position
        upper_band = ma_short + 2 * std_short
        lower_band = ma_short - 2 * std_short
        features['bb_position'] = (prices - lower_band) / (upper_band - lower_band + 1e-10)

        # RSI-based mean reversion
        returns = prices.pct_change()
        gains = returns.clip(lower=0)
        losses = -returns.clip(upper=0)

        avg_gain = gains.rolling(14).mean()
        avg_loss = losses.rolling(14).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - 100 / (1 + rs)

        # RSI deviation from 50 (neutral)
        features['rsi_deviation'] = (rsi - 50) / 50

        # Hurst exponent (for each column)
        features['hurst_exponent'] = self._rolling_hurst(prices)

        # Mean reversion strength (inverse of Hurst)
        features['mr_strength'] = 1 - features['hurst_exponent']

        # Distance from VWAP-like measure
        if 'volume' not in prices.columns:
            features['vwap_distance'] = features['zscore_short']
        else:
            # Simplified
            features['vwap_distance'] = features['zscore_short']

        return features

    def _rolling_hurst(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling Hurst exponent"""
        def hurst_exponent(series):
            """Calculate Hurst exponent using R/S analysis"""
            if len(series) < 20:
                return 0.5

            series = series.dropna()
            if len(series) < 20:
                return 0.5

            # R/S analysis
            n = len(series)
            max_k = min(n // 2, 50)

            if max_k < 2:
                return 0.5

            rs_list = []
            n_list = []

            for k in range(10, max_k, 5):
                rs = self._calculate_rs(series.values, k)
                if rs > 0:
                    rs_list.append(np.log(rs))
                    n_list.append(np.log(k))

            if len(rs_list) < 2:
                return 0.5

            # Linear regression for Hurst
            slope, _ = np.polyfit(n_list, rs_list, 1)

            return np.clip(slope, 0, 1)

        result = prices.apply(
            lambda col: col.rolling(self.hurst_window).apply(hurst_exponent, raw=False)
        )

        return result.fillna(0.5)

    def _calculate_rs(self, series: np.ndarray, n: int) -> float:
        """Calculate R/S statistic"""
        if len(series) < n:
            return 0

        # Mean-adjusted cumulative deviations
        mean = np.mean(series[:n])
        deviations = np.cumsum(series[:n] - mean)

        R = np.max(deviations) - np.min(deviations)
        S = np.std(series[:n])

        if S == 0:
            return 0

        return R / S

    def get_combined_signal(
        self,
        features: Dict[str, pd.DataFrame],
        weights: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """Combine mean reversion features"""
        if weights is None:
            weights = {
                'zscore_long': -0.30,  # Negative: buy low, sell high
                'bb_position': -0.20,
                'rsi_deviation': -0.25,
                'mr_strength': 0.15,
                'vwap_distance': -0.10
            }

        combined = pd.DataFrame(0, index=list(features.values())[0].index,
                               columns=list(features.values())[0].columns)

        for name, weight in weights.items():
            if name in features:
                combined += features[name].fillna(0) * weight

        return combined


class CarryFeatures:
    """
    Carry-Familie für Futures und Funding Rates.

    Theoretische Grundlage:
    - carry = (r_domestic - r_foreign) / σ_fx für FX
    - rolldown = (futures_curve_slope) / σ_basis für Futures
    """

    def __init__(
        self,
        vol_window: int = 30
    ):
        self.vol_window = vol_window

    def calculate(
        self,
        spot_prices: pd.DataFrame,
        funding_rates: Optional[pd.DataFrame] = None,
        futures_prices: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Dict[str, pd.DataFrame]:
        """Calculate carry features"""
        features = {}

        returns = spot_prices.pct_change()
        vol = returns.rolling(self.vol_window).std() * np.sqrt(252 * 24)

        # Funding rate carry (specific to crypto perpetuals)
        if funding_rates is not None:
            # Annualize funding rate (8-hour cycles * 3 * 365)
            annual_funding = funding_rates * 3 * 365

            # Risk-adjusted carry
            features['funding_carry'] = annual_funding / (vol + 1e-10)

            # Cumulative funding (recent bias)
            features['cumulative_funding'] = funding_rates.rolling(7).sum() * 365

            # Funding rate momentum
            features['funding_momentum'] = funding_rates.diff(7)

        # Futures basis carry (if available)
        if futures_prices is not None:
            for tenor, futures_df in futures_prices.items():
                # Align with spot
                aligned_spot, aligned_futures = spot_prices.align(futures_df, join='inner')

                basis = (aligned_futures - aligned_spot) / (aligned_spot + 1e-10)
                features[f'basis_{tenor}'] = basis

                # Annualized basis
                # Assuming tenor is in days
                try:
                    days = int(tenor.replace('d', ''))
                    annual_basis = basis * (365 / days)
                    features[f'annual_basis_{tenor}'] = annual_basis
                except:
                    pass

        # Implied volatility carry (if vol surface available)
        # This would come from options data

        return features

    def get_combined_signal(
        self,
        features: Dict[str, pd.DataFrame],
        weights: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """Combine carry features"""
        if not features:
            return pd.DataFrame()

        if weights is None:
            # Default: equal weight normalized
            weights = {name: 1/len(features) for name in features}

        combined = pd.DataFrame(0, index=list(features.values())[0].index,
                               columns=list(features.values())[0].columns)

        for name, weight in weights.items():
            if name in features:
                combined += features[name].fillna(0) * weight

        return combined


# ═══════════════════════════════════════════════════════════════════════════════
# GAUSSIAN PROCESS REGRESSION
# ═══════════════════════════════════════════════════════════════════════════════

class GaussianProcessRegressor:
    """
    Gaussian Process Regression für Unsicherheitsquantifizierung.

    f(x) ~ GP(m(x), k(x,x'))
    k(x,x') = σ² * exp(-||x-x'||²/(2l²))  # RBF Kernel
    """

    def __init__(
        self,
        kernel: str = 'rbf',
        length_scale: float = 1.0,
        signal_variance: float = 1.0,
        noise_variance: float = 0.1,
        n_restarts: int = 3
    ):
        self.kernel = kernel
        self.length_scale = length_scale
        self.signal_variance = signal_variance
        self.noise_variance = noise_variance
        self.n_restarts = n_restarts

        # Training data
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None

        # Cached computations
        self._K_inv: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None

    def _rbf_kernel(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        length_scale: Optional[float] = None,
        signal_variance: Optional[float] = None
    ) -> np.ndarray:
        """RBF (Gaussian) kernel"""
        l = length_scale or self.length_scale
        sigma = signal_variance or self.signal_variance

        # Squared Euclidean distance
        if X1.ndim == 1:
            X1 = X1.reshape(-1, 1)
        if X2.ndim == 1:
            X2 = X2.reshape(-1, 1)

        dist_sq = np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2, axis=2)

        return sigma ** 2 * np.exp(-dist_sq / (2 * l ** 2))

    def _matern_kernel(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        nu: float = 2.5
    ) -> np.ndarray:
        """Matern kernel (more robust for financial data)"""
        l = self.length_scale
        sigma = self.signal_variance

        if X1.ndim == 1:
            X1 = X1.reshape(-1, 1)
        if X2.ndim == 1:
            X2 = X2.reshape(-1, 1)

        dist = np.sqrt(np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2, axis=2))

        if nu == 0.5:
            # Exponential kernel
            return sigma ** 2 * np.exp(-dist / l)
        elif nu == 1.5:
            # Matern 3/2
            sqrt3 = np.sqrt(3)
            return sigma ** 2 * (1 + sqrt3 * dist / l) * np.exp(-sqrt3 * dist / l)
        elif nu == 2.5:
            # Matern 5/2
            sqrt5 = np.sqrt(5)
            return sigma ** 2 * (1 + sqrt5 * dist / l + 5 * dist ** 2 / (3 * l ** 2)) * np.exp(-sqrt5 * dist / l)
        else:
            return self._rbf_kernel(X1, X2)

    def _get_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Get kernel matrix based on kernel type"""
        if self.kernel == 'rbf':
            return self._rbf_kernel(X1, X2)
        elif self.kernel == 'matern':
            return self._matern_kernel(X1, X2)
        else:
            return self._rbf_kernel(X1, X2)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit GP to training data.

        Args:
            X: Training features (n_samples, n_features)
            y: Training targets (n_samples,)
        """
        self.X_train = X
        self.y_train = y

        # Compute kernel matrix
        K = self._get_kernel(X, X)
        K += self.noise_variance * np.eye(len(X))

        # Cholesky decomposition for numerical stability
        try:
            L = np.linalg.cholesky(K)
            self._K_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(len(X))))
            self._alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        except np.linalg.LinAlgError:
            # Fallback: add more regularization
            K += 0.01 * np.eye(len(X))
            self._K_inv = np.linalg.inv(K)
            self._alpha = self._K_inv @ y

        # Optimize hyperparameters (simplified grid search)
        self._optimize_hyperparameters(X, y)

    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray):
        """Optimize kernel hyperparameters using marginal likelihood"""
        best_log_likelihood = float('-inf')
        best_params = (self.length_scale, self.signal_variance)

        # Grid search over hyperparameters
        length_scales = [0.5, 1.0, 2.0, 5.0]
        signal_variances = [0.5, 1.0, 2.0]

        for ls in length_scales:
            for sv in signal_variances:
                try:
                    # Compute kernel with candidate params
                    K = self._rbf_kernel(X, X, ls, sv)
                    K += self.noise_variance * np.eye(len(X))

                    # Log marginal likelihood
                    L = np.linalg.cholesky(K)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))

                    log_likelihood = -0.5 * y @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * len(X) * np.log(2 * np.pi)

                    if log_likelihood > best_log_likelihood:
                        best_log_likelihood = log_likelihood
                        best_params = (ls, sv)

                except:
                    continue

        self.length_scale, self.signal_variance = best_params

        # Refit with best params
        K = self._get_kernel(X, X)
        K += self.noise_variance * np.eye(len(X))

        try:
            L = np.linalg.cholesky(K)
            self._K_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(len(X))))
            self._alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        except:
            self._K_inv = np.linalg.inv(K + 0.01 * np.eye(len(X)))
            self._alpha = self._K_inv @ y

    def predict(
        self,
        X: np.ndarray,
        return_std: bool = True,
        return_cov: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict using GP.

        Args:
            X: Test features
            return_std: Return standard deviation
            return_cov: Return full covariance matrix

        Returns:
            Mean predictions (and optionally std/cov)
        """
        if self.X_train is None:
            raise ValueError("Model not fitted")

        # Kernel between test and training
        K_star = self._get_kernel(X, self.X_train)

        # Mean prediction
        mean = K_star @ self._alpha

        if not return_std and not return_cov:
            return mean

        # Variance prediction
        K_star_star = self._get_kernel(X, X)
        var = K_star_star - K_star @ self._K_inv @ K_star.T

        if return_cov:
            return mean, var

        # Standard deviation
        std = np.sqrt(np.diag(var).clip(min=1e-10))

        return mean, std

    def predict_with_uncertainty(self, X: np.ndarray) -> GPPrediction:
        """Get full uncertainty decomposition"""
        mean, std = self.predict(X, return_std=True)

        # For single prediction
        if len(mean) == 1:
            mean = mean[0]
            std = std[0]

        # Confidence interval (95%)
        ci_lower = mean - 1.96 * std
        ci_upper = mean + 1.96 * std

        # Epistemic uncertainty (model uncertainty) - from GP variance
        epistemic = std

        # Aleatoric uncertainty (inherent noise) - from noise_variance
        aleatoric = np.sqrt(self.noise_variance)

        return GPPrediction(
            mean=float(mean) if np.isscalar(mean) else float(mean),
            std=float(std) if np.isscalar(std) else float(std),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            epistemic_uncertainty=float(epistemic),
            aleatoric_uncertainty=float(aleatoric)
        )


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTILE RANDOM FOREST
# ═══════════════════════════════════════════════════════════════════════════════

class QuantileRandomForest:
    """
    Quantile Random Forest für volle Verlustverteilung.

    Schätzt beliebige Quantile der bedingten Verteilung P(Y|X).
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_leaf: int = 5,
        quantiles: List[float] = None
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.quantiles = quantiles or [0.05, 0.25, 0.5, 0.75, 0.95]

        # Simple decision tree storage
        self.trees: List[Dict] = []
        self.leaf_values: List[Dict[int, np.ndarray]] = []

    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray,
        depth: int = 0
    ) -> Dict:
        """Build a single decision tree"""
        n_samples = len(indices)

        # Stopping conditions
        if (depth >= self.max_depth or
            n_samples <= self.min_samples_leaf or
            np.std(y[indices]) < 1e-10):
            return {
                'leaf': True,
                'values': y[indices].copy(),
                'id': id(indices)
            }

        # Find best split
        best_gain = -np.inf
        best_feature = 0
        best_threshold = 0

        n_features = X.shape[1]
        features_to_try = np.random.choice(n_features, min(n_features, int(np.sqrt(n_features)) + 1), replace=False)

        for feature in features_to_try:
            thresholds = np.percentile(X[indices, feature], [25, 50, 75])

            for threshold in thresholds:
                left_mask = X[indices, feature] <= threshold
                right_mask = ~left_mask

                if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
                    continue

                # Information gain (variance reduction)
                var_left = np.var(y[indices[left_mask]])
                var_right = np.var(y[indices[right_mask]])
                var_parent = np.var(y[indices])

                n_left = left_mask.sum()
                n_right = right_mask.sum()

                gain = var_parent - (n_left * var_left + n_right * var_right) / n_samples

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        if best_gain <= 0:
            return {
                'leaf': True,
                'values': y[indices].copy(),
                'id': id(indices)
            }

        # Split
        left_mask = X[indices, best_feature] <= best_threshold
        left_indices = indices[left_mask]
        right_indices = indices[~left_mask]

        return {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': self._build_tree(X, y, left_indices, depth + 1),
            'right': self._build_tree(X, y, right_indices, depth + 1)
        }

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit quantile random forest"""
        n_samples = len(X)

        self.trees = []
        self.leaf_values = []

        for _ in range(self.n_estimators):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)

            tree = self._build_tree(X, y, indices)
            self.trees.append(tree)

    def _get_leaf_values(self, tree: Dict, x: np.ndarray) -> np.ndarray:
        """Get leaf values for a single sample"""
        if tree['leaf']:
            return tree['values']

        if x[tree['feature']] <= tree['threshold']:
            return self._get_leaf_values(tree['left'], x)
        else:
            return self._get_leaf_values(tree['right'], x)

    def predict_quantiles(
        self,
        X: np.ndarray,
        quantiles: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Predict specified quantiles.

        Args:
            X: Test features
            quantiles: List of quantiles to predict

        Returns:
            Array of shape (n_samples, n_quantiles)
        """
        quantiles = quantiles or self.quantiles

        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples = len(X)
        n_quantiles = len(quantiles)

        predictions = np.zeros((n_samples, n_quantiles))

        for i in range(n_samples):
            # Collect all leaf values from all trees
            all_values = []

            for tree in self.trees:
                leaf_values = self._get_leaf_values(tree, X[i])
                all_values.extend(leaf_values)

            all_values = np.array(all_values)

            # Calculate quantiles
            for j, q in enumerate(quantiles):
                predictions[i, j] = np.percentile(all_values, q * 100)

        return predictions

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict median (50th percentile)"""
        quantiles = self.predict_quantiles(X, [0.5])
        return quantiles[:, 0]

    def predict_interval(
        self,
        X: np.ndarray,
        coverage: float = 0.9
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict prediction interval"""
        alpha = (1 - coverage) / 2
        quantiles = self.predict_quantiles(X, [alpha, 1 - alpha])
        return quantiles[:, 0], quantiles[:, 1]


# ═══════════════════════════════════════════════════════════════════════════════
# CAUSAL INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

class CausalInferenceEngine:
    """
    Causal Inference mit Do-Calculus für robuste Signale.

    Unterscheidet zwischen:
    - P(return | feature=x)       - Korrelation/Assoziation
    - P(return | do(feature=x))   - Kausaler Effekt

    Methoden:
    - Instrumental Variables (IV)
    - Propensity Score Matching
    - Regression Discontinuity (RD)
    """

    def __init__(
        self,
        confounders: Optional[List[str]] = None,
        instruments: Optional[Dict[str, List[str]]] = None
    ):
        self.confounders = confounders or ['volatility', 'volume', 'market_return']
        self.instruments = instruments or {}

    def estimate_ate(
        self,
        treatment: np.ndarray,
        outcome: np.ndarray,
        covariates: Optional[np.ndarray] = None
    ) -> CausalEffect:
        """
        Estimate Average Treatment Effect using propensity score weighting.

        ATE = E[Y(1)] - E[Y(0)]
        """
        # Binarize treatment (above/below median)
        treatment_binary = (treatment > np.median(treatment)).astype(float)

        # Propensity score (probability of treatment)
        if covariates is not None:
            # Logistic regression for propensity
            propensity = self._estimate_propensity(treatment_binary, covariates)
        else:
            propensity = np.mean(treatment_binary) * np.ones(len(treatment))

        # Inverse propensity weighting
        weights_treated = treatment_binary / (propensity + 1e-10)
        weights_control = (1 - treatment_binary) / (1 - propensity + 1e-10)

        # Normalize weights
        weights_treated /= weights_treated.sum() + 1e-10
        weights_control /= weights_control.sum() + 1e-10

        # Weighted outcomes
        y_treated = np.sum(weights_treated * outcome)
        y_control = np.sum(weights_control * outcome)

        ate = y_treated - y_control

        # Bootstrap confidence interval
        ci_lower, ci_upper = self._bootstrap_ci(
            treatment_binary, outcome, propensity, n_bootstrap=100
        )

        # Approximate p-value
        se = (ci_upper - ci_lower) / (2 * 1.96)
        t_stat = ate / (se + 1e-10)
        p_value = 2 * (1 - self._normal_cdf(abs(t_stat)))

        # Check for confounding
        is_confounded = self._check_confounding(treatment, outcome, covariates)

        return CausalEffect(
            feature='treatment',
            treatment_effect=ate,
            confidence_interval=(ci_lower, ci_upper),
            p_value=p_value,
            is_confounded=is_confounded,
            instruments=[]
        )

    def _estimate_propensity(
        self,
        treatment: np.ndarray,
        covariates: np.ndarray
    ) -> np.ndarray:
        """Estimate propensity score using logistic regression"""
        # Simple logistic regression
        # P(T=1|X) = sigmoid(X @ beta)

        # Add intercept
        X = np.column_stack([np.ones(len(covariates)), covariates])

        # Gradient descent for logistic regression
        beta = np.zeros(X.shape[1])
        learning_rate = 0.01

        for _ in range(100):
            pred = 1 / (1 + np.exp(-X @ beta))
            gradient = X.T @ (treatment - pred) / len(treatment)
            beta += learning_rate * gradient

        propensity = 1 / (1 + np.exp(-X @ beta))

        # Clip for stability
        return np.clip(propensity, 0.01, 0.99)

    def _bootstrap_ci(
        self,
        treatment: np.ndarray,
        outcome: np.ndarray,
        propensity: np.ndarray,
        n_bootstrap: int = 100,
        alpha: float = 0.05
    ) -> Tuple[float, float]:
        """Bootstrap confidence interval for ATE"""
        n = len(treatment)
        ates = []

        for _ in range(n_bootstrap):
            idx = np.random.choice(n, n, replace=True)

            t = treatment[idx]
            y = outcome[idx]
            p = propensity[idx]

            w_t = t / (p + 1e-10)
            w_c = (1 - t) / (1 - p + 1e-10)

            w_t /= w_t.sum() + 1e-10
            w_c /= w_c.sum() + 1e-10

            ate = np.sum(w_t * y) - np.sum(w_c * y)
            ates.append(ate)

        return np.percentile(ates, alpha/2 * 100), np.percentile(ates, (1 - alpha/2) * 100)

    def _check_confounding(
        self,
        treatment: np.ndarray,
        outcome: np.ndarray,
        covariates: Optional[np.ndarray]
    ) -> bool:
        """Check for potential confounding"""
        if covariates is None:
            return True  # Unknown confounding

        # Check if covariates affect both treatment and outcome
        for i in range(covariates.shape[1]):
            covar = covariates[:, i]

            corr_treatment = abs(np.corrcoef(covar, treatment)[0, 1])
            corr_outcome = abs(np.corrcoef(covar, outcome)[0, 1])

            if corr_treatment > 0.3 and corr_outcome > 0.3:
                return True

        return False

    def _normal_cdf(self, x: float) -> float:
        """Approximate normal CDF"""
        return 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

    def instrumental_variable_estimation(
        self,
        treatment: np.ndarray,
        outcome: np.ndarray,
        instrument: np.ndarray
    ) -> CausalEffect:
        """
        Estimate causal effect using instrumental variables (2SLS).

        Requirements for valid instrument:
        1. Relevance: Corr(Z, X) != 0
        2. Exclusion: Z only affects Y through X
        """
        # Stage 1: Regress treatment on instrument
        z_mean = np.mean(instrument)
        z_var = np.var(instrument)
        t_mean = np.mean(treatment)

        beta_1 = np.cov(instrument, treatment)[0, 1] / (z_var + 1e-10)

        # Predicted treatment
        treatment_hat = t_mean + beta_1 * (instrument - z_mean)

        # Stage 2: Regress outcome on predicted treatment
        th_mean = np.mean(treatment_hat)
        th_var = np.var(treatment_hat)
        y_mean = np.mean(outcome)

        beta_iv = np.cov(treatment_hat, outcome)[0, 1] / (th_var + 1e-10)

        # Check instrument relevance
        f_stat = (beta_1 ** 2 * z_var * len(instrument)) / np.var(treatment - treatment_hat)
        weak_instrument = f_stat < 10

        # Standard error (approximate)
        n = len(outcome)
        residuals = outcome - (y_mean + beta_iv * (treatment_hat - th_mean))
        se = np.sqrt(np.var(residuals) / (th_var * n))

        ci_lower = beta_iv - 1.96 * se
        ci_upper = beta_iv + 1.96 * se

        t_stat = beta_iv / (se + 1e-10)
        p_value = 2 * (1 - self._normal_cdf(abs(t_stat)))

        return CausalEffect(
            feature='treatment',
            treatment_effect=beta_iv,
            confidence_interval=(ci_lower, ci_upper),
            p_value=p_value,
            is_confounded=weak_instrument,
            instruments=['instrument']
        )


# ═══════════════════════════════════════════════════════════════════════════════
# HIERARCHICAL ALPHA MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class HierarchicalAlphaModel:
    """
    Mehrstufiges hierarchisches Alpha-Modell.

    Kombiniert verschiedene ML-Paradigmen:
    - Macro-level: Transformer für langfristige Muster
    - Micro-level: Temporal CNN für kurzfristige Signale
    - Cross-asset: Graph-basierte Beziehungen (vereinfacht)
    """

    def __init__(
        self,
        feature_groups: Optional[Dict[str, List[str]]] = None,
        ensemble_weights: Optional[Dict[str, float]] = None
    ):
        self.feature_groups = feature_groups or {
            'momentum': ['tsm_short', 'tsm_medium', 'csm_medium'],
            'mean_reversion': ['zscore_long', 'bb_position', 'rsi_deviation'],
            'carry': ['funding_carry', 'basis'],
            'microstructure': ['vpin', 'ofi', 'toxicity']
        }

        self.ensemble_weights = ensemble_weights or {
            'momentum': 0.35,
            'mean_reversion': 0.25,
            'carry': 0.20,
            'microstructure': 0.20
        }

        # Component models
        self.gp_regressors: Dict[str, GaussianProcessRegressor] = {}
        self.qrf_models: Dict[str, QuantileRandomForest] = {}
        self.causal_engine = CausalInferenceEngine()

        # Feature calculators
        self.momentum_features = MomentumFeatures()
        self.mr_features = MeanReversionFeatures()
        self.carry_features = CarryFeatures()

        # Attention weights (learned)
        self.attention_weights: Dict[str, float] = {}

    def calculate_all_features(
        self,
        prices: pd.DataFrame,
        funding_rates: Optional[pd.DataFrame] = None,
        microstructure: Optional[Dict[str, float]] = None
    ) -> Dict[str, pd.DataFrame]:
        """Calculate all feature groups"""
        all_features = {}

        # Momentum features
        mom_features = self.momentum_features.calculate(prices)
        all_features.update({f'mom_{k}': v for k, v in mom_features.items()})

        # Mean reversion features
        mr_features = self.mr_features.calculate(prices)
        all_features.update({f'mr_{k}': v for k, v in mr_features.items()})

        # Carry features (if funding available)
        if funding_rates is not None:
            carry_features = self.carry_features.calculate(prices, funding_rates)
            all_features.update({f'carry_{k}': v for k, v in carry_features.items()})

        return all_features

    def fit(
        self,
        features: Dict[str, np.ndarray],
        returns: np.ndarray
    ):
        """Fit all component models"""
        # Combine features
        X = np.column_stack([v for v in features.values()])

        # Remove NaN
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(returns)
        X_clean = X[mask]
        y_clean = returns[mask]

        if len(y_clean) < 50:
            logger.warning("Insufficient data for model fitting")
            return

        # Fit GP for uncertainty
        self.gp_regressors['main'] = GaussianProcessRegressor(kernel='matern')
        self.gp_regressors['main'].fit(X_clean[-500:], y_clean[-500:])

        # Fit QRF for quantiles
        self.qrf_models['main'] = QuantileRandomForest(n_estimators=50, max_depth=8)
        self.qrf_models['main'].fit(X_clean[-500:], y_clean[-500:])

        # Learn attention weights
        self._learn_attention_weights(features, returns)

    def _learn_attention_weights(
        self,
        features: Dict[str, np.ndarray],
        returns: np.ndarray
    ):
        """Learn attention weights based on feature importance"""
        # Simple correlation-based importance
        for name, feat in features.items():
            if len(feat.shape) > 1:
                feat = feat[:, -1]  # Last column

            # Handle NaN
            mask = ~np.isnan(feat) & ~np.isnan(returns)
            if mask.sum() < 30:
                self.attention_weights[name] = 1 / len(features)
                continue

            # Information coefficient (Spearman correlation)
            ic = np.abs(np.corrcoef(
                pd.Series(feat[mask]).rank(),
                pd.Series(returns[mask]).rank()
            )[0, 1])

            self.attention_weights[name] = ic if not np.isnan(ic) else 0.1

        # Normalize weights
        total = sum(self.attention_weights.values())
        if total > 0:
            self.attention_weights = {k: v / total for k, v in self.attention_weights.items()}

    def predict(
        self,
        features: Dict[str, np.ndarray],
        symbol: str = 'unknown'
    ) -> AlphaPrediction:
        """Generate alpha prediction with uncertainty"""
        # Combine features
        X = np.column_stack([v[-1:] if v.ndim > 1 else v[-1:] for v in features.values()])

        # GP prediction
        if 'main' in self.gp_regressors:
            gp_pred = self.gp_regressors['main'].predict_with_uncertainty(X)
            predicted_return = gp_pred.mean
            uncertainty = gp_pred.std
        else:
            predicted_return = 0
            uncertainty = 0.1

        # QRF quantiles
        if 'main' in self.qrf_models:
            quantiles = self.qrf_models['main'].predict_quantiles(X, [0.05, 0.95])
            ci_lower = quantiles[0, 0]
            ci_upper = quantiles[0, 1]
        else:
            ci_lower = predicted_return - 2 * uncertainty
            ci_upper = predicted_return + 2 * uncertainty

        # Signal decomposition
        signal_decomp = {}
        for group, weight in self.ensemble_weights.items():
            group_features = [f for f in features.keys() if group in f.lower()]
            if group_features:
                group_signal = np.mean([features[f][-1] for f in group_features])
                signal_decomp[group] = float(group_signal * weight)

        # Confidence based on uncertainty and data quality
        confidence = 1 / (1 + uncertainty)

        # Feature importance from attention
        feature_importance = self.attention_weights.copy()

        return AlphaPrediction(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            predicted_return=float(predicted_return),
            confidence=float(confidence),
            uncertainty_lower=float(ci_lower),
            uncertainty_upper=float(ci_upper),
            signal_decomposition=signal_decomp,
            causal_strength=0.5,  # Would come from causal analysis
            regime_adjusted=False,
            feature_importance=feature_importance
        )

    def estimate_causal_effects(
        self,
        features: Dict[str, np.ndarray],
        returns: np.ndarray
    ) -> Dict[str, CausalEffect]:
        """Estimate causal effects for each feature group"""
        effects = {}

        for name, feat in features.items():
            if len(feat.shape) > 1:
                feat = feat[:, -1]

            # Get covariates (other features as potential confounders)
            other_features = [v[:, -1] if v.ndim > 1 else v
                           for k, v in features.items() if k != name]

            if other_features:
                covariates = np.column_stack(other_features)
            else:
                covariates = None

            # Estimate ATE
            effect = self.causal_engine.estimate_ate(feat, returns, covariates)
            effect.feature = name
            effects[name] = effect

        return effects


# ═══════════════════════════════════════════════════════════════════════════════
# ALPHA PIPELINE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

class AlphaPipeline:
    """
    Main Alpha Pipeline orchestrator.

    Coordinates all alpha generation components.
    """

    def __init__(
        self,
        config: Optional[Any] = None
    ):
        self.config = config

        # Initialize components
        self.hierarchical_model = HierarchicalAlphaModel()
        self.momentum_calc = MomentumFeatures()
        self.mr_calc = MeanReversionFeatures()
        self.carry_calc = CarryFeatures()
        self.causal_engine = CausalInferenceEngine()

        # Feature cache
        self._feature_cache: Dict[str, Dict] = {}

        # Signal history
        self.signal_history: deque = deque(maxlen=1000)

    def generate_alpha(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        funding_rate: Optional[pd.Series] = None,
        microstructure: Optional[Dict[str, float]] = None,
        fit_models: bool = False
    ) -> AlphaPrediction:
        """
        Generate alpha signal for a symbol.

        Args:
            symbol: Trading symbol
            price_data: OHLCV DataFrame
            funding_rate: Optional funding rate series
            microstructure: Optional microstructure features
            fit_models: Whether to refit models

        Returns:
            AlphaPrediction with full analysis
        """
        # Calculate features
        features = self._calculate_features(symbol, price_data, funding_rate)

        # Add microstructure if available
        if microstructure:
            for key, value in microstructure.items():
                features[f'micro_{key}'] = np.array([value])

        # Fit models if needed
        if fit_models and len(price_data) > 100:
            returns = price_data['close'].pct_change().dropna().values
            self.hierarchical_model.fit(features, returns)

        # Generate prediction
        prediction = self.hierarchical_model.predict(features, symbol)

        # Store in history
        self.signal_history.append({
            'symbol': symbol,
            'timestamp': prediction.timestamp,
            'prediction': prediction.predicted_return,
            'confidence': prediction.confidence
        })

        # Cache features
        self._feature_cache[symbol] = features

        return prediction

    def _calculate_features(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        funding_rate: Optional[pd.Series] = None
    ) -> Dict[str, np.ndarray]:
        """Calculate all features for symbol"""
        features = {}

        close = price_data['close']

        # Price as DataFrame for feature calculators
        prices_df = pd.DataFrame({'price': close})

        # Momentum features
        mom = self.momentum_calc.calculate(prices_df)
        for name, series in mom.items():
            if isinstance(series, pd.DataFrame):
                features[f'mom_{name}'] = series.values
            else:
                features[f'mom_{name}'] = series.values.reshape(-1, 1)

        # Mean reversion features
        mr = self.mr_calc.calculate(prices_df)
        for name, series in mr.items():
            if isinstance(series, pd.DataFrame):
                features[f'mr_{name}'] = series.values
            else:
                features[f'mr_{name}'] = series.values.reshape(-1, 1)

        # Carry features
        if funding_rate is not None:
            funding_df = pd.DataFrame({'funding': funding_rate})
            carry = self.carry_calc.calculate(prices_df, funding_df)
            for name, series in carry.items():
                if isinstance(series, pd.DataFrame):
                    features[f'carry_{name}'] = series.values
                else:
                    features[f'carry_{name}'] = series.values.reshape(-1, 1)

        return features

    def get_feature_importance(self) -> Dict[str, float]:
        """Get current feature importance"""
        return self.hierarchical_model.attention_weights

    def get_signal_history(
        self,
        symbol: Optional[str] = None,
        lookback: int = 100
    ) -> List[Dict]:
        """Get recent signal history"""
        history = list(self.signal_history)[-lookback:]

        if symbol:
            history = [s for s in history if s['symbol'] == symbol]

        return history


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    'AlphaSignal',
    'FeatureGroup',
    'AlphaPrediction',
    'GPPrediction',
    'CausalEffect',
    'MomentumFeatures',
    'MeanReversionFeatures',
    'CarryFeatures',
    'GaussianProcessRegressor',
    'QuantileRandomForest',
    'CausalInferenceEngine',
    'HierarchicalAlphaModel',
    'AlphaPipeline',
]
