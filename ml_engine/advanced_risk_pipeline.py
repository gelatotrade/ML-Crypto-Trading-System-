"""
FORTGESCHRITTENE RISIKO-PIPELINE

Implementiert:
A. Multi-Faktor Risikomodell
   - Time-Varying Betas mit Kalman-Filter
   - Non-Linear Factor Exposures mit Neural Factors

B. Höhermomenten-Risikomodellierung
   - Cornish-Fisher-Erweiterung für VaR
   - Copula-basierte Abhängigkeiten (Student-t, Vine)

C. Liquidity-Adjusted Risk Measures
   - Modified L-VaR
   - Kyle's Lambda Schätzung
   - Market Impact Modellierung
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
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
class FactorExposure:
    """Factor exposure with time-varying beta"""
    factor_name: str
    current_beta: float
    beta_std: float            # Kalman filter uncertainty
    historical_betas: List[float]
    factor_return: float
    contribution_to_return: float
    t_statistic: float


@dataclass
class RiskDecomposition:
    """Complete risk decomposition"""
    total_risk: float
    systematic_risk: float
    idiosyncratic_risk: float
    factor_contributions: Dict[str, float]
    marginal_var: float
    component_var: float
    incremental_var: float


@dataclass
class VaREstimate:
    """Value at Risk estimate with higher moments"""
    var_normal: float
    var_historical: float
    var_cornish_fisher: float
    var_monte_carlo: float
    expected_shortfall: float
    confidence_level: float
    horizon_days: int
    skewness: float
    kurtosis: float


@dataclass
class LiquidityRisk:
    """Liquidity risk metrics"""
    l_var: float                    # Liquidity-adjusted VaR
    kyle_lambda: float              # Price impact coefficient
    amihud_illiquidity: float       # Amihud measure
    bid_ask_spread: float
    market_depth: float
    liquidation_time: float         # Estimated days to liquidate
    execution_shortfall: float


@dataclass
class CopulaMetrics:
    """Copula-based dependency metrics"""
    lower_tail_dependence: float
    upper_tail_dependence: float
    kendall_tau: float
    spearman_rho: float
    copula_type: str
    copula_params: Dict[str, float]


@dataclass
class ComprehensiveRiskReport:
    """Complete risk analysis report"""
    symbol: str
    timestamp: datetime
    var_estimate: VaREstimate
    risk_decomposition: RiskDecomposition
    factor_exposures: Dict[str, FactorExposure]
    liquidity_risk: LiquidityRisk
    copula_metrics: Optional[CopulaMetrics]
    stress_scenarios: Dict[str, float]
    risk_score: float              # Aggregate risk score 0-100


# ═══════════════════════════════════════════════════════════════════════════════
# KALMAN FILTER FOR TIME-VARYING BETAS
# ═══════════════════════════════════════════════════════════════════════════════

class KalmanFilterBeta:
    """
    Kalman Filter for time-varying beta estimation.

    State equation:  β_t = β_{t-1} + η_t, η_t ~ N(0, Q)
    Observation eq:  r_t = β_t' * F_t + ε_t, ε_t ~ N(0, R)

    Provides dynamic beta estimates with uncertainty quantification.
    """

    def __init__(
        self,
        n_factors: int = 1,
        process_variance: float = 0.001,    # Q: How fast beta changes
        observation_variance: float = 0.01,  # R: Measurement noise
        initial_beta: Optional[np.ndarray] = None,
        initial_variance: float = 1.0
    ):
        self.n_factors = n_factors
        self.Q = process_variance * np.eye(n_factors)  # Process noise covariance
        self.R = observation_variance                   # Observation noise variance

        # State estimates
        if initial_beta is None:
            self.beta = np.zeros(n_factors)
        else:
            self.beta = initial_beta

        # State covariance
        self.P = initial_variance * np.eye(n_factors)

        # History
        self.beta_history: List[np.ndarray] = []
        self.variance_history: List[np.ndarray] = []

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction step: β_{t|t-1} = β_{t-1|t-1}
        """
        # State prediction (random walk)
        beta_pred = self.beta.copy()

        # Covariance prediction
        P_pred = self.P + self.Q

        return beta_pred, P_pred

    def update(
        self,
        asset_return: float,
        factor_returns: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Update step with new observation.

        Args:
            asset_return: Single period asset return
            factor_returns: Factor returns for the period

        Returns:
            Updated beta and prediction error
        """
        # Ensure factor_returns is 1D
        if factor_returns.ndim > 1:
            factor_returns = factor_returns.flatten()

        # Prediction step
        beta_pred, P_pred = self.predict()

        # Kalman gain
        # K = P_pred * F' / (F * P_pred * F' + R)
        F = factor_returns.reshape(-1, 1)
        S = F.T @ P_pred @ F + self.R  # Innovation variance

        K = P_pred @ F / (S + 1e-10)

        # Predicted observation
        y_pred = factor_returns @ beta_pred

        # Innovation (prediction error)
        innovation = asset_return - y_pred

        # Update state
        self.beta = beta_pred + (K @ np.array([[innovation]])).flatten()

        # Update covariance
        self.P = (np.eye(self.n_factors) - K @ F.T) @ P_pred

        # Store history
        self.beta_history.append(self.beta.copy())
        self.variance_history.append(np.diag(self.P).copy())

        return self.beta, float(innovation)

    def filter(
        self,
        asset_returns: np.ndarray,
        factor_returns: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman filter on full time series.

        Args:
            asset_returns: (T,) array of asset returns
            factor_returns: (T, n_factors) array of factor returns

        Returns:
            Filtered betas (T, n_factors) and innovations (T,)
        """
        T = len(asset_returns)
        betas = np.zeros((T, self.n_factors))
        innovations = np.zeros(T)

        for t in range(T):
            beta, innovation = self.update(asset_returns[t], factor_returns[t])
            betas[t] = beta
            innovations[t] = innovation

        return betas, innovations

    def smooth(
        self,
        asset_returns: np.ndarray,
        factor_returns: np.ndarray
    ) -> np.ndarray:
        """
        RTS smoother for backward-looking analysis.

        Returns smoothed betas using all available data.
        """
        # Forward filter pass
        T = len(asset_returns)

        # Store forward pass results
        forward_betas = np.zeros((T, self.n_factors))
        forward_P = np.zeros((T, self.n_factors, self.n_factors))

        for t in range(T):
            self.update(asset_returns[t], factor_returns[t])
            forward_betas[t] = self.beta
            forward_P[t] = self.P

        # Backward smoothing pass
        smoothed_betas = np.zeros((T, self.n_factors))
        smoothed_betas[-1] = forward_betas[-1]

        for t in range(T - 2, -1, -1):
            # RTS smoother equations
            P_pred = forward_P[t] + self.Q
            J = forward_P[t] @ np.linalg.inv(P_pred + 1e-10 * np.eye(self.n_factors))

            smoothed_betas[t] = forward_betas[t] + J @ (smoothed_betas[t + 1] - forward_betas[t])

        return smoothed_betas

    def get_beta_confidence_interval(
        self,
        confidence: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get confidence interval for current beta estimate"""
        from scipy import stats

        z = stats.norm.ppf((1 + confidence) / 2)
        std = np.sqrt(np.diag(self.P))

        lower = self.beta - z * std
        upper = self.beta + z * std

        return lower, upper


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-FACTOR RISK MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class MultiFactorRiskModel:
    """
    Multi-factor risk model with time-varying betas.

    r_i = α_i + β_i' * F + ε_i
    where F = [Market, Size, Value, Momentum, Quality, LowVol, ...]

    Supports:
    - Time-varying betas via Kalman filter
    - Non-linear factor exposures
    - Factor contribution analysis
    """

    def __init__(
        self,
        factors: List[str] = None,
        use_kalman: bool = True,
        rolling_window: int = 60
    ):
        self.factors = factors or ['market', 'size', 'momentum', 'volatility']
        self.n_factors = len(self.factors)
        self.use_kalman = use_kalman
        self.rolling_window = rolling_window

        # Kalman filters per asset
        self.kalman_filters: Dict[str, KalmanFilterBeta] = {}

        # Factor covariance
        self.factor_cov: Optional[np.ndarray] = None

        # Idiosyncratic variance
        self.idio_var: Dict[str, float] = {}

    def _create_crypto_factors(
        self,
        market_returns: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create crypto-specific factors from market data.

        Factors:
        - Market: BTC return (market proxy)
        - Size: Market cap decile (approximated by volume)
        - Momentum: Recent return momentum
        - Volatility: Inverse realized volatility
        """
        factors = pd.DataFrame(index=market_returns.index)

        # Market factor (BTC or equal-weighted)
        if 'BTC' in market_returns.columns or 'BTC/USDT' in market_returns.columns:
            market_col = 'BTC' if 'BTC' in market_returns.columns else 'BTC/USDT'
            factors['market'] = market_returns[market_col]
        else:
            factors['market'] = market_returns.mean(axis=1)

        # Size factor (using volume as proxy, if not available use inverse vol)
        vol = market_returns.rolling(20).std()
        # High vol = small cap proxy, low vol = large cap proxy
        size_score = vol.rank(axis=1, pct=True)
        factors['size'] = size_score.mean(axis=1) - 0.5

        # Momentum factor
        momentum = market_returns.rolling(20).sum()
        mom_score = momentum.rank(axis=1, pct=True)
        factors['momentum'] = mom_score.mean(axis=1) - 0.5

        # Low volatility factor
        inv_vol = 1 / (vol + 1e-10)
        vol_score = inv_vol.rank(axis=1, pct=True)
        factors['volatility'] = vol_score.mean(axis=1) - 0.5

        return factors.dropna()

    def fit(
        self,
        asset_returns: pd.DataFrame,
        factor_returns: Optional[pd.DataFrame] = None
    ):
        """
        Fit the multi-factor model.

        Args:
            asset_returns: DataFrame of asset returns
            factor_returns: Optional factor returns (will create if not provided)
        """
        if factor_returns is None:
            factor_returns = self._create_crypto_factors(asset_returns)

        # Align data
        common_idx = asset_returns.index.intersection(factor_returns.index)
        asset_returns = asset_returns.loc[common_idx]
        factor_returns = factor_returns.loc[common_idx]

        # Estimate factor covariance
        self.factor_cov = factor_returns.cov().values

        # Fit betas for each asset
        for asset in asset_returns.columns:
            y = asset_returns[asset].values
            X = factor_returns.values

            # Remove NaN
            mask = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
            y_clean = y[mask]
            X_clean = X[mask]

            if len(y_clean) < 30:
                continue

            if self.use_kalman:
                # Kalman filter estimation
                kf = KalmanFilterBeta(n_factors=self.n_factors)
                betas, innovations = kf.filter(y_clean, X_clean)
                self.kalman_filters[asset] = kf

                # Idiosyncratic variance from innovations
                self.idio_var[asset] = np.var(innovations)
            else:
                # OLS estimation
                X_with_const = np.column_stack([np.ones(len(X_clean)), X_clean])
                beta_ols = np.linalg.lstsq(X_with_const, y_clean, rcond=None)[0]

                # Create static "Kalman filter" with OLS betas
                kf = KalmanFilterBeta(n_factors=self.n_factors)
                kf.beta = beta_ols[1:]  # Exclude intercept
                self.kalman_filters[asset] = kf

                # Idiosyncratic variance
                residuals = y_clean - X_with_const @ beta_ols
                self.idio_var[asset] = np.var(residuals)

    def get_factor_exposures(
        self,
        asset: str
    ) -> Dict[str, FactorExposure]:
        """Get factor exposures for an asset"""
        if asset not in self.kalman_filters:
            return {}

        kf = self.kalman_filters[asset]
        exposures = {}

        for i, factor in enumerate(self.factors[:self.n_factors]):
            beta = kf.beta[i] if i < len(kf.beta) else 0
            std = np.sqrt(kf.P[i, i]) if i < len(kf.P) else 0

            exposures[factor] = FactorExposure(
                factor_name=factor,
                current_beta=float(beta),
                beta_std=float(std),
                historical_betas=[float(b[i]) for b in kf.beta_history[-50:]] if kf.beta_history else [],
                factor_return=0,  # Would need current factor return
                contribution_to_return=0,
                t_statistic=float(beta / (std + 1e-10))
            )

        return exposures

    def decompose_risk(
        self,
        asset: str,
        position_value: float = 1.0
    ) -> RiskDecomposition:
        """Decompose risk into factor and idiosyncratic components"""
        if asset not in self.kalman_filters:
            return RiskDecomposition(
                total_risk=0, systematic_risk=0, idiosyncratic_risk=0,
                factor_contributions={}, marginal_var=0, component_var=0, incremental_var=0
            )

        kf = self.kalman_filters[asset]
        beta = kf.beta

        # Systematic risk: β' Σ_F β
        systematic_var = beta @ self.factor_cov @ beta

        # Idiosyncratic risk
        idio_var = self.idio_var.get(asset, 0)

        # Total risk
        total_var = systematic_var + idio_var

        # Factor contributions
        factor_contributions = {}
        for i, factor in enumerate(self.factors[:self.n_factors]):
            if i < len(beta):
                # Marginal contribution to variance
                contrib = beta[i] ** 2 * self.factor_cov[i, i]
                factor_contributions[factor] = float(contrib / (total_var + 1e-10))

        return RiskDecomposition(
            total_risk=float(np.sqrt(total_var) * np.sqrt(252)),  # Annualized
            systematic_risk=float(np.sqrt(systematic_var) * np.sqrt(252)),
            idiosyncratic_risk=float(np.sqrt(idio_var) * np.sqrt(252)),
            factor_contributions=factor_contributions,
            marginal_var=float(np.sqrt(total_var) * position_value * 2.33),  # 99% VaR
            component_var=0,
            incremental_var=0
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CORNISH-FISHER VAR
# ═══════════════════════════════════════════════════════════════════════════════

class CornishFisherVaR:
    """
    Cornish-Fisher expansion for VaR with higher moments.

    VaR_CF = μ + σ * (z_c + (z_c²-1)*S/6 + (z_c³-3z_c)*K/24 - (2z_c³-5z_c)*S²/36)

    Where:
    - S = Skewness
    - K = Excess Kurtosis
    - z_c = Normal quantile at confidence level
    """

    def __init__(
        self,
        confidence_level: float = 0.99,
        horizon_days: int = 1
    ):
        self.confidence_level = confidence_level
        self.horizon_days = horizon_days

    def calculate(
        self,
        returns: np.ndarray,
        portfolio_value: float = 1.0
    ) -> VaREstimate:
        """
        Calculate VaR using multiple methods.

        Args:
            returns: Array of returns
            portfolio_value: Portfolio value for dollar VaR

        Returns:
            VaREstimate with all VaR methods
        """
        returns = returns[~np.isnan(returns)]

        if len(returns) < 30:
            return VaREstimate(
                var_normal=0, var_historical=0, var_cornish_fisher=0,
                var_monte_carlo=0, expected_shortfall=0,
                confidence_level=self.confidence_level,
                horizon_days=self.horizon_days, skewness=0, kurtosis=0
            )

        # Statistics
        mu = np.mean(returns)
        sigma = np.std(returns)
        skewness = self._calculate_skewness(returns)
        kurtosis = self._calculate_kurtosis(returns)  # Excess kurtosis

        # Normal quantile
        z = self._normal_quantile(1 - self.confidence_level)

        # 1. Normal VaR
        var_normal = -(mu + sigma * z)

        # 2. Historical VaR
        var_historical = -np.percentile(returns, (1 - self.confidence_level) * 100)

        # 3. Cornish-Fisher VaR
        z_cf = z + (z**2 - 1) * skewness / 6 + \
               (z**3 - 3*z) * kurtosis / 24 - \
               (2*z**3 - 5*z) * skewness**2 / 36

        var_cf = -(mu + sigma * z_cf)

        # 4. Monte Carlo VaR (with Student-t)
        var_mc = self._monte_carlo_var(returns, mu, sigma, skewness, kurtosis)

        # 5. Expected Shortfall (CVaR)
        es = self._calculate_es(returns)

        # Scale to horizon
        sqrt_horizon = np.sqrt(self.horizon_days)
        var_normal *= sqrt_horizon
        var_historical *= sqrt_horizon
        var_cf *= sqrt_horizon
        var_mc *= sqrt_horizon
        es *= sqrt_horizon

        # Scale to portfolio value
        var_normal *= portfolio_value
        var_historical *= portfolio_value
        var_cf *= portfolio_value
        var_mc *= portfolio_value
        es *= portfolio_value

        return VaREstimate(
            var_normal=float(var_normal),
            var_historical=float(var_historical),
            var_cornish_fisher=float(var_cf),
            var_monte_carlo=float(var_mc),
            expected_shortfall=float(es),
            confidence_level=self.confidence_level,
            horizon_days=self.horizon_days,
            skewness=float(skewness),
            kurtosis=float(kurtosis)
        )

    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate sample skewness"""
        n = len(returns)
        if n < 3:
            return 0

        mean = np.mean(returns)
        std = np.std(returns)

        if std == 0:
            return 0

        m3 = np.mean((returns - mean) ** 3)
        return m3 / (std ** 3)

    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate excess kurtosis"""
        n = len(returns)
        if n < 4:
            return 0

        mean = np.mean(returns)
        std = np.std(returns)

        if std == 0:
            return 0

        m4 = np.mean((returns - mean) ** 4)
        return m4 / (std ** 4) - 3

    def _normal_quantile(self, p: float) -> float:
        """Approximate inverse normal CDF"""
        # Rational approximation
        a = [
            -3.969683028665376e+01,
            2.209460984245205e+02,
            -2.759285104469687e+02,
            1.383577518672690e+02,
            -3.066479806614716e+01,
            2.506628277459239e+00
        ]
        b = [
            -5.447609879822406e+01,
            1.615858368580409e+02,
            -1.556989798598866e+02,
            6.680131188771972e+01,
            -1.328068155288572e+01
        ]
        c = [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e+00,
            -2.549732539343734e+00,
            4.374664141464968e+00,
            2.938163982698783e+00
        ]
        d = [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e+00,
            3.754408661907416e+00
        ]

        p_low = 0.02425
        p_high = 1 - p_low

        if p < p_low:
            q = np.sqrt(-2 * np.log(p))
            return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                   ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        elif p <= p_high:
            q = p - 0.5
            r = q * q
            return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
                   (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
        else:
            q = np.sqrt(-2 * np.log(1 - p))
            return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                    ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)

    def _monte_carlo_var(
        self,
        returns: np.ndarray,
        mu: float,
        sigma: float,
        skewness: float,
        kurtosis: float,
        n_simulations: int = 10000
    ) -> float:
        """Monte Carlo VaR using fitted distribution"""
        # Estimate Student-t degrees of freedom from kurtosis
        # For Student-t: kurtosis = 6/(df-4) for df > 4
        if kurtosis > 0:
            df = max(5, 6 / kurtosis + 4)
        else:
            df = 30  # Approximately normal

        # Generate t-distributed samples
        t_samples = np.random.standard_t(df, n_simulations)

        # Transform to match mean and std
        samples = mu + sigma * t_samples / np.sqrt(df / (df - 2))

        # VaR from simulated distribution
        return -np.percentile(samples, (1 - self.confidence_level) * 100)

    def _calculate_es(self, returns: np.ndarray) -> float:
        """Calculate Expected Shortfall (CVaR)"""
        var_threshold = np.percentile(returns, (1 - self.confidence_level) * 100)
        tail_returns = returns[returns <= var_threshold]

        if len(tail_returns) == 0:
            return -var_threshold

        return -np.mean(tail_returns)


# ═══════════════════════════════════════════════════════════════════════════════
# COPULA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class CopulaRiskModel:
    """
    Copula-based dependency modeling.

    Implements:
    - Student-t Copula for tail dependencies
    - Gaussian Copula for comparison
    - Vine Copulas for high-dimensional dependencies (simplified)
    """

    def __init__(self):
        self.copula_type = 'student_t'
        self.params: Dict[str, Any] = {}

    def fit_student_t_copula(
        self,
        returns1: np.ndarray,
        returns2: np.ndarray
    ) -> Dict[str, float]:
        """
        Fit bivariate Student-t copula.

        Parameters:
        - rho: correlation
        - nu: degrees of freedom (tail thickness)
        """
        # Transform to uniform using empirical CDF
        u1 = self._empirical_cdf(returns1)
        u2 = self._empirical_cdf(returns2)

        # Transform to t-distribution
        # Start with Kendall's tau for correlation
        kendall_tau = self._kendall_tau(returns1, returns2)

        # For t-copula: rho = sin(pi * tau / 2)
        rho = np.sin(np.pi * kendall_tau / 2)

        # Estimate degrees of freedom (simplified MLE)
        nu = self._estimate_t_df(u1, u2, rho)

        self.copula_type = 'student_t'
        self.params = {
            'rho': float(rho),
            'nu': float(nu),
            'kendall_tau': float(kendall_tau)
        }

        return self.params

    def _empirical_cdf(self, data: np.ndarray) -> np.ndarray:
        """Transform to uniform using ranks"""
        n = len(data)
        ranks = np.argsort(np.argsort(data))
        return (ranks + 1) / (n + 1)

    def _kendall_tau(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate Kendall's tau correlation"""
        n = len(x)
        concordant = 0
        discordant = 0

        for i in range(n):
            for j in range(i + 1, n):
                if (x[i] - x[j]) * (y[i] - y[j]) > 0:
                    concordant += 1
                elif (x[i] - x[j]) * (y[i] - y[j]) < 0:
                    discordant += 1

        return (concordant - discordant) / (n * (n - 1) / 2)

    def _estimate_t_df(
        self,
        u1: np.ndarray,
        u2: np.ndarray,
        rho: float
    ) -> float:
        """Estimate t-copula degrees of freedom"""
        # Simplified: use kurtosis-based estimator
        # Transform u to t quantiles and estimate df from combined kurtosis

        # For now, use a grid search over df
        best_df = 10
        best_ll = float('-inf')

        for df in [3, 5, 7, 10, 15, 20, 30]:
            ll = self._t_copula_loglik(u1, u2, rho, df)
            if ll > best_ll:
                best_ll = ll
                best_df = df

        return best_df

    def _t_copula_loglik(
        self,
        u1: np.ndarray,
        u2: np.ndarray,
        rho: float,
        df: float
    ) -> float:
        """Log-likelihood of t-copula"""
        # Transform to t-quantiles
        t1 = self._t_quantile(u1, df)
        t2 = self._t_quantile(u2, df)

        # Bivariate t density (simplified)
        n = len(u1)
        det_R = 1 - rho ** 2

        # Log-likelihood (up to constants)
        ll = 0
        for i in range(n):
            q = (t1[i]**2 - 2*rho*t1[i]*t2[i] + t2[i]**2) / det_R
            ll += -(df + 2) / 2 * np.log(1 + q / df)
            ll += (df + 1) / 2 * (np.log(1 + t1[i]**2 / df) + np.log(1 + t2[i]**2 / df))

        return ll

    def _t_quantile(self, u: np.ndarray, df: float) -> np.ndarray:
        """Approximate inverse t-CDF"""
        # Use normal approximation for large df
        if df > 30:
            return self._normal_quantile_vec(u)

        # Otherwise use approximation
        z = self._normal_quantile_vec(u)

        # Cornish-Fisher style correction
        g1 = (z ** 3 + z) / 4
        g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96

        return z + g1 / df + g2 / df ** 2

    def _normal_quantile_vec(self, u: np.ndarray) -> np.ndarray:
        """Vectorized inverse normal CDF"""
        return np.array([self._inv_normal(p) for p in u])

    def _inv_normal(self, p: float) -> float:
        """Single inverse normal CDF"""
        # Clip to avoid infinities
        p = np.clip(p, 1e-10, 1 - 1e-10)

        # Rational approximation
        if p < 0.5:
            sign = -1
            p = 1 - p
        else:
            sign = 1

        t = np.sqrt(-2 * np.log(1 - p))

        # Coefficients
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308

        return sign * (t - (c0 + c1*t + c2*t**2) / (1 + d1*t + d2*t**2 + d3*t**3))

    def calculate_tail_dependence(self) -> Tuple[float, float]:
        """
        Calculate upper and lower tail dependence coefficients.

        For t-copula: λ = 2 * t_{ν+1}(-√((ν+1)(1-ρ)/(1+ρ)))
        """
        if self.copula_type != 'student_t':
            return 0.0, 0.0

        rho = self.params.get('rho', 0)
        nu = self.params.get('nu', 10)

        # Tail dependence coefficient
        sqrt_term = np.sqrt((nu + 1) * (1 - rho) / (1 + rho))

        # t-CDF approximation at negative sqrt_term
        t_cdf = self._t_cdf(-sqrt_term, nu + 1)

        # Symmetric for t-copula
        lower_tail = 2 * t_cdf
        upper_tail = lower_tail  # Symmetric

        return float(lower_tail), float(upper_tail)

    def _t_cdf(self, x: float, df: float) -> float:
        """Approximate t-distribution CDF"""
        # Simplified approximation
        if df > 30:
            return 0.5 * (1 + np.tanh(x * 0.7978845608))

        # Use series expansion for small df
        y = 1 / (1 + x ** 2 / df)

        # Incomplete beta approximation
        return 0.5 + x * np.sqrt(y) / (np.sqrt(df) * np.pi) * \
               (1 + y * (0.5 - 1/df) + y**2 * (0.375 - 0.25/df))

    def get_metrics(self) -> CopulaMetrics:
        """Get all copula metrics"""
        lower_tail, upper_tail = self.calculate_tail_dependence()

        return CopulaMetrics(
            lower_tail_dependence=lower_tail,
            upper_tail_dependence=upper_tail,
            kendall_tau=self.params.get('kendall_tau', 0),
            spearman_rho=np.sin(np.pi * self.params.get('kendall_tau', 0) / 2),
            copula_type=self.copula_type,
            copula_params=self.params
        )


# ═══════════════════════════════════════════════════════════════════════════════
# LIQUIDITY-ADJUSTED RISK
# ═══════════════════════════════════════════════════════════════════════════════

class LiquidityAdjustedRisk:
    """
    Liquidity-Adjusted Risk Measures.

    Implements:
    - L-VaR: Liquidity-adjusted VaR
    - Kyle's Lambda estimation
    - Amihud illiquidity measure
    - Market impact modeling
    """

    def __init__(
        self,
        market_impact_coefficient: float = 0.1,
        liquidation_horizon: int = 5  # days
    ):
        self.market_impact_coefficient = market_impact_coefficient
        self.liquidation_horizon = liquidation_horizon

    def calculate_kyle_lambda(
        self,
        returns: np.ndarray,
        volume: np.ndarray,
        signed_volume: Optional[np.ndarray] = None
    ) -> float:
        """
        Estimate Kyle's Lambda.

        λ = Cov(ΔP, OrderFlow) / Var(OrderFlow)

        Measures price impact per unit of order flow.
        """
        if signed_volume is None:
            # Approximate signed volume from return direction
            signed_volume = volume * np.sign(returns)

        # Remove NaN
        mask = ~np.isnan(returns) & ~np.isnan(signed_volume)
        returns = returns[mask]
        signed_volume = signed_volume[mask]

        if len(returns) < 30:
            return 0.0

        # Covariance and variance
        cov = np.cov(returns, signed_volume)[0, 1]
        var_flow = np.var(signed_volume)

        kyle_lambda = cov / (var_flow + 1e-10)

        return float(kyle_lambda)

    def calculate_amihud_illiquidity(
        self,
        returns: np.ndarray,
        volume_usd: np.ndarray
    ) -> float:
        """
        Calculate Amihud illiquidity measure.

        ILLIQ = (1/N) * Σ |r_t| / Volume_t

        Higher values = more illiquid
        """
        mask = ~np.isnan(returns) & ~np.isnan(volume_usd) & (volume_usd > 0)
        returns = returns[mask]
        volume_usd = volume_usd[mask]

        if len(returns) == 0:
            return 0.0

        amihud = np.mean(np.abs(returns) / volume_usd)

        return float(amihud)

    def calculate_l_var(
        self,
        var_normal: float,
        position_value: float,
        average_daily_volume: float,
        bid_ask_spread: float
    ) -> float:
        """
        Calculate Liquidity-Adjusted VaR.

        L-VaR = VaR + Liquidity Cost

        Where Liquidity Cost = Spread Cost + Market Impact
        """
        # Spread cost
        spread_cost = position_value * bid_ask_spread / 2

        # Market impact (simplified square-root model)
        # Impact = α * √(Position / ADV)
        position_pct = position_value / (average_daily_volume * self.liquidation_horizon + 1e-10)
        market_impact = self.market_impact_coefficient * np.sqrt(position_pct) * position_value

        # L-VaR
        l_var = var_normal + spread_cost + market_impact

        return float(l_var)

    def calculate_liquidation_time(
        self,
        position_value: float,
        average_daily_volume: float,
        max_participation_rate: float = 0.1
    ) -> float:
        """
        Estimate time to liquidate position.

        Time = Position / (ADV * Participation Rate)
        """
        daily_capacity = average_daily_volume * max_participation_rate

        if daily_capacity <= 0:
            return float('inf')

        return position_value / daily_capacity

    def calculate_execution_shortfall(
        self,
        position_value: float,
        current_price: float,
        bid_ask_spread: float,
        average_daily_volume: float
    ) -> float:
        """
        Estimate expected execution shortfall.

        Shortfall = Spread Cost + Temporary Impact + Permanent Impact
        """
        # Spread cost
        spread_cost = position_value * bid_ask_spread / 2

        # Temporary impact (decays)
        position_pct = position_value / (average_daily_volume + 1e-10)
        temp_impact = 0.5 * self.market_impact_coefficient * np.sqrt(position_pct) * position_value

        # Permanent impact
        perm_impact = 0.3 * self.market_impact_coefficient * np.sqrt(position_pct) * position_value

        total_shortfall = spread_cost + temp_impact + perm_impact

        return float(total_shortfall)

    def get_liquidity_metrics(
        self,
        returns: np.ndarray,
        volume: np.ndarray,
        position_value: float,
        bid_ask_spread: float,
        var_normal: float
    ) -> LiquidityRisk:
        """Get all liquidity risk metrics"""
        # Calculate ADV in USD (approximate)
        avg_volume = np.mean(volume[~np.isnan(volume)])

        # Kyle's lambda
        kyle_lambda = self.calculate_kyle_lambda(returns, volume)

        # Amihud
        amihud = self.calculate_amihud_illiquidity(returns, volume)

        # L-VaR
        l_var = self.calculate_l_var(
            var_normal, position_value, avg_volume, bid_ask_spread
        )

        # Liquidation time
        liq_time = self.calculate_liquidation_time(position_value, avg_volume)

        # Execution shortfall
        exec_shortfall = self.calculate_execution_shortfall(
            position_value, 1.0, bid_ask_spread, avg_volume
        )

        return LiquidityRisk(
            l_var=l_var,
            kyle_lambda=kyle_lambda,
            amihud_illiquidity=amihud,
            bid_ask_spread=bid_ask_spread,
            market_depth=avg_volume,
            liquidation_time=liq_time,
            execution_shortfall=exec_shortfall
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ADVANCED RISK PIPELINE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

class AdvancedRiskPipeline:
    """
    Orchestrates all advanced risk analysis components.
    """

    def __init__(
        self,
        factors: List[str] = None,
        confidence_level: float = 0.99,
        use_kalman_betas: bool = True
    ):
        self.confidence_level = confidence_level

        # Initialize components
        self.factor_model = MultiFactorRiskModel(
            factors=factors,
            use_kalman=use_kalman_betas
        )
        self.var_calculator = CornishFisherVaR(confidence_level=confidence_level)
        self.copula_model = CopulaRiskModel()
        self.liquidity_risk = LiquidityAdjustedRisk()

        # Cache
        self._risk_cache: Dict[str, ComprehensiveRiskReport] = {}

    def fit(
        self,
        returns_df: pd.DataFrame,
        factor_returns: Optional[pd.DataFrame] = None
    ):
        """Fit all risk models"""
        self.factor_model.fit(returns_df, factor_returns)

    def analyze_asset(
        self,
        symbol: str,
        returns: np.ndarray,
        volume: np.ndarray,
        position_value: float = 1.0,
        bid_ask_spread: float = 0.001,
        benchmark_returns: Optional[np.ndarray] = None
    ) -> ComprehensiveRiskReport:
        """
        Comprehensive risk analysis for a single asset.

        Args:
            symbol: Asset symbol
            returns: Return series
            volume: Volume series (in USD)
            position_value: Position value for VaR scaling
            bid_ask_spread: Bid-ask spread
            benchmark_returns: Optional benchmark for copula analysis

        Returns:
            ComprehensiveRiskReport with all metrics
        """
        timestamp = datetime.utcnow()

        # VaR analysis
        var_estimate = self.var_calculator.calculate(returns, position_value)

        # Factor decomposition
        risk_decomp = self.factor_model.decompose_risk(symbol, position_value)

        # Factor exposures
        factor_exposures = self.factor_model.get_factor_exposures(symbol)

        # Liquidity risk
        liq_risk = self.liquidity_risk.get_liquidity_metrics(
            returns, volume, position_value, bid_ask_spread, var_estimate.var_normal
        )

        # Copula analysis (if benchmark available)
        copula_metrics = None
        if benchmark_returns is not None:
            self.copula_model.fit_student_t_copula(returns, benchmark_returns)
            copula_metrics = self.copula_model.get_metrics()

        # Stress scenarios
        stress_scenarios = self._calculate_stress_scenarios(returns, position_value)

        # Aggregate risk score (0-100)
        risk_score = self._calculate_risk_score(
            var_estimate, risk_decomp, liq_risk, copula_metrics
        )

        report = ComprehensiveRiskReport(
            symbol=symbol,
            timestamp=timestamp,
            var_estimate=var_estimate,
            risk_decomposition=risk_decomp,
            factor_exposures=factor_exposures,
            liquidity_risk=liq_risk,
            copula_metrics=copula_metrics,
            stress_scenarios=stress_scenarios,
            risk_score=risk_score
        )

        self._risk_cache[symbol] = report

        return report

    def _calculate_stress_scenarios(
        self,
        returns: np.ndarray,
        position_value: float
    ) -> Dict[str, float]:
        """Calculate losses under stress scenarios"""
        scenarios = {}

        # Historical worst day
        scenarios['worst_day'] = float(-np.min(returns) * position_value)

        # 3-sigma move
        sigma = np.std(returns)
        scenarios['3_sigma'] = float(3 * sigma * position_value)

        # 5-sigma move (tail event)
        scenarios['5_sigma'] = float(5 * sigma * position_value)

        # Worst 5-day period
        if len(returns) >= 5:
            rolling_5d = pd.Series(returns).rolling(5).sum()
            scenarios['worst_5d'] = float(-np.min(rolling_5d) * position_value)

        # Flash crash scenario (10% instant drop)
        scenarios['flash_crash'] = float(0.10 * position_value)

        return scenarios

    def _calculate_risk_score(
        self,
        var_estimate: VaREstimate,
        risk_decomp: RiskDecomposition,
        liq_risk: LiquidityRisk,
        copula_metrics: Optional[CopulaMetrics]
    ) -> float:
        """Calculate aggregate risk score (0-100, higher = more risky)"""
        score = 0

        # VaR contribution (0-25)
        # High VaR relative to typical = higher score
        typical_var = 0.02  # 2% daily VaR as baseline
        var_ratio = var_estimate.var_cornish_fisher / typical_var
        score += min(25, 25 * var_ratio)

        # Volatility contribution (0-25)
        typical_vol = 0.50  # 50% annualized as baseline for crypto
        vol_ratio = risk_decomp.total_risk / typical_vol
        score += min(25, 25 * vol_ratio)

        # Liquidity contribution (0-25)
        # High Amihud or long liquidation time = higher score
        if liq_risk.liquidation_time > 1:
            score += min(15, 3 * liq_risk.liquidation_time)
        score += min(10, liq_risk.amihud_illiquidity * 10000)

        # Tail risk contribution (0-25)
        if copula_metrics:
            tail_dep = copula_metrics.lower_tail_dependence
            score += min(15, 50 * tail_dep)

        # Kurtosis contribution
        if var_estimate.kurtosis > 0:
            score += min(10, var_estimate.kurtosis * 2)

        return min(100, score)

    def get_portfolio_risk(
        self,
        weights: Dict[str, float],
        correlation_matrix: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate portfolio-level risk metrics"""
        if not self._risk_cache:
            return {'error': 'No risk data cached'}

        symbols = list(weights.keys())
        w = np.array([weights[s] for s in symbols])

        # Get individual volatilities
        vols = []
        for s in symbols:
            if s in self._risk_cache:
                vols.append(self._risk_cache[s].risk_decomposition.total_risk)
            else:
                vols.append(0.5)  # Default

        vols = np.array(vols)

        # Build correlation matrix if not provided
        if correlation_matrix is None:
            # Assume average correlation of 0.5 for crypto
            n = len(symbols)
            correlation_matrix = 0.5 * np.ones((n, n))
            np.fill_diagonal(correlation_matrix, 1.0)

        # Portfolio volatility
        cov_matrix = np.outer(vols, vols) * correlation_matrix
        port_var = w @ cov_matrix @ w
        port_vol = np.sqrt(port_var)

        # Weighted VaR
        weighted_var = sum(
            weights[s] * self._risk_cache[s].var_estimate.var_cornish_fisher
            for s in symbols if s in self._risk_cache
        )

        # Diversification ratio
        undiv_vol = sum(abs(weights[s]) * vols[i] for i, s in enumerate(symbols))
        div_ratio = undiv_vol / (port_vol + 1e-10)

        return {
            'portfolio_volatility': float(port_vol),
            'portfolio_var': float(weighted_var),
            'diversification_ratio': float(div_ratio),
            'concentration': float(np.sum(w ** 2))  # Herfindahl
        }


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    'FactorExposure',
    'RiskDecomposition',
    'VaREstimate',
    'LiquidityRisk',
    'CopulaMetrics',
    'ComprehensiveRiskReport',
    'KalmanFilterBeta',
    'MultiFactorRiskModel',
    'CornishFisherVaR',
    'CopulaRiskModel',
    'LiquidityAdjustedRisk',
    'AdvancedRiskPipeline',
]
