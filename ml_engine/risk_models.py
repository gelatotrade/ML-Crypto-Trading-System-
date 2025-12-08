"""
Risk Models - VaR, CVaR, Volatility Estimation, GARCH, Beta Calculation
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class RiskModels:
    """
    Comprehensive risk modeling including:
    - VaR and CVaR (Historical, Parametric, Cornish-Fisher)
    - Volatility estimation (EWMA, GARCH, Realized)
    - Correlation and covariance matrices with shrinkage
    - Beta calculation
    - Max drawdown analysis
    """

    def __init__(self, config=None):
        """Initialize risk models"""
        self.config = config
        self.risk_free_rate = 0.05 if config is None else config.options.risk_free_rate

    # ===================== VaR Methods =====================

    def calculate_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
        method: str = 'historical',
        horizon: int = 1
    ) -> float:
        """
        Calculate Value at Risk.

        Args:
            returns: Series of returns
            confidence: Confidence level (e.g., 0.95 for 95% VaR)
            method: 'historical', 'parametric', or 'cornish_fisher'
            horizon: Time horizon in periods

        Returns:
            VaR value (positive number representing potential loss)
        """
        returns = returns.dropna()

        if method == 'historical':
            return self._var_historical(returns, confidence, horizon)
        elif method == 'parametric':
            return self._var_parametric(returns, confidence, horizon)
        elif method == 'cornish_fisher':
            return self._var_cornish_fisher(returns, confidence, horizon)
        else:
            raise ValueError(f"Unknown VaR method: {method}")

    def _var_historical(self, returns: pd.Series, confidence: float, horizon: int) -> float:
        """Historical VaR"""
        alpha = 1 - confidence
        var = -np.percentile(returns, alpha * 100)
        return var * np.sqrt(horizon)

    def _var_parametric(self, returns: pd.Series, confidence: float, horizon: int) -> float:
        """Parametric (Normal) VaR"""
        mu = returns.mean()
        sigma = returns.std()
        z_score = stats.norm.ppf(1 - confidence)
        var = -(mu + z_score * sigma)
        return var * np.sqrt(horizon)

    def _var_cornish_fisher(self, returns: pd.Series, confidence: float, horizon: int) -> float:
        """
        Cornish-Fisher VaR - adjusts for skewness and kurtosis
        """
        mu = returns.mean()
        sigma = returns.std()
        skew = returns.skew()
        kurt = returns.kurtosis()

        z = stats.norm.ppf(1 - confidence)

        # Cornish-Fisher expansion
        z_cf = (z +
                (z**2 - 1) * skew / 6 +
                (z**3 - 3*z) * (kurt - 3) / 24 -
                (2*z**3 - 5*z) * skew**2 / 36)

        var = -(mu + z_cf * sigma)
        return var * np.sqrt(horizon)

    def calculate_cvar(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
        method: str = 'historical'
    ) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).

        Args:
            returns: Series of returns
            confidence: Confidence level
            method: 'historical' or 'parametric'

        Returns:
            CVaR value
        """
        returns = returns.dropna()

        if method == 'historical':
            alpha = 1 - confidence
            cutoff = np.percentile(returns, alpha * 100)
            return -returns[returns <= cutoff].mean()
        elif method == 'parametric':
            mu = returns.mean()
            sigma = returns.std()
            z = stats.norm.ppf(1 - confidence)
            cvar = -(mu + sigma * stats.norm.pdf(z) / (1 - confidence))
            return cvar
        else:
            raise ValueError(f"Unknown CVaR method: {method}")

    # ===================== Volatility Methods =====================

    def calculate_volatility(
        self,
        returns: pd.Series,
        method: str = 'standard',
        window: int = 20,
        annualize: bool = True,
        periods_per_year: int = 252 * 24  # Hourly data
    ) -> pd.Series:
        """
        Calculate volatility using various methods.

        Args:
            returns: Series of returns
            method: 'standard', 'ewma', 'garch', 'realized', 'parkinson', 'garman_klass'
            window: Rolling window size
            annualize: Whether to annualize volatility
            periods_per_year: Number of periods per year

        Returns:
            Series of volatility estimates
        """
        if method == 'standard':
            vol = returns.rolling(window).std()
        elif method == 'ewma':
            vol = self._volatility_ewma(returns, window)
        elif method == 'garch':
            vol = self._volatility_garch(returns)
        elif method == 'realized':
            vol = self._volatility_realized(returns, window)
        else:
            vol = returns.rolling(window).std()

        if annualize:
            vol = vol * np.sqrt(periods_per_year)

        return vol

    def _volatility_ewma(self, returns: pd.Series, span: int = 20) -> pd.Series:
        """Exponentially Weighted Moving Average volatility"""
        return returns.ewm(span=span, adjust=False).std()

    def _volatility_garch(self, returns: pd.Series, p: int = 1, q: int = 1) -> pd.Series:
        """
        GARCH(p,q) volatility estimation.
        Returns conditional volatility series.
        """
        try:
            from arch import arch_model

            # Fit GARCH model
            returns_clean = returns.dropna() * 100  # Scale for numerical stability

            model = arch_model(
                returns_clean,
                vol='Garch',
                p=p,
                q=q,
                mean='Constant',
                rescale=False
            )

            result = model.fit(disp='off', show_warning=False)

            # Get conditional volatility
            cond_vol = result.conditional_volatility / 100

            # Align with original index
            vol_series = pd.Series(index=returns.index, dtype=float)
            vol_series.loc[cond_vol.index] = cond_vol.values

            return vol_series.fillna(method='ffill')

        except ImportError:
            logger.warning("arch package not installed, falling back to EWMA")
            return self._volatility_ewma(returns)
        except Exception as e:
            logger.warning(f"GARCH fitting failed: {e}, falling back to EWMA")
            return self._volatility_ewma(returns)

    def _volatility_realized(self, returns: pd.Series, window: int = 20) -> pd.Series:
        """Realized volatility using sum of squared returns"""
        return np.sqrt((returns**2).rolling(window).sum())

    def estimate_volatility_parkinson(
        self,
        df: pd.DataFrame,
        window: int = 20,
        annualize: bool = True
    ) -> pd.Series:
        """
        Parkinson volatility estimator using high-low range.
        More efficient than close-to-close for same number of observations.
        """
        hl_ratio = np.log(df['high'] / df['low'])
        parkinson_factor = 1 / (4 * np.log(2))
        vol = np.sqrt(parkinson_factor * (hl_ratio**2).rolling(window).mean())

        if annualize:
            vol = vol * np.sqrt(252 * 24)

        return vol

    def estimate_volatility_garman_klass(
        self,
        df: pd.DataFrame,
        window: int = 20,
        annualize: bool = True
    ) -> pd.Series:
        """
        Garman-Klass volatility estimator.
        Uses OHLC data for more efficient estimation.
        """
        log_hl = np.log(df['high'] / df['low'])**2
        log_co = np.log(df['close'] / df['open'])**2

        gk_var = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        vol = np.sqrt(gk_var.rolling(window).mean())

        if annualize:
            vol = vol * np.sqrt(252 * 24)

        return vol

    # ===================== Correlation & Covariance =====================

    def calculate_correlation_matrix(
        self,
        returns_df: pd.DataFrame,
        method: str = 'pearson'
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix.

        Args:
            returns_df: DataFrame with returns for multiple assets
            method: 'pearson', 'spearman', or 'kendall'

        Returns:
            Correlation matrix
        """
        return returns_df.corr(method=method)

    def calculate_covariance_matrix(
        self,
        returns_df: pd.DataFrame,
        method: str = 'standard',
        shrinkage_target: str = 'identity'
    ) -> pd.DataFrame:
        """
        Calculate covariance matrix with optional shrinkage.

        Args:
            returns_df: DataFrame with returns for multiple assets
            method: 'standard', 'shrinkage', 'ewma'
            shrinkage_target: 'identity', 'constant_correlation', 'single_factor'

        Returns:
            Covariance matrix
        """
        if method == 'standard':
            return returns_df.cov()
        elif method == 'shrinkage':
            return self._covariance_shrinkage(returns_df, shrinkage_target)
        elif method == 'ewma':
            return self._covariance_ewma(returns_df)
        else:
            return returns_df.cov()

    def _covariance_shrinkage(
        self,
        returns_df: pd.DataFrame,
        target: str = 'identity'
    ) -> pd.DataFrame:
        """
        Ledoit-Wolf shrinkage estimator for covariance matrix.
        """
        sample_cov = returns_df.cov()
        n_assets = len(sample_cov)

        # Calculate target matrix
        if target == 'identity':
            avg_var = np.diag(sample_cov).mean()
            target_matrix = np.eye(n_assets) * avg_var
        elif target == 'constant_correlation':
            std_devs = np.sqrt(np.diag(sample_cov))
            avg_corr = (sample_cov.values / np.outer(std_devs, std_devs)).mean()
            target_matrix = avg_corr * np.outer(std_devs, std_devs)
            np.fill_diagonal(target_matrix, np.diag(sample_cov))
        else:
            target_matrix = np.eye(n_assets) * np.diag(sample_cov).mean()

        # Calculate optimal shrinkage intensity (simplified Ledoit-Wolf)
        n_samples = len(returns_df)
        delta = ((sample_cov.values - target_matrix)**2).sum() / n_assets**2
        shrinkage_intensity = min(1, max(0, (1/n_samples) / (delta + 1e-10)))

        # Apply shrinkage
        shrunk_cov = (1 - shrinkage_intensity) * sample_cov.values + shrinkage_intensity * target_matrix

        return pd.DataFrame(shrunk_cov, index=sample_cov.index, columns=sample_cov.columns)

    def _covariance_ewma(self, returns_df: pd.DataFrame, span: int = 60) -> pd.DataFrame:
        """EWMA covariance matrix"""
        return returns_df.ewm(span=span).cov().iloc[-len(returns_df.columns):]

    # ===================== Beta Calculation =====================

    def calculate_beta(
        self,
        asset_returns: pd.Series,
        market_returns: pd.Series,
        method: str = 'ols',
        window: Optional[int] = None
    ) -> float:
        """
        Calculate beta of an asset relative to market.

        Args:
            asset_returns: Asset return series
            market_returns: Market (benchmark) return series
            method: 'ols' or 'rolling'
            window: Rolling window for rolling beta

        Returns:
            Beta value
        """
        # Align indices
        aligned = pd.DataFrame({
            'asset': asset_returns,
            'market': market_returns
        }).dropna()

        if len(aligned) < 2:
            return 1.0

        if method == 'ols':
            covariance = aligned['asset'].cov(aligned['market'])
            market_variance = aligned['market'].var()
            return covariance / market_variance if market_variance > 0 else 1.0
        elif method == 'rolling' and window:
            rolling_cov = aligned['asset'].rolling(window).cov(aligned['market'])
            rolling_var = aligned['market'].rolling(window).var()
            return (rolling_cov / rolling_var).iloc[-1]
        else:
            return self.calculate_beta(asset_returns, market_returns, method='ols')

    def calculate_portfolio_beta(
        self,
        weights: Dict[str, float],
        betas: Dict[str, float]
    ) -> float:
        """
        Calculate portfolio beta from asset weights and betas.

        Args:
            weights: Dictionary of asset weights
            betas: Dictionary of asset betas

        Returns:
            Portfolio beta
        """
        portfolio_beta = 0.0
        for asset, weight in weights.items():
            if asset in betas:
                portfolio_beta += weight * betas[asset]
        return portfolio_beta

    # ===================== Drawdown Analysis =====================

    def calculate_max_drawdown(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Calculate maximum drawdown and related metrics.

        Args:
            returns: Series of returns

        Returns:
            Dictionary with drawdown metrics
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        max_dd = drawdown.min()
        max_dd_idx = drawdown.idxmin()

        # Find peak and trough
        peak_idx = cumulative[:max_dd_idx].idxmax() if max_dd_idx else None
        trough_idx = max_dd_idx

        # Recovery (if any)
        if max_dd_idx and max_dd_idx in cumulative.index:
            recovery_data = cumulative[cumulative.index > max_dd_idx]
            if len(recovery_data) > 0:
                recovery_level = cumulative[peak_idx] if peak_idx else cumulative.iloc[0]
                recovered_idx = recovery_data[recovery_data >= recovery_level].index
                recovery_idx = recovered_idx[0] if len(recovered_idx) > 0 else None
            else:
                recovery_idx = None
        else:
            recovery_idx = None

        return {
            'max_drawdown': max_dd,
            'drawdown_series': drawdown,
            'peak_date': peak_idx,
            'trough_date': trough_idx,
            'recovery_date': recovery_idx,
            'current_drawdown': drawdown.iloc[-1] if len(drawdown) > 0 else 0,
        }

    def calculate_calmar_ratio(self, returns: pd.Series, periods_per_year: int = 252 * 24) -> float:
        """
        Calculate Calmar ratio (annualized return / max drawdown).

        Args:
            returns: Series of returns
            periods_per_year: Number of periods per year

        Returns:
            Calmar ratio
        """
        ann_return = returns.mean() * periods_per_year
        dd_info = self.calculate_max_drawdown(returns)
        max_dd = abs(dd_info['max_drawdown'])
        return ann_return / max_dd if max_dd > 0 else 0

    # ===================== Risk Metrics Summary =====================

    def calculate_risk_metrics(
        self,
        returns: pd.Series,
        market_returns: Optional[pd.Series] = None,
        confidence: float = 0.95
    ) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics.

        Args:
            returns: Series of returns
            market_returns: Optional market returns for beta calculation
            confidence: Confidence level for VaR/CVaR

        Returns:
            Dictionary of risk metrics
        """
        returns = returns.dropna()

        metrics = {
            # Basic statistics
            'mean_return': returns.mean(),
            'std_dev': returns.std(),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),

            # VaR metrics
            'var_historical': self.calculate_var(returns, confidence, 'historical'),
            'var_parametric': self.calculate_var(returns, confidence, 'parametric'),
            'var_cornish_fisher': self.calculate_var(returns, confidence, 'cornish_fisher'),

            # CVaR
            'cvar_historical': self.calculate_cvar(returns, confidence, 'historical'),
            'cvar_parametric': self.calculate_cvar(returns, confidence, 'parametric'),

            # Volatility
            'volatility_annualized': returns.std() * np.sqrt(252 * 24),

            # Drawdown
            'max_drawdown': self.calculate_max_drawdown(returns)['max_drawdown'],

            # Performance ratios
            'sharpe_ratio': self._calculate_sharpe(returns),
            'sortino_ratio': self._calculate_sortino(returns),
            'calmar_ratio': self.calculate_calmar_ratio(returns),
        }

        # Add beta if market returns provided
        if market_returns is not None:
            metrics['beta'] = self.calculate_beta(returns, market_returns)
            metrics['alpha'] = self._calculate_alpha(returns, market_returns)

        return metrics

    def _calculate_sharpe(
        self,
        returns: pd.Series,
        risk_free_rate: Optional[float] = None,
        periods_per_year: int = 252 * 24
    ) -> float:
        """Calculate Sharpe ratio"""
        rf = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        rf_per_period = rf / periods_per_year

        excess_returns = returns - rf_per_period
        if excess_returns.std() == 0:
            return 0

        return (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year)

    def _calculate_sortino(
        self,
        returns: pd.Series,
        risk_free_rate: Optional[float] = None,
        periods_per_year: int = 252 * 24
    ) -> float:
        """Calculate Sortino ratio (uses downside deviation)"""
        rf = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        rf_per_period = rf / periods_per_year

        excess_returns = returns - rf_per_period
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0

        downside_std = np.sqrt((downside_returns**2).mean())
        return (excess_returns.mean() / downside_std) * np.sqrt(periods_per_year)

    def _calculate_alpha(
        self,
        asset_returns: pd.Series,
        market_returns: pd.Series
    ) -> float:
        """Calculate Jensen's alpha"""
        beta = self.calculate_beta(asset_returns, market_returns)
        rf_per_period = self.risk_free_rate / (252 * 24)

        expected_return = rf_per_period + beta * (market_returns.mean() - rf_per_period)
        actual_return = asset_returns.mean()

        return (actual_return - expected_return) * (252 * 24)  # Annualized

    # ===================== Position Sizing =====================

    def calculate_position_size_volatility(
        self,
        portfolio_value: float,
        target_volatility: float,
        asset_volatility: float,
        max_position: float = 0.2
    ) -> float:
        """
        Calculate position size based on volatility targeting.

        Args:
            portfolio_value: Total portfolio value
            target_volatility: Target portfolio volatility
            asset_volatility: Asset volatility
            max_position: Maximum position size as fraction

        Returns:
            Position size in currency
        """
        if asset_volatility <= 0:
            return 0

        position_pct = min(target_volatility / asset_volatility, max_position)
        return portfolio_value * position_pct

    def calculate_position_size_var(
        self,
        portfolio_value: float,
        max_var: float,
        asset_var: float,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate position size based on VaR limit.

        Args:
            portfolio_value: Total portfolio value
            max_var: Maximum VaR as fraction of portfolio
            asset_var: Asset VaR
            confidence: Confidence level

        Returns:
            Position size in currency
        """
        if asset_var <= 0:
            return portfolio_value * 0.1  # Default 10%

        position_pct = max_var / asset_var
        return portfolio_value * min(position_pct, 0.3)  # Cap at 30%
