"""
Portfolio Optimizer - Sharpe Ratio > 2.5 Optimization with Dynamic Beta Management
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Container for optimization results"""
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    beta: Optional[float] = None
    net_exposure: float = 0.0
    gross_exposure: float = 0.0
    long_exposure: float = 0.0
    short_exposure: float = 0.0
    success: bool = True
    message: str = ""


class PortfolioOptimizer:
    """
    Portfolio optimization with market neutrality and dynamic beta targeting.

    Features:
    - Mean-variance optimization (Markowitz)
    - Maximum Sharpe ratio optimization (target > 2.5)
    - Market-neutral constraints
    - Dynamic beta management based on regime
    - Risk parity allocation
    - Black-Litterman model
    """

    def __init__(
        self,
        min_sharpe_ratio: float = 2.5,
        target_beta: float = 0.0,
        max_net_exposure: float = 0.1,
        max_gross_exposure: float = 2.0,
        risk_free_rate: float = 0.05,
        max_position_size: float = 0.2,
        config=None
    ):
        """
        Initialize portfolio optimizer.

        Args:
            min_sharpe_ratio: Minimum target Sharpe ratio (default 2.5)
            target_beta: Target portfolio beta (0 for market neutral)
            max_net_exposure: Maximum net exposure (long - short)
            max_gross_exposure: Maximum gross exposure (long + short)
            risk_free_rate: Annual risk-free rate
            max_position_size: Maximum position size per asset
            config: Optional configuration object
        """
        self.min_sharpe_ratio = min_sharpe_ratio
        self.target_beta = target_beta
        self.max_net_exposure = max_net_exposure
        self.max_gross_exposure = max_gross_exposure
        self.risk_free_rate = risk_free_rate
        self.max_position_size = max_position_size
        self.config = config

        # Periods per year for annualization (hourly data)
        self.periods_per_year = 252 * 24

    def optimize_mean_variance(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        market_returns: Optional[pd.Series] = None,
        asset_betas: Optional[Dict[str, float]] = None,
        position_limits: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> OptimizationResult:
        """
        Perform mean-variance optimization with market neutrality constraints.

        Args:
            expected_returns: Expected returns for each asset (annualized)
            cov_matrix: Covariance matrix of returns
            market_returns: Market returns for beta calculation
            asset_betas: Pre-calculated asset betas
            position_limits: Dict of (min, max) position limits per asset

        Returns:
            OptimizationResult with optimal weights
        """
        assets = list(expected_returns.index)
        n_assets = len(assets)

        if n_assets == 0:
            return OptimizationResult(
                weights={},
                expected_return=0,
                volatility=0,
                sharpe_ratio=0,
                success=False,
                message="No assets provided"
            )

        # Convert to numpy arrays
        mu = expected_returns.values
        sigma = cov_matrix.values

        # Calculate betas if not provided
        if asset_betas is None and market_returns is not None:
            asset_betas = self._calculate_betas(expected_returns, cov_matrix, market_returns)

        betas = np.array([asset_betas.get(a, 1.0) for a in assets]) if asset_betas else np.ones(n_assets)

        # Initial weights (equal weight long/short)
        x0 = np.zeros(n_assets)
        x0[::2] = 0.1  # Long alternating
        x0[1::2] = -0.1  # Short alternating

        # Objective: Maximize Sharpe ratio (minimize negative Sharpe)
        def neg_sharpe(weights):
            port_return = np.dot(weights, mu)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
            if port_vol < 1e-10:
                return 1e6
            sharpe = (port_return - self.risk_free_rate) / port_vol
            return -sharpe

        # Constraints
        constraints = []

        # Net exposure constraint: |sum(weights)| <= max_net_exposure
        def net_exposure_constraint(weights):
            return self.max_net_exposure - abs(np.sum(weights))

        constraints.append({
            'type': 'ineq',
            'fun': net_exposure_constraint
        })

        # Gross exposure constraint: sum(|weights|) <= max_gross_exposure
        def gross_exposure_constraint(weights):
            return self.max_gross_exposure - np.sum(np.abs(weights))

        constraints.append({
            'type': 'ineq',
            'fun': gross_exposure_constraint
        })

        # Beta constraint: portfolio beta close to target
        if asset_betas is not None:
            def beta_constraint(weights):
                port_beta = np.dot(weights, betas)
                return 0.05 - abs(port_beta - self.target_beta)  # Within 5% of target

            constraints.append({
                'type': 'ineq',
                'fun': beta_constraint
            })

        # Position limits
        if position_limits:
            bounds = []
            for asset in assets:
                if asset in position_limits:
                    bounds.append(position_limits[asset])
                else:
                    bounds.append((-self.max_position_size, self.max_position_size))
        else:
            bounds = [(-self.max_position_size, self.max_position_size)] * n_assets

        # Optimize
        result = minimize(
            neg_sharpe,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )

        if not result.success:
            logger.warning(f"Optimization did not converge: {result.message}")

        weights = result.x

        # Calculate portfolio metrics
        port_return = np.dot(weights, mu)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
        sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0

        # Calculate exposures
        long_exp = np.sum(weights[weights > 0])
        short_exp = abs(np.sum(weights[weights < 0]))
        net_exp = np.sum(weights)
        gross_exp = long_exp + short_exp

        # Calculate portfolio beta
        port_beta = np.dot(weights, betas) if asset_betas else None

        return OptimizationResult(
            weights={asset: weights[i] for i, asset in enumerate(assets)},
            expected_return=port_return,
            volatility=port_vol,
            sharpe_ratio=sharpe,
            beta=port_beta,
            net_exposure=net_exp,
            gross_exposure=gross_exp,
            long_exposure=long_exp,
            short_exposure=short_exp,
            success=result.success,
            message=result.message if not result.success else "Optimization successful"
        )

    def optimize_max_sharpe(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        **kwargs
    ) -> OptimizationResult:
        """
        Optimize for maximum Sharpe ratio with target > 2.5.
        Alias for optimize_mean_variance with Sharpe maximization.
        """
        return self.optimize_mean_variance(expected_returns, cov_matrix, **kwargs)

    def optimize_risk_parity(
        self,
        cov_matrix: pd.DataFrame,
        target_risk: Optional[float] = None
    ) -> OptimizationResult:
        """
        Risk parity optimization - equal risk contribution from each asset.

        Args:
            cov_matrix: Covariance matrix
            target_risk: Target portfolio volatility

        Returns:
            OptimizationResult with risk parity weights
        """
        assets = list(cov_matrix.index)
        n_assets = len(assets)
        sigma = cov_matrix.values

        # Objective: minimize deviation from equal risk contribution
        def risk_parity_objective(weights):
            weights = np.abs(weights)  # Risk parity uses absolute weights
            port_var = np.dot(weights.T, np.dot(sigma, weights))
            port_vol = np.sqrt(port_var)

            # Marginal risk contribution
            marginal_contrib = np.dot(sigma, weights)

            # Risk contribution
            risk_contrib = weights * marginal_contrib / port_vol

            # Target: equal risk contribution
            target_contrib = port_vol / n_assets

            # Minimize squared deviation from target
            return np.sum((risk_contrib - target_contrib) ** 2)

        # Constraints
        constraints = []

        # Budget constraint (weights sum to 1 in absolute terms for long-only version)
        def budget_constraint(weights):
            return np.sum(np.abs(weights)) - 1.0

        constraints.append({'type': 'eq', 'fun': budget_constraint})

        # Initial guess
        x0 = np.ones(n_assets) / n_assets

        # Bounds (allow both long and short)
        bounds = [(-0.5, 0.5)] * n_assets

        result = minimize(
            risk_parity_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        weights = result.x

        # Calculate metrics
        port_vol = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
        net_exp = np.sum(weights)
        gross_exp = np.sum(np.abs(weights))

        return OptimizationResult(
            weights={asset: weights[i] for i, asset in enumerate(assets)},
            expected_return=0,  # Risk parity doesn't optimize returns
            volatility=port_vol,
            sharpe_ratio=0,
            net_exposure=net_exp,
            gross_exposure=gross_exp,
            long_exposure=np.sum(weights[weights > 0]),
            short_exposure=abs(np.sum(weights[weights < 0])),
            success=result.success,
            message="Risk parity optimization"
        )

    def optimize_black_litterman(
        self,
        market_weights: pd.Series,
        cov_matrix: pd.DataFrame,
        views: Dict[str, float],
        view_confidences: Dict[str, float],
        market_returns: Optional[pd.Series] = None,
        tau: float = 0.05
    ) -> OptimizationResult:
        """
        Black-Litterman model combining market equilibrium with investor views.

        Args:
            market_weights: Market capitalization weights
            cov_matrix: Covariance matrix
            views: Dict of {asset: expected_return_view}
            view_confidences: Dict of {asset: confidence} (0-1)
            market_returns: Historical market returns for equilibrium
            tau: Uncertainty parameter

        Returns:
            OptimizationResult with Black-Litterman weights
        """
        assets = list(cov_matrix.index)
        n_assets = len(assets)
        sigma = cov_matrix.values

        # Calculate equilibrium returns (CAPM)
        risk_aversion = 2.5
        w_mkt = market_weights.reindex(assets).fillna(1/n_assets).values
        pi = risk_aversion * np.dot(sigma, w_mkt)  # Equilibrium excess returns

        # Build view matrix P and view vector Q
        view_assets = [a for a in views.keys() if a in assets]
        if not view_assets:
            # No views - return equilibrium
            return self.optimize_mean_variance(
                pd.Series(pi, index=assets),
                cov_matrix
            )

        k = len(view_assets)
        P = np.zeros((k, n_assets))
        Q = np.zeros(k)
        omega_diag = np.zeros(k)

        for i, asset in enumerate(view_assets):
            j = assets.index(asset)
            P[i, j] = 1
            Q[i] = views[asset]
            confidence = view_confidences.get(asset, 0.5)
            omega_diag[i] = (1 - confidence) * sigma[j, j]

        omega = np.diag(omega_diag)

        # Black-Litterman formula
        tau_sigma = tau * sigma
        tau_sigma_inv = np.linalg.inv(tau_sigma)
        omega_inv = np.linalg.inv(omega + 1e-10 * np.eye(k))

        # Posterior expected returns
        M = np.linalg.inv(tau_sigma_inv + P.T @ omega_inv @ P)
        bl_returns = M @ (tau_sigma_inv @ pi + P.T @ omega_inv @ Q)

        # Optimize with BL returns
        return self.optimize_mean_variance(
            pd.Series(bl_returns, index=assets),
            cov_matrix
        )

    def optimize_with_regime(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        regime: str,
        asset_betas: Dict[str, float],
        **kwargs
    ) -> OptimizationResult:
        """
        Optimize portfolio based on market regime.

        Args:
            expected_returns: Expected returns
            cov_matrix: Covariance matrix
            regime: 'risk_on', 'risk_off', or 'neutral'
            asset_betas: Asset betas

        Returns:
            OptimizationResult with regime-adjusted weights
        """
        # Adjust target beta based on regime
        if regime == 'risk_on':
            self.target_beta = 0.3  # Long bias
        elif regime == 'risk_off':
            self.target_beta = -0.3  # Short bias
        else:
            self.target_beta = 0.0  # Market neutral

        logger.info(f"Regime: {regime}, Target Beta: {self.target_beta}")

        return self.optimize_mean_variance(
            expected_returns,
            cov_matrix,
            asset_betas=asset_betas,
            **kwargs
        )

    def adjust_for_neutrality(
        self,
        weights: Dict[str, float],
        max_net_exposure: float = 0.1
    ) -> Dict[str, float]:
        """
        Adjust weights to ensure market neutrality.

        Args:
            weights: Current portfolio weights
            max_net_exposure: Maximum allowed net exposure

        Returns:
            Adjusted weights
        """
        net_exposure = sum(weights.values())

        if abs(net_exposure) <= max_net_exposure:
            return weights

        # Calculate adjustment needed
        adjustment = net_exposure - np.sign(net_exposure) * max_net_exposure

        # Distribute adjustment
        adjusted = {}
        n_assets = len(weights)

        for asset, weight in weights.items():
            adjusted[asset] = weight - adjustment / n_assets

        return adjusted

    def _calculate_betas(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        market_returns: pd.Series
    ) -> Dict[str, float]:
        """Calculate betas from covariance matrix"""
        betas = {}
        market_var = market_returns.var()

        for asset in expected_returns.index:
            if asset in cov_matrix.index:
                # Simplified beta calculation from covariance
                asset_var = cov_matrix.loc[asset, asset]
                betas[asset] = np.sqrt(asset_var / market_var) if market_var > 0 else 1.0

        return betas

    def calculate_efficient_frontier(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        n_points: int = 100
    ) -> pd.DataFrame:
        """
        Calculate efficient frontier points.

        Args:
            expected_returns: Expected returns
            cov_matrix: Covariance matrix
            n_points: Number of points on frontier

        Returns:
            DataFrame with return, volatility, sharpe for each point
        """
        assets = list(expected_returns.index)
        n_assets = len(assets)
        mu = expected_returns.values
        sigma = cov_matrix.values

        # Find min and max feasible returns
        min_ret = mu.min()
        max_ret = mu.max()
        target_returns = np.linspace(min_ret, max_ret, n_points)

        frontier = []

        for target in target_returns:
            # Minimize variance for target return
            def variance(weights):
                return np.dot(weights.T, np.dot(sigma, weights))

            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
                {'type': 'eq', 'fun': lambda w, t=target: np.dot(w, mu) - t}  # Target return
            ]

            bounds = [(0, 1)] * n_assets

            result = minimize(
                variance,
                np.ones(n_assets) / n_assets,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )

            if result.success:
                vol = np.sqrt(result.fun)
                sharpe = (target - self.risk_free_rate) / vol if vol > 0 else 0
                frontier.append({
                    'return': target,
                    'volatility': vol,
                    'sharpe': sharpe
                })

        return pd.DataFrame(frontier)

    def rebalance_portfolio(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        threshold: float = 0.02
    ) -> Dict[str, float]:
        """
        Calculate rebalancing trades with threshold.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            threshold: Minimum weight change to trigger rebalance

        Returns:
            Dictionary of weight changes (trades)
        """
        trades = {}

        all_assets = set(current_weights.keys()) | set(target_weights.keys())

        for asset in all_assets:
            current = current_weights.get(asset, 0)
            target = target_weights.get(asset, 0)
            diff = target - current

            if abs(diff) > threshold:
                trades[asset] = diff

        return trades
