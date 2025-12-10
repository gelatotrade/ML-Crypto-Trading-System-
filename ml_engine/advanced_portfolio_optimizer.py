"""
OPTIMIERTE PORTFOLIO-KONSTRUKTION

Implementiert:
A. Robustes Portfolio-Optimierung
   - Black-Litterman mit Bayes'scher Aktualisierung
   - Online Convex Optimization für sequentielle Entscheidungen
   - Robust Optimization unter Parameterunsicherheit

B. Positionsgrößen-Algorithmus mit Transaktionskosten
   - Optimale Kontrollformulierung
   - Hamilton-Jacobi-Bellman basierte Lösung
   - Almgren-Chriss Ausführungsmodell
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

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PortfolioWeights:
    """Portfolio weights with metadata"""
    weights: Dict[str, float]
    timestamp: datetime
    method: str
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    turnover: float
    transaction_costs: float


@dataclass
class BlackLittermanView:
    """Black-Litterman view specification"""
    assets: List[str]           # Assets involved in view
    view_weights: List[float]   # P matrix row (long/short weights)
    expected_return: float      # Q: Expected return of view
    confidence: float           # Omega: Confidence (lower = more confident)


@dataclass
class TradeSchedule:
    """Optimal trade execution schedule"""
    symbol: str
    target_quantity: float
    current_quantity: float
    schedule: List[Tuple[datetime, float]]  # (time, quantity to trade)
    expected_cost: float
    expected_risk: float
    participation_rate: float


@dataclass
class TransactionCostModel:
    """Transaction cost parameters"""
    spread: float               # Half bid-ask spread
    impact_coef: float         # a in impact model
    impact_power: float        # b in impact model (typically 0.5)
    fixed_cost: float          # Fixed cost per trade


# ═══════════════════════════════════════════════════════════════════════════════
# BLACK-LITTERMAN MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class BlackLittermanOptimizer:
    """
    Black-Litterman Portfolio Optimization.

    Combines market equilibrium (prior) with investor views (likelihood)
    to produce posterior expected returns.

    Π = δ * Σ * w_market  # Prior from market equilibrium
    E[R] = [(τΣ)^{-1} + P'Ω^{-1}P]^{-1} * [(τΣ)^{-1}Π + P'Ω^{-1}Q]
    """

    def __init__(
        self,
        risk_aversion: float = 2.5,
        tau: float = 0.05,  # Uncertainty in equilibrium
        risk_free_rate: float = 0.05
    ):
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.risk_free_rate = risk_free_rate

        # Cached data
        self.market_weights: Optional[np.ndarray] = None
        self.cov_matrix: Optional[np.ndarray] = None
        self.equilibrium_returns: Optional[np.ndarray] = None

    def set_market_data(
        self,
        market_weights: np.ndarray,
        cov_matrix: np.ndarray
    ):
        """
        Set market equilibrium data.

        Args:
            market_weights: Market cap weights
            cov_matrix: Covariance matrix
        """
        self.market_weights = market_weights
        self.cov_matrix = cov_matrix

        # Calculate equilibrium returns (reverse optimization)
        # Π = δ * Σ * w
        self.equilibrium_returns = self.risk_aversion * cov_matrix @ market_weights

    def add_views(
        self,
        views: List[BlackLittermanView],
        assets: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add investor views and calculate posterior returns.

        Args:
            views: List of BlackLittermanView objects
            assets: List of asset names (order matches cov_matrix)

        Returns:
            Posterior expected returns and posterior covariance
        """
        if self.cov_matrix is None:
            raise ValueError("Must call set_market_data first")

        n_assets = len(assets)
        n_views = len(views)

        # Build P matrix (views on assets)
        P = np.zeros((n_views, n_assets))
        Q = np.zeros(n_views)  # View returns
        omega_diag = np.zeros(n_views)  # View uncertainty

        asset_idx = {a: i for i, a in enumerate(assets)}

        for i, view in enumerate(views):
            for j, asset in enumerate(view.assets):
                if asset in asset_idx:
                    P[i, asset_idx[asset]] = view.view_weights[j]
            Q[i] = view.expected_return
            omega_diag[i] = view.confidence

        Omega = np.diag(omega_diag)

        # Black-Litterman posterior
        # M = [(τΣ)^{-1} + P'Ω^{-1}P]^{-1}
        tau_sigma_inv = np.linalg.inv(self.tau * self.cov_matrix + 1e-8 * np.eye(n_assets))

        # Only include views with non-zero P rows
        if n_views > 0:
            omega_inv = np.linalg.inv(Omega + 1e-8 * np.eye(n_views))
            M_inv = tau_sigma_inv + P.T @ omega_inv @ P
            M = np.linalg.inv(M_inv + 1e-8 * np.eye(n_assets))

            # Posterior return
            posterior_returns = M @ (tau_sigma_inv @ self.equilibrium_returns + P.T @ omega_inv @ Q)
        else:
            posterior_returns = self.equilibrium_returns
            M = self.tau * self.cov_matrix

        # Posterior covariance
        posterior_cov = self.cov_matrix + M

        return posterior_returns, posterior_cov

    def optimize(
        self,
        posterior_returns: np.ndarray,
        posterior_cov: np.ndarray,
        constraints: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Optimize portfolio weights given posterior.

        Args:
            posterior_returns: BL posterior returns
            posterior_cov: BL posterior covariance
            constraints: Optional constraints dict

        Returns:
            Optimal weights
        """
        n = len(posterior_returns)

        # Default constraints
        if constraints is None:
            constraints = {
                'long_only': False,
                'max_weight': 0.3,
                'min_weight': -0.3,
                'max_leverage': 2.0
            }

        # Mean-variance optimization
        # w* = (1/δ) * Σ^{-1} * μ
        cov_inv = np.linalg.inv(posterior_cov + 1e-8 * np.eye(n))
        raw_weights = (1 / self.risk_aversion) * cov_inv @ posterior_returns

        # Apply constraints
        weights = self._apply_constraints(raw_weights, constraints)

        return weights

    def _apply_constraints(
        self,
        weights: np.ndarray,
        constraints: Dict[str, Any]
    ) -> np.ndarray:
        """Apply portfolio constraints"""
        w = weights.copy()

        # Long-only
        if constraints.get('long_only', False):
            w = np.maximum(w, 0)

        # Position limits
        max_w = constraints.get('max_weight', 1.0)
        min_w = constraints.get('min_weight', -1.0)
        w = np.clip(w, min_w, max_w)

        # Leverage constraint
        max_leverage = constraints.get('max_leverage', 2.0)
        total_exposure = np.sum(np.abs(w))

        if total_exposure > max_leverage:
            w = w * max_leverage / total_exposure

        # Normalize if needed (sum to 1 for long-only)
        if constraints.get('long_only', False):
            w = w / (np.sum(w) + 1e-10)

        return w


# ═══════════════════════════════════════════════════════════════════════════════
# ONLINE CONVEX OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════

class OnlineConvexOptimizer:
    """
    Online Convex Optimization for sequential portfolio decisions.

    Implements:
    - Online Gradient Descent (OGD)
    - Follow the Regularized Leader (FTRL)
    - Online Newton Step (ONS)

    w_{t+1} = argmin_w [η_t * ∇loss(w_t)'w + Bregman(w, w_t)]
    """

    def __init__(
        self,
        n_assets: int,
        learning_rate: float = 0.01,
        algorithm: str = 'ogd',  # 'ogd', 'ftrl', 'ons'
        regularization: float = 0.001
    ):
        self.n_assets = n_assets
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        self.regularization = regularization

        # Current weights
        self.weights = np.ones(n_assets) / n_assets

        # History for adaptive learning
        self.gradient_history: List[np.ndarray] = []
        self.loss_history: List[float] = []

        # For ONS
        self.A = np.eye(n_assets) * regularization

    def update(
        self,
        returns: np.ndarray,
        risk_penalty: float = 0.5
    ) -> np.ndarray:
        """
        Update weights based on realized returns.

        Args:
            returns: Realized asset returns
            risk_penalty: Penalty for variance

        Returns:
            New portfolio weights
        """
        # Portfolio return
        port_return = self.weights @ returns

        # Loss function: -return + risk_penalty * variance
        # Gradient of -return wrt weights is -returns
        gradient = -returns

        # Add regularization gradient (toward equal weights)
        gradient += self.regularization * (self.weights - 1/self.n_assets)

        self.gradient_history.append(gradient)
        self.loss_history.append(-port_return)

        if self.algorithm == 'ogd':
            self.weights = self._ogd_update(gradient)
        elif self.algorithm == 'ftrl':
            self.weights = self._ftrl_update()
        elif self.algorithm == 'ons':
            self.weights = self._ons_update(gradient)

        return self.weights

    def _ogd_update(self, gradient: np.ndarray) -> np.ndarray:
        """Online Gradient Descent update"""
        # Adaptive learning rate
        t = len(self.gradient_history)
        lr = self.learning_rate / np.sqrt(t + 1)

        # Gradient step
        new_weights = self.weights - lr * gradient

        # Project to simplex (if long-only)
        new_weights = self._project_simplex(new_weights)

        return new_weights

    def _ftrl_update(self) -> np.ndarray:
        """Follow the Regularized Leader update"""
        if len(self.gradient_history) == 0:
            return self.weights

        # Sum of gradients
        sum_gradients = np.sum(self.gradient_history, axis=0)

        # Adaptive learning rate
        sum_sq_gradients = np.sum([g**2 for g in self.gradient_history], axis=0)
        adaptive_lr = self.learning_rate / (np.sqrt(sum_sq_gradients) + 1e-8)

        # Update
        new_weights = -adaptive_lr * sum_gradients

        # Regularization (L2)
        new_weights -= self.regularization * (new_weights - 1/self.n_assets)

        # Project
        new_weights = self._project_simplex(new_weights)

        return new_weights

    def _ons_update(self, gradient: np.ndarray) -> np.ndarray:
        """Online Newton Step update"""
        # Update Hessian approximation
        self.A += np.outer(gradient, gradient)

        # Newton step
        A_inv = np.linalg.inv(self.A + 1e-8 * np.eye(self.n_assets))

        new_weights = self.weights - self.learning_rate * A_inv @ gradient

        # Project
        new_weights = self._project_simplex(new_weights)

        return new_weights

    def _project_simplex(self, weights: np.ndarray) -> np.ndarray:
        """Project weights to probability simplex (or box constraints)"""
        # For long/short, use box constraints instead
        w = weights.copy()

        # Clip to reasonable range
        w = np.clip(w, -0.5, 0.5)

        # Normalize if total exposure > max
        max_leverage = 2.0
        if np.sum(np.abs(w)) > max_leverage:
            w = w * max_leverage / np.sum(np.abs(w))

        return w

    def get_regret(self) -> float:
        """Calculate cumulative regret"""
        if len(self.loss_history) == 0:
            return 0

        # Best fixed weights in hindsight
        if len(self.gradient_history) > 10:
            avg_gradients = np.mean(self.gradient_history, axis=0)
            best_weights = -avg_gradients / (np.abs(avg_gradients).sum() + 1e-8)
            best_weights = self._project_simplex(best_weights)

            # Regret = sum of (our loss - best loss)
            regret = sum(self.loss_history) - len(self.loss_history) * (best_weights @ avg_gradients)
        else:
            regret = 0

        return regret


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSACTION COST AWARE OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════

class TransactionCostOptimizer:
    """
    Portfolio optimization with transaction costs.

    Objective: max E[Σ(alpha_t * position_t) - costs]
    under risk constraints

    Cost model:
    TC = spread/2 + a * (volume/VWAP)^b + market_impact * sign(trade)
    """

    def __init__(
        self,
        cost_model: Optional[TransactionCostModel] = None,
        risk_aversion: float = 1.0,
        turnover_penalty: float = 0.01
    ):
        self.cost_model = cost_model or TransactionCostModel(
            spread=0.0005,        # 5 bps half-spread
            impact_coef=0.1,      # Impact coefficient
            impact_power=0.5,     # Square-root impact
            fixed_cost=0.0001    # 1 bp fixed cost
        )
        self.risk_aversion = risk_aversion
        self.turnover_penalty = turnover_penalty

    def estimate_transaction_cost(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        portfolio_value: float,
        adv: np.ndarray  # Average daily volume per asset
    ) -> float:
        """
        Estimate total transaction cost for rebalancing.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            portfolio_value: Total portfolio value
            adv: Average daily volume per asset

        Returns:
            Estimated transaction cost as fraction of portfolio
        """
        trades = target_weights - current_weights
        trade_values = np.abs(trades) * portfolio_value

        total_cost = 0

        for i, (trade_val, daily_vol) in enumerate(zip(trade_values, adv)):
            if trade_val < 1e-10:
                continue

            # Spread cost
            spread_cost = trade_val * self.cost_model.spread

            # Market impact (square-root model)
            participation = trade_val / (daily_vol + 1e-10)
            impact_cost = self.cost_model.impact_coef * (participation ** self.cost_model.impact_power) * trade_val

            # Fixed cost
            fixed_cost = self.cost_model.fixed_cost * portfolio_value

            total_cost += spread_cost + impact_cost + fixed_cost

        return total_cost / portfolio_value

    def optimize_with_costs(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        current_weights: np.ndarray,
        adv: np.ndarray,
        portfolio_value: float,
        constraints: Optional[Dict] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Optimize portfolio considering transaction costs.

        Uses iterative approach to find optimal trade-off between
        expected return improvement and transaction costs.

        Args:
            expected_returns: Expected asset returns
            cov_matrix: Covariance matrix
            current_weights: Current weights
            adv: Average daily volume
            portfolio_value: Portfolio value
            constraints: Optional constraints

        Returns:
            Optimal weights and expected net return
        """
        n = len(expected_returns)

        # Unconstrained optimal (ignoring costs)
        cov_inv = np.linalg.inv(cov_matrix + 1e-8 * np.eye(n))
        unconstrained_opt = (1 / self.risk_aversion) * cov_inv @ expected_returns

        # Normalize
        unconstrained_opt = self._normalize_weights(unconstrained_opt, constraints)

        # Cost of moving to unconstrained optimal
        full_cost = self.estimate_transaction_cost(
            current_weights, unconstrained_opt, portfolio_value, adv
        )

        # Expected return improvement
        current_return = expected_returns @ current_weights
        optimal_return = expected_returns @ unconstrained_opt

        return_improvement = optimal_return - current_return

        # If cost exceeds benefit, stay put or partially rebalance
        if full_cost > return_improvement * 0.5:  # Allow some trading if benefit > 2x cost
            # Partial rebalancing: move fraction toward target
            fraction = min(1.0, return_improvement / (2 * full_cost + 1e-10))
            fraction = max(0.1, fraction)  # At least 10% move

            target_weights = current_weights + fraction * (unconstrained_opt - current_weights)
            actual_cost = self.estimate_transaction_cost(
                current_weights, target_weights, portfolio_value, adv
            )
        else:
            target_weights = unconstrained_opt
            actual_cost = full_cost

        # Normalize final weights
        target_weights = self._normalize_weights(target_weights, constraints)

        expected_net_return = expected_returns @ target_weights - actual_cost

        return target_weights, expected_net_return

    def _normalize_weights(
        self,
        weights: np.ndarray,
        constraints: Optional[Dict] = None
    ) -> np.ndarray:
        """Normalize weights to satisfy constraints"""
        w = weights.copy()

        if constraints is None:
            constraints = {}

        # Long-only
        if constraints.get('long_only', False):
            w = np.maximum(w, 0)

        # Position limits
        max_w = constraints.get('max_weight', 0.3)
        min_w = constraints.get('min_weight', -0.3)
        w = np.clip(w, min_w, max_w)

        # Leverage
        max_leverage = constraints.get('max_leverage', 2.0)
        if np.sum(np.abs(w)) > max_leverage:
            w = w * max_leverage / np.sum(np.abs(w))

        return w


# ═══════════════════════════════════════════════════════════════════════════════
# ALMGREN-CHRISS EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

class AlmgrenChrissExecutor:
    """
    Almgren-Chriss Optimal Execution Model.

    Minimizes execution cost = E[Cost] + λ * Var[Cost]

    For linear temporary impact:
    Optimal trajectory: n_k = n_0 * sinh(κ(T-t_k)) / sinh(κT)

    Where κ = sqrt(λσ²/η)
    """

    def __init__(
        self,
        risk_aversion: float = 1e-6,
        temporary_impact: float = 0.1,
        permanent_impact: float = 0.05,
        volatility: float = 0.02  # Daily volatility
    ):
        self.risk_aversion = risk_aversion
        self.temporary_impact = temporary_impact  # η
        self.permanent_impact = permanent_impact  # γ
        self.volatility = volatility

    def compute_optimal_trajectory(
        self,
        total_shares: float,
        n_periods: int,
        period_length: float = 1.0  # In days
    ) -> Tuple[np.ndarray, float, float]:
        """
        Compute optimal execution trajectory.

        Args:
            total_shares: Total shares to execute
            n_periods: Number of trading periods
            period_length: Length of each period in days

        Returns:
            Trajectory (shares remaining), expected cost, cost variance
        """
        T = n_periods * period_length

        # Kappa parameter
        kappa_sq = self.risk_aversion * self.volatility**2 / self.temporary_impact
        kappa = np.sqrt(max(kappa_sq, 1e-10))

        # Optimal trajectory
        trajectory = np.zeros(n_periods + 1)
        trajectory[0] = total_shares

        for k in range(1, n_periods + 1):
            t_k = k * period_length
            trajectory[k] = total_shares * np.sinh(kappa * (T - t_k)) / np.sinh(kappa * T)

        # Trade sizes
        trade_sizes = -np.diff(trajectory)

        # Expected cost
        expected_cost = self._compute_expected_cost(total_shares, trade_sizes, T)

        # Cost variance
        cost_variance = self._compute_cost_variance(trajectory, T)

        return trajectory, expected_cost, cost_variance

    def _compute_expected_cost(
        self,
        total_shares: float,
        trade_sizes: np.ndarray,
        T: float
    ) -> float:
        """Compute expected execution cost"""
        # Permanent impact cost
        perm_cost = 0.5 * self.permanent_impact * total_shares**2

        # Temporary impact cost
        temp_cost = self.temporary_impact * np.sum(trade_sizes**2)

        return perm_cost + temp_cost

    def _compute_cost_variance(
        self,
        trajectory: np.ndarray,
        T: float
    ) -> float:
        """Compute variance of execution cost"""
        # Variance from price uncertainty
        variance = self.volatility**2 * np.sum(trajectory[:-1]**2)

        return variance

    def get_trade_schedule(
        self,
        symbol: str,
        target_quantity: float,
        current_quantity: float,
        trading_days: int = 5,
        trades_per_day: int = 4
    ) -> TradeSchedule:
        """
        Generate optimal trade schedule.

        Args:
            symbol: Asset symbol
            target_quantity: Target position
            current_quantity: Current position
            trading_days: Days to complete execution
            trades_per_day: Trades per day

        Returns:
            TradeSchedule with optimal execution plan
        """
        net_trade = target_quantity - current_quantity
        n_periods = trading_days * trades_per_day
        period_length = 1.0 / trades_per_day

        trajectory, expected_cost, cost_var = self.compute_optimal_trajectory(
            abs(net_trade), n_periods, period_length
        )

        # Generate schedule
        schedule = []
        base_time = datetime.utcnow()
        hours_per_period = 24 / trades_per_day

        for k in range(n_periods):
            trade_time = base_time + timedelta(hours=k * hours_per_period)
            trade_qty = (trajectory[k] - trajectory[k + 1]) * np.sign(net_trade)
            schedule.append((trade_time, trade_qty))

        # Participation rate
        participation = abs(net_trade) / (n_periods * 1e6)  # Rough estimate

        return TradeSchedule(
            symbol=symbol,
            target_quantity=target_quantity,
            current_quantity=current_quantity,
            schedule=schedule,
            expected_cost=expected_cost,
            expected_risk=np.sqrt(cost_var),
            participation_rate=participation
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ROBUST OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════

class RobustPortfolioOptimizer:
    """
    Robust Portfolio Optimization under parameter uncertainty.

    Handles:
    - Estimation error in expected returns
    - Estimation error in covariance
    - Parameter uncertainty sets
    """

    def __init__(
        self,
        kappa: float = 1.0,  # Uncertainty aversion parameter
        estimation_error_return: float = 0.05,
        estimation_error_cov: float = 0.1
    ):
        self.kappa = kappa
        self.error_return = estimation_error_return
        self.error_cov = estimation_error_cov

    def optimize_robust(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        n_samples: int = 100,  # Samples used for estimation
        constraints: Optional[Dict] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Robust optimization accounting for estimation error.

        Uses worst-case approach within uncertainty set.

        Args:
            expected_returns: Estimated expected returns
            cov_matrix: Estimated covariance
            n_samples: Number of samples used in estimation
            constraints: Optional constraints

        Returns:
            Robust optimal weights and worst-case Sharpe
        """
        n = len(expected_returns)

        # Estimation error bounds
        # Return uncertainty: proportional to standard errors
        return_std = np.sqrt(np.diag(cov_matrix) / n_samples)
        return_uncertainty = self.kappa * return_std

        # Covariance uncertainty (simplified)
        cov_uncertainty = self.error_cov * cov_matrix

        # Worst-case expected return: subtract uncertainty
        worst_case_returns = expected_returns - return_uncertainty

        # Worst-case covariance: add uncertainty
        worst_case_cov = cov_matrix + cov_uncertainty

        # Optimize for worst case
        cov_inv = np.linalg.inv(worst_case_cov + 1e-8 * np.eye(n))
        raw_weights = cov_inv @ worst_case_returns

        # Normalize
        weights = self._apply_constraints(raw_weights, constraints)

        # Worst-case Sharpe
        port_return = worst_case_returns @ weights
        port_risk = np.sqrt(weights @ worst_case_cov @ weights)
        worst_sharpe = port_return / (port_risk + 1e-10)

        return weights, worst_sharpe

    def _apply_constraints(
        self,
        weights: np.ndarray,
        constraints: Optional[Dict]
    ) -> np.ndarray:
        """Apply constraints to weights"""
        w = weights.copy()

        if constraints is None:
            constraints = {}

        # Clip to bounds
        max_w = constraints.get('max_weight', 0.3)
        min_w = constraints.get('min_weight', -0.3)
        w = np.clip(w, min_w, max_w)

        # Leverage
        max_lev = constraints.get('max_leverage', 2.0)
        if np.sum(np.abs(w)) > max_lev:
            w = w * max_lev / np.sum(np.abs(w))

        return w


# ═══════════════════════════════════════════════════════════════════════════════
# ADVANCED PORTFOLIO PIPELINE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

class AdvancedPortfolioPipeline:
    """
    Orchestrates all portfolio optimization components.
    """

    def __init__(
        self,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        use_black_litterman: bool = True,
        use_transaction_costs: bool = True,
        use_robust_optimization: bool = True
    ):
        self.risk_aversion = risk_aversion

        # Initialize components
        self.bl_optimizer = BlackLittermanOptimizer(
            risk_aversion=risk_aversion,
            tau=tau
        )
        self.online_optimizer = None  # Created per portfolio
        self.tc_optimizer = TransactionCostOptimizer(risk_aversion=risk_aversion)
        self.robust_optimizer = RobustPortfolioOptimizer(kappa=1.0)
        self.execution_model = AlmgrenChrissExecutor()

        self.use_bl = use_black_litterman
        self.use_tc = use_transaction_costs
        self.use_robust = use_robust_optimization

        # History
        self.weight_history: List[PortfolioWeights] = []

    def optimize(
        self,
        assets: List[str],
        returns_df: pd.DataFrame,
        current_weights: Optional[Dict[str, float]] = None,
        views: Optional[List[BlackLittermanView]] = None,
        adv: Optional[Dict[str, float]] = None,
        portfolio_value: float = 1000000,
        constraints: Optional[Dict] = None
    ) -> PortfolioWeights:
        """
        Full portfolio optimization pipeline.

        Args:
            assets: List of asset names
            returns_df: Historical returns DataFrame
            current_weights: Current portfolio weights
            views: Optional BL views
            adv: Average daily volume per asset
            portfolio_value: Total portfolio value
            constraints: Optimization constraints

        Returns:
            PortfolioWeights with optimal allocation
        """
        n = len(assets)
        timestamp = datetime.utcnow()

        # Default current weights
        if current_weights is None:
            current_weights = {a: 1/n for a in assets}

        current_w = np.array([current_weights.get(a, 0) for a in assets])

        # Calculate returns and covariance
        returns = returns_df[assets].dropna()
        expected_returns = returns.mean().values * 252 * 24  # Annualized
        cov_matrix = returns.cov().values * 252 * 24  # Annualized

        # Market weights (equal weight as proxy)
        market_weights = np.ones(n) / n

        # 1. Black-Litterman (if enabled and views provided)
        if self.use_bl:
            self.bl_optimizer.set_market_data(market_weights, cov_matrix)

            if views:
                posterior_returns, posterior_cov = self.bl_optimizer.add_views(views, assets)
            else:
                posterior_returns = expected_returns
                posterior_cov = cov_matrix
        else:
            posterior_returns = expected_returns
            posterior_cov = cov_matrix

        # 2. Robust optimization (if enabled)
        if self.use_robust:
            n_samples = len(returns)
            optimal_weights, _ = self.robust_optimizer.optimize_robust(
                posterior_returns, posterior_cov, n_samples, constraints
            )
        else:
            # Simple mean-variance
            cov_inv = np.linalg.inv(posterior_cov + 1e-8 * np.eye(n))
            optimal_weights = (1 / self.risk_aversion) * cov_inv @ posterior_returns
            optimal_weights = self._apply_constraints(optimal_weights, constraints)

        # 3. Transaction cost optimization (if enabled)
        if self.use_tc and adv:
            adv_array = np.array([adv.get(a, 1e9) for a in assets])
            optimal_weights, expected_net_return = self.tc_optimizer.optimize_with_costs(
                posterior_returns, posterior_cov, current_w,
                adv_array, portfolio_value, constraints
            )
        else:
            expected_net_return = posterior_returns @ optimal_weights

        # Calculate metrics
        port_return = posterior_returns @ optimal_weights
        port_risk = np.sqrt(optimal_weights @ posterior_cov @ optimal_weights)
        sharpe = (port_return - 0.05) / (port_risk + 1e-10)  # 5% risk-free

        # Turnover
        turnover = np.sum(np.abs(optimal_weights - current_w))

        # Transaction costs
        if adv:
            adv_array = np.array([adv.get(a, 1e9) for a in assets])
            tc = self.tc_optimizer.estimate_transaction_cost(
                current_w, optimal_weights, portfolio_value, adv_array
            )
        else:
            tc = 0

        # Build result
        weights_dict = {assets[i]: float(optimal_weights[i]) for i in range(n)}

        result = PortfolioWeights(
            weights=weights_dict,
            timestamp=timestamp,
            method='BL+Robust+TC' if self.use_bl else 'MeanVariance',
            expected_return=float(port_return),
            expected_risk=float(port_risk),
            sharpe_ratio=float(sharpe),
            turnover=float(turnover),
            transaction_costs=float(tc)
        )

        self.weight_history.append(result)

        return result

    def _apply_constraints(
        self,
        weights: np.ndarray,
        constraints: Optional[Dict]
    ) -> np.ndarray:
        """Apply constraints"""
        w = weights.copy()

        if constraints is None:
            constraints = {}

        max_w = constraints.get('max_weight', 0.3)
        min_w = constraints.get('min_weight', -0.3)
        w = np.clip(w, min_w, max_w)

        max_lev = constraints.get('max_leverage', 2.0)
        if np.sum(np.abs(w)) > max_lev:
            w = w * max_lev / np.sum(np.abs(w))

        return w

    def get_execution_schedule(
        self,
        current_positions: Dict[str, float],
        target_weights: Dict[str, float],
        portfolio_value: float,
        prices: Dict[str, float],
        trading_days: int = 3
    ) -> Dict[str, TradeSchedule]:
        """
        Generate execution schedules for all assets.

        Args:
            current_positions: Current positions (quantities)
            target_weights: Target weights from optimization
            portfolio_value: Total portfolio value
            prices: Current prices
            trading_days: Days to execute

        Returns:
            Dict of asset -> TradeSchedule
        """
        schedules = {}

        for asset, target_weight in target_weights.items():
            target_value = target_weight * portfolio_value
            price = prices.get(asset, 1)
            target_qty = target_value / price

            current_qty = current_positions.get(asset, 0)

            if abs(target_qty - current_qty) > 0.01 * abs(current_qty + 1):
                schedule = self.execution_model.get_trade_schedule(
                    symbol=asset,
                    target_quantity=target_qty,
                    current_quantity=current_qty,
                    trading_days=trading_days
                )
                schedules[asset] = schedule

        return schedules

    def update_online(
        self,
        realized_returns: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Online update based on realized returns.

        Args:
            realized_returns: Dict of asset -> realized return

        Returns:
            Updated weights
        """
        assets = list(realized_returns.keys())
        n = len(assets)

        if self.online_optimizer is None:
            self.online_optimizer = OnlineConvexOptimizer(n_assets=n)

        returns_array = np.array([realized_returns[a] for a in assets])

        new_weights = self.online_optimizer.update(returns_array)

        return {assets[i]: float(new_weights[i]) for i in range(n)}


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    'PortfolioWeights',
    'BlackLittermanView',
    'TradeSchedule',
    'TransactionCostModel',
    'BlackLittermanOptimizer',
    'OnlineConvexOptimizer',
    'TransactionCostOptimizer',
    'AlmgrenChrissExecutor',
    'RobustPortfolioOptimizer',
    'AdvancedPortfolioPipeline',
]
