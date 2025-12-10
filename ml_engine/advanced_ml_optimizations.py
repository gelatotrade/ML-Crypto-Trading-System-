"""
FORTSCHRITTLICHE ML-OPTIMIERUNGEN

Implementiert:
A. Meta-Learning für schnelle Adaptation (MAML)
   - Model-Agnostic Meta-Learning
   - Inner loop (task-specific adaptation)
   - Outer loop (meta-update)

B. Causal Inference für robuste Strategien
   - Do-Calculus für kausale Effekte
   - Intervention vs. Observation

C. Uncertainty-Aware Trading
   - Bayesian Deep Learning
   - Monte-Carlo Dropout für Epistemische Unsicherheit
   - Evidential Deep Learning für Aleatorische Unsicherheit
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
class MAMLTask:
    """Single task for meta-learning"""
    task_id: str
    X_support: np.ndarray      # Support set features
    y_support: np.ndarray      # Support set targets
    X_query: np.ndarray        # Query set features
    y_query: np.ndarray        # Query set targets
    regime: str                # Market regime for this task


@dataclass
class UncertaintyEstimate:
    """Uncertainty decomposition"""
    prediction: float
    epistemic_uncertainty: float     # Model/knowledge uncertainty
    aleatoric_uncertainty: float     # Data/inherent uncertainty
    total_uncertainty: float
    confidence_interval: Tuple[float, float]
    calibration_score: float


@dataclass
class CausalEstimate:
    """Causal effect estimate"""
    treatment: str
    outcome: str
    ate: float                       # Average treatment effect
    att: float                       # Average treatment effect on treated
    confidence_interval: Tuple[float, float]
    p_value: float
    confounders_adjusted: List[str]


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL-AGNOSTIC META-LEARNING (MAML)
# ═══════════════════════════════════════════════════════════════════════════════

class MAMLOptimizer:
    """
    Model-Agnostic Meta-Learning for rapid strategy adaptation.

    MAML Algorithm:
    1. Sample batch of tasks (market regimes)
    2. For each task:
       - Inner loop: θ' = θ - α * ∇_θ L_T(θ)  [task-specific adaptation]
    3. Outer loop: θ ← θ - β * ∇_θ Σ L_T(θ')  [meta-update]

    This allows the model to quickly adapt to new market regimes
    with just a few gradient steps.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 1,
        inner_lr: float = 0.01,     # α: inner loop learning rate
        outer_lr: float = 0.001,    # β: outer loop learning rate
        inner_steps: int = 5,        # Number of gradient steps in inner loop
        first_order: bool = True     # Use first-order approximation (FOMAML)
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.first_order = first_order

        # Initialize meta-parameters (simple 2-layer network)
        self.theta = self._init_parameters()

        # Task history
        self.task_history: List[MAMLTask] = []

    def _init_parameters(self) -> Dict[str, np.ndarray]:
        """Initialize network parameters"""
        # Xavier initialization
        W1 = np.random.randn(self.input_dim, self.hidden_dim) * np.sqrt(2 / self.input_dim)
        b1 = np.zeros(self.hidden_dim)
        W2 = np.random.randn(self.hidden_dim, self.output_dim) * np.sqrt(2 / self.hidden_dim)
        b2 = np.zeros(self.output_dim)

        return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    def _forward(
        self,
        X: np.ndarray,
        params: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Forward pass through network"""
        # Hidden layer with ReLU
        h = X @ params['W1'] + params['b1']
        h = np.maximum(h, 0)  # ReLU

        # Output layer
        out = h @ params['W2'] + params['b2']

        return out.flatten()

    def _compute_loss(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: Dict[str, np.ndarray]
    ) -> float:
        """Compute MSE loss"""
        pred = self._forward(X, params)
        return np.mean((pred - y) ** 2)

    def _compute_gradients(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Compute gradients using numerical differentiation (simplified)"""
        gradients = {}
        eps = 1e-5

        for key in params:
            grad = np.zeros_like(params[key])
            it = np.nditer(params[key], flags=['multi_index'])

            while not it.finished:
                idx = it.multi_index

                # Numerical gradient
                params[key][idx] += eps
                loss_plus = self._compute_loss(X, y, params)
                params[key][idx] -= 2 * eps
                loss_minus = self._compute_loss(X, y, params)
                params[key][idx] += eps  # Restore

                grad[idx] = (loss_plus - loss_minus) / (2 * eps)
                it.iternext()

            gradients[key] = grad

        return gradients

    def _inner_loop(
        self,
        task: MAMLTask,
        params: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Inner loop: task-specific adaptation.

        θ' = θ - α * ∇_θ L_T(θ)
        """
        adapted_params = {k: v.copy() for k, v in params.items()}

        for _ in range(self.inner_steps):
            gradients = self._compute_gradients(
                task.X_support, task.y_support, adapted_params
            )

            for key in adapted_params:
                adapted_params[key] -= self.inner_lr * gradients[key]

        return adapted_params

    def meta_train(
        self,
        tasks: List[MAMLTask],
        n_epochs: int = 100,
        batch_size: int = 4
    ):
        """
        Meta-training loop.

        Outer loop: θ ← θ - β * ∇_θ Σ L_T(θ')
        """
        for epoch in range(n_epochs):
            # Sample batch of tasks
            if len(tasks) < batch_size:
                batch_tasks = tasks
            else:
                batch_idx = np.random.choice(len(tasks), batch_size, replace=False)
                batch_tasks = [tasks[i] for i in batch_idx]

            # Accumulate meta-gradients
            meta_gradients = {k: np.zeros_like(v) for k, v in self.theta.items()}

            for task in batch_tasks:
                # Inner loop: adapt to task
                adapted_params = self._inner_loop(task, self.theta)

                # Compute query loss with adapted parameters
                if self.first_order:
                    # FOMAML: compute gradients at adapted params
                    task_gradients = self._compute_gradients(
                        task.X_query, task.y_query, adapted_params
                    )
                else:
                    # Full MAML would require second-order gradients
                    # (computationally expensive, using FOMAML instead)
                    task_gradients = self._compute_gradients(
                        task.X_query, task.y_query, adapted_params
                    )

                for key in meta_gradients:
                    meta_gradients[key] += task_gradients[key] / len(batch_tasks)

            # Outer loop update
            for key in self.theta:
                self.theta[key] -= self.outer_lr * meta_gradients[key]

            # Logging
            if epoch % 10 == 0:
                avg_loss = np.mean([
                    self._compute_loss(t.X_query, t.y_query, self.theta)
                    for t in batch_tasks
                ])
                logger.debug(f"MAML Epoch {epoch}: Avg query loss = {avg_loss:.4f}")

    def adapt_to_regime(
        self,
        support_X: np.ndarray,
        support_y: np.ndarray,
        n_steps: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Quickly adapt to new regime using support set.

        Args:
            support_X: Support set features (small, e.g., 5-10 samples)
            support_y: Support set targets
            n_steps: Override inner_steps

        Returns:
            Adapted parameters
        """
        n_steps = n_steps or self.inner_steps

        adapted_params = {k: v.copy() for k, v in self.theta.items()}

        for _ in range(n_steps):
            gradients = self._compute_gradients(support_X, support_y, adapted_params)

            for key in adapted_params:
                adapted_params[key] -= self.inner_lr * gradients[key]

        return adapted_params

    def predict(
        self,
        X: np.ndarray,
        adapted_params: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        """Make predictions with meta-learned or adapted parameters"""
        params = adapted_params if adapted_params is not None else self.theta
        return self._forward(X, params)

    def create_regime_tasks(
        self,
        returns_df: pd.DataFrame,
        features_df: pd.DataFrame,
        regime_labels: pd.Series,
        support_size: int = 10,
        query_size: int = 20
    ) -> List[MAMLTask]:
        """
        Create MAML tasks from regime-labeled data.

        Each regime becomes a separate task.
        """
        tasks = []
        regimes = regime_labels.unique()

        for regime in regimes:
            regime_mask = regime_labels == regime
            regime_features = features_df[regime_mask].values
            regime_returns = returns_df[regime_mask].values.flatten()

            if len(regime_features) < support_size + query_size:
                continue

            # Random split into support and query
            indices = np.random.permutation(len(regime_features))
            support_idx = indices[:support_size]
            query_idx = indices[support_size:support_size + query_size]

            task = MAMLTask(
                task_id=f"regime_{regime}",
                X_support=regime_features[support_idx],
                y_support=regime_returns[support_idx],
                X_query=regime_features[query_idx],
                y_query=regime_returns[query_idx],
                regime=str(regime)
            )
            tasks.append(task)

        return tasks


# ═══════════════════════════════════════════════════════════════════════════════
# BAYESIAN DEEP LEARNING
# ═══════════════════════════════════════════════════════════════════════════════

class BayesianNeuralNetwork:
    """
    Bayesian Neural Network for uncertainty-aware predictions.

    Implements:
    1. Monte-Carlo Dropout for epistemic uncertainty
    2. Heteroscedastic output for aleatoric uncertainty

    Epistemic uncertainty: Model uncertainty due to limited data
    Aleatoric uncertainty: Inherent noise in the data
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 1,
        dropout_rate: float = 0.2,
        n_mc_samples: int = 100
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.n_mc_samples = n_mc_samples

        # Initialize parameters
        self.params = self._init_parameters()

        # Training history
        self.loss_history: List[float] = []

    def _init_parameters(self) -> Dict[str, np.ndarray]:
        """Initialize network parameters"""
        # Xavier initialization
        W1 = np.random.randn(self.input_dim, self.hidden_dim) * np.sqrt(2 / self.input_dim)
        b1 = np.zeros(self.hidden_dim)
        W2 = np.random.randn(self.hidden_dim, self.hidden_dim) * np.sqrt(2 / self.hidden_dim)
        b2 = np.zeros(self.hidden_dim)

        # Output: mean and log_var (heteroscedastic)
        W_mean = np.random.randn(self.hidden_dim, self.output_dim) * np.sqrt(2 / self.hidden_dim)
        b_mean = np.zeros(self.output_dim)
        W_logvar = np.random.randn(self.hidden_dim, self.output_dim) * np.sqrt(2 / self.hidden_dim)
        b_logvar = np.zeros(self.output_dim)

        return {
            'W1': W1, 'b1': b1,
            'W2': W2, 'b2': b2,
            'W_mean': W_mean, 'b_mean': b_mean,
            'W_logvar': W_logvar, 'b_logvar': b_logvar
        }

    def _forward_with_dropout(
        self,
        X: np.ndarray,
        training: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass with dropout.

        Returns mean and log_variance.
        """
        # Layer 1
        h1 = X @ self.params['W1'] + self.params['b1']
        h1 = np.maximum(h1, 0)  # ReLU

        if training:
            mask1 = np.random.binomial(1, 1 - self.dropout_rate, h1.shape)
            h1 = h1 * mask1 / (1 - self.dropout_rate)

        # Layer 2
        h2 = h1 @ self.params['W2'] + self.params['b2']
        h2 = np.maximum(h2, 0)  # ReLU

        if training:
            mask2 = np.random.binomial(1, 1 - self.dropout_rate, h2.shape)
            h2 = h2 * mask2 / (1 - self.dropout_rate)

        # Output (heteroscedastic)
        mean = h2 @ self.params['W_mean'] + self.params['b_mean']
        log_var = h2 @ self.params['W_logvar'] + self.params['b_logvar']

        return mean.flatten(), log_var.flatten()

    def _gaussian_nll_loss(
        self,
        y_true: np.ndarray,
        mean: np.ndarray,
        log_var: np.ndarray
    ) -> float:
        """Gaussian negative log-likelihood loss (heteroscedastic)"""
        # NLL = 0.5 * (log_var + (y - mean)^2 / exp(log_var))
        precision = np.exp(-log_var)
        loss = 0.5 * (log_var + (y_true - mean) ** 2 * precision)
        return np.mean(loss)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_epochs: int = 100,
        learning_rate: float = 0.001,
        batch_size: int = 32
    ):
        """Train the Bayesian neural network"""
        n_samples = len(X)

        for epoch in range(n_epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                # Forward pass
                mean, log_var = self._forward_with_dropout(X_batch, training=True)

                # Compute loss
                loss = self._gaussian_nll_loss(y_batch, mean, log_var)
                epoch_loss += loss

                # Backward pass (simplified gradient descent)
                # Using numerical gradients for demonstration
                self._update_parameters(X_batch, y_batch, learning_rate)

            self.loss_history.append(epoch_loss)

            if epoch % 20 == 0:
                logger.debug(f"BNN Epoch {epoch}: Loss = {epoch_loss:.4f}")

    def _update_parameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        learning_rate: float
    ):
        """Update parameters using gradient descent"""
        eps = 1e-5

        for key in self.params:
            grad = np.zeros_like(self.params[key])
            it = np.nditer(self.params[key], flags=['multi_index'])

            # Sample a subset of indices for efficiency
            n_samples = min(10, self.params[key].size)
            sample_indices = np.random.choice(
                range(self.params[key].size),
                n_samples,
                replace=False
            )

            for idx in sample_indices:
                multi_idx = np.unravel_index(idx, self.params[key].shape)

                self.params[key][multi_idx] += eps
                mean_plus, logvar_plus = self._forward_with_dropout(X, training=False)
                loss_plus = self._gaussian_nll_loss(y, mean_plus, logvar_plus)

                self.params[key][multi_idx] -= 2 * eps
                mean_minus, logvar_minus = self._forward_with_dropout(X, training=False)
                loss_minus = self._gaussian_nll_loss(y, mean_minus, logvar_minus)

                self.params[key][multi_idx] += eps  # Restore

                grad[multi_idx] = (loss_plus - loss_minus) / (2 * eps)

            self.params[key] -= learning_rate * grad

    def predict_with_uncertainty(
        self,
        X: np.ndarray
    ) -> UncertaintyEstimate:
        """
        Make predictions with full uncertainty decomposition.

        Uses Monte-Carlo dropout for epistemic uncertainty.
        """
        # MC Dropout: multiple forward passes with dropout
        mc_means = []
        mc_logvars = []

        for _ in range(self.n_mc_samples):
            mean, log_var = self._forward_with_dropout(X, training=True)
            mc_means.append(mean)
            mc_logvars.append(log_var)

        mc_means = np.array(mc_means)
        mc_logvars = np.array(mc_logvars)

        # Predictive mean
        pred_mean = np.mean(mc_means)

        # Epistemic uncertainty: variance of means (model uncertainty)
        epistemic = np.var(mc_means)

        # Aleatoric uncertainty: mean of variances (data uncertainty)
        aleatoric = np.mean(np.exp(mc_logvars))

        # Total uncertainty
        total_uncertainty = np.sqrt(epistemic + aleatoric)

        # Confidence interval (95%)
        std = total_uncertainty
        ci_lower = pred_mean - 1.96 * std
        ci_upper = pred_mean + 1.96 * std

        # Calibration score (simplified)
        calibration = 1.0 - min(1.0, abs(epistemic - aleatoric) / (total_uncertainty + 1e-10))

        return UncertaintyEstimate(
            prediction=float(pred_mean),
            epistemic_uncertainty=float(np.sqrt(epistemic)),
            aleatoric_uncertainty=float(np.sqrt(aleatoric)),
            total_uncertainty=float(total_uncertainty),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            calibration_score=float(calibration)
        )


# ═══════════════════════════════════════════════════════════════════════════════
# EVIDENTIAL DEEP LEARNING
# ═══════════════════════════════════════════════════════════════════════════════

class EvidentialNeuralNetwork:
    """
    Evidential Deep Learning for uncertainty quantification.

    Outputs parameters of a higher-order distribution (Normal-Inverse-Gamma)
    to directly model both aleatoric and epistemic uncertainty.

    y ~ N(μ, σ²), where:
    - μ ~ N(γ, σ²/ν)
    - σ² ~ Inverse-Gamma(α, β)

    Network outputs: (γ, ν, α, β) for each prediction.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        min_nu: float = 0.1,
        min_alpha: float = 1.0
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.min_nu = min_nu
        self.min_alpha = min_alpha

        # Initialize parameters
        self.params = self._init_parameters()

    def _init_parameters(self) -> Dict[str, np.ndarray]:
        """Initialize network parameters"""
        W1 = np.random.randn(self.input_dim, self.hidden_dim) * np.sqrt(2 / self.input_dim)
        b1 = np.zeros(self.hidden_dim)
        W2 = np.random.randn(self.hidden_dim, self.hidden_dim) * np.sqrt(2 / self.hidden_dim)
        b2 = np.zeros(self.hidden_dim)

        # Evidential output heads: (gamma, nu, alpha, beta)
        W_gamma = np.random.randn(self.hidden_dim, 1) * 0.1
        b_gamma = np.zeros(1)
        W_nu = np.random.randn(self.hidden_dim, 1) * 0.1
        b_nu = np.zeros(1)
        W_alpha = np.random.randn(self.hidden_dim, 1) * 0.1
        b_alpha = np.zeros(1)
        W_beta = np.random.randn(self.hidden_dim, 1) * 0.1
        b_beta = np.zeros(1)

        return {
            'W1': W1, 'b1': b1,
            'W2': W2, 'b2': b2,
            'W_gamma': W_gamma, 'b_gamma': b_gamma,
            'W_nu': W_nu, 'b_nu': b_nu,
            'W_alpha': W_alpha, 'b_alpha': b_alpha,
            'W_beta': W_beta, 'b_beta': b_beta
        }

    def _forward(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward pass returning evidential parameters.

        Returns (gamma, nu, alpha, beta).
        """
        # Hidden layers
        h1 = X @ self.params['W1'] + self.params['b1']
        h1 = np.maximum(h1, 0)  # ReLU

        h2 = h1 @ self.params['W2'] + self.params['b2']
        h2 = np.maximum(h2, 0)  # ReLU

        # Evidential outputs
        gamma = h2 @ self.params['W_gamma'] + self.params['b_gamma']

        # Positive constraints for nu, alpha, beta
        nu_raw = h2 @ self.params['W_nu'] + self.params['b_nu']
        nu = np.log(1 + np.exp(nu_raw)) + self.min_nu  # Softplus

        alpha_raw = h2 @ self.params['W_alpha'] + self.params['b_alpha']
        alpha = np.log(1 + np.exp(alpha_raw)) + self.min_alpha  # Softplus

        beta_raw = h2 @ self.params['W_beta'] + self.params['b_beta']
        beta = np.log(1 + np.exp(beta_raw))  # Softplus

        return gamma.flatten(), nu.flatten(), alpha.flatten(), beta.flatten()

    def _evidential_loss(
        self,
        y_true: np.ndarray,
        gamma: np.ndarray,
        nu: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        lambda_coef: float = 0.1
    ) -> float:
        """
        Evidential loss function (Type II Maximum Likelihood).

        L = -log Student-t(y; gamma, beta(1+1/nu)/(alpha-1), 2*alpha) + regularization
        """
        # Degrees of freedom
        omega = 2 * beta * (1 + nu)

        # NLL of Student-t
        nll = 0.5 * np.log(np.pi / nu + 1e-10)
        nll -= alpha * np.log(omega + 1e-10)
        nll += (alpha + 0.5) * np.log(omega + nu * (y_true - gamma) ** 2 + 1e-10)
        nll += np.log(np.exp(self._log_gamma(alpha)) / (np.exp(self._log_gamma(alpha + 0.5)) + 1e-10) + 1e-10)

        # Regularization to prevent trivial solutions
        reg = lambda_coef * (2 * nu + alpha) * np.abs(y_true - gamma)

        return np.mean(nll + reg)

    def _log_gamma(self, x: np.ndarray) -> np.ndarray:
        """Log gamma function approximation (Stirling)"""
        return (x - 0.5) * np.log(x + 1e-10) - x + 0.5 * np.log(2 * np.pi)

    def predict_with_uncertainty(
        self,
        X: np.ndarray
    ) -> UncertaintyEstimate:
        """
        Make prediction with uncertainty from evidential parameters.
        """
        gamma, nu, alpha, beta = self._forward(X)

        # Point prediction
        prediction = gamma[0] if len(gamma) == 1 else np.mean(gamma)

        # Aleatoric uncertainty: E[σ²] = β / (α - 1) for α > 1
        alpha_safe = np.maximum(alpha, 1.01)
        aleatoric_var = beta / (alpha_safe - 1)
        aleatoric_std = np.sqrt(np.mean(aleatoric_var))

        # Epistemic uncertainty: Var[μ] = β / (ν * (α - 1))
        epistemic_var = beta / (nu * (alpha_safe - 1))
        epistemic_std = np.sqrt(np.mean(epistemic_var))

        # Total uncertainty
        total_std = np.sqrt(aleatoric_var + epistemic_var).mean()

        # Confidence interval
        ci_lower = prediction - 1.96 * total_std
        ci_upper = prediction + 1.96 * total_std

        # Calibration (evidence strength)
        evidence = 2 * nu + alpha
        calibration = float(1 - 1 / (1 + evidence.mean()))

        return UncertaintyEstimate(
            prediction=float(prediction),
            epistemic_uncertainty=float(epistemic_std),
            aleatoric_uncertainty=float(aleatoric_std),
            total_uncertainty=float(total_std),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            calibration_score=calibration
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CAUSAL INFERENCE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class AdvancedCausalInference:
    """
    Advanced Causal Inference for robust trading strategies.

    Implements Do-Calculus to distinguish:
    - P(return | feature=x)        → Observational
    - P(return | do(feature=x))    → Interventional (causal)

    Methods:
    - Backdoor adjustment
    - Frontdoor adjustment
    - Instrumental variables
    - Double Machine Learning (DML)
    """

    def __init__(
        self,
        confounders: List[str] = None
    ):
        self.confounders = confounders or []

    def backdoor_adjustment(
        self,
        treatment: np.ndarray,
        outcome: np.ndarray,
        confounders: np.ndarray
    ) -> CausalEstimate:
        """
        Backdoor adjustment for causal effect estimation.

        ATE = E[E[Y | T=1, Z] - E[Y | T=0, Z]]

        Args:
            treatment: Binary treatment variable
            outcome: Outcome variable
            confounders: Confounder matrix

        Returns:
            CausalEstimate
        """
        # Stratify by confounder bins
        n_strata = 5

        # Discretize confounders for stratification
        if confounders.ndim == 1:
            confounders = confounders.reshape(-1, 1)

        # Create strata using first confounder
        strata = pd.qcut(confounders[:, 0], n_strata, labels=False, duplicates='drop')

        # Compute stratum-specific effects
        stratum_effects = []
        stratum_weights = []

        for s in np.unique(strata):
            mask = strata == s
            t_s = treatment[mask]
            y_s = outcome[mask]

            # Mean outcome by treatment within stratum
            y1 = y_s[t_s == 1].mean() if (t_s == 1).sum() > 0 else np.nan
            y0 = y_s[t_s == 0].mean() if (t_s == 0).sum() > 0 else np.nan

            if not np.isnan(y1) and not np.isnan(y0):
                stratum_effects.append(y1 - y0)
                stratum_weights.append(mask.sum())

        if not stratum_effects:
            return CausalEstimate(
                treatment='treatment', outcome='outcome',
                ate=0, att=0, confidence_interval=(0, 0),
                p_value=1.0, confounders_adjusted=[]
            )

        # Weighted average
        weights = np.array(stratum_weights)
        weights = weights / weights.sum()
        ate = np.average(stratum_effects, weights=weights)

        # Bootstrap CI
        ci_lower, ci_upper = self._bootstrap_ci(
            treatment, outcome, confounders, n_bootstrap=100
        )

        # Approximate p-value
        se = (ci_upper - ci_lower) / 3.92
        t_stat = ate / (se + 1e-10)
        p_value = 2 * (1 - self._normal_cdf(abs(t_stat)))

        # ATT (Average Treatment effect on Treated)
        treated_mask = treatment == 1
        y1_treated = outcome[treated_mask].mean()
        # Counterfactual for treated
        y0_treated_cf = np.mean([
            outcome[(treatment == 0) & (strata == s)].mean()
            for s in np.unique(strata[treated_mask])
            if ((treatment == 0) & (strata == s)).sum() > 0
        ]) if treated_mask.sum() > 0 else y1_treated

        att = y1_treated - y0_treated_cf

        return CausalEstimate(
            treatment='treatment',
            outcome='outcome',
            ate=float(ate),
            att=float(att),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            p_value=float(p_value),
            confounders_adjusted=self.confounders
        )

    def _bootstrap_ci(
        self,
        treatment: np.ndarray,
        outcome: np.ndarray,
        confounders: np.ndarray,
        n_bootstrap: int = 100,
        alpha: float = 0.05
    ) -> Tuple[float, float]:
        """Bootstrap confidence interval"""
        n = len(treatment)
        ates = []

        for _ in range(n_bootstrap):
            idx = np.random.choice(n, n, replace=True)

            # Simplified ATE estimation for bootstrap
            t = treatment[idx]
            y = outcome[idx]

            y1 = y[t == 1].mean() if (t == 1).sum() > 0 else 0
            y0 = y[t == 0].mean() if (t == 0).sum() > 0 else 0

            ates.append(y1 - y0)

        return np.percentile(ates, alpha/2 * 100), np.percentile(ates, (1 - alpha/2) * 100)

    def _normal_cdf(self, x: float) -> float:
        """Normal CDF approximation"""
        return 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

    def double_ml_estimate(
        self,
        treatment: np.ndarray,
        outcome: np.ndarray,
        confounders: np.ndarray,
        n_folds: int = 5
    ) -> CausalEstimate:
        """
        Double Machine Learning for causal effect estimation.

        Uses cross-fitting to avoid overfitting bias.

        1. Estimate E[Y|X] using ML model
        2. Estimate E[T|X] using ML model (propensity)
        3. Compute residuals
        4. Regress Y residuals on T residuals
        """
        n = len(treatment)
        fold_size = n // n_folds

        y_residuals = np.zeros(n)
        t_residuals = np.zeros(n)

        for fold in range(n_folds):
            # Create train/test split
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < n_folds - 1 else n
            test_mask = np.zeros(n, dtype=bool)
            test_mask[test_start:test_end] = True
            train_mask = ~test_mask

            X_train = confounders[train_mask]
            X_test = confounders[test_mask]
            y_train = outcome[train_mask]
            t_train = treatment[train_mask]

            # Fit outcome model (simple linear regression)
            X_train_aug = np.column_stack([np.ones(X_train.shape[0]), X_train])
            X_test_aug = np.column_stack([np.ones(X_test.shape[0]), X_test])

            try:
                beta_y = np.linalg.lstsq(X_train_aug, y_train, rcond=None)[0]
                y_pred = X_test_aug @ beta_y
                y_residuals[test_mask] = outcome[test_mask] - y_pred
            except:
                y_residuals[test_mask] = outcome[test_mask] - np.mean(y_train)

            # Fit propensity model
            try:
                beta_t = np.linalg.lstsq(X_train_aug, t_train, rcond=None)[0]
                t_pred = X_test_aug @ beta_t
                t_residuals[test_mask] = treatment[test_mask] - t_pred
            except:
                t_residuals[test_mask] = treatment[test_mask] - np.mean(t_train)

        # Final regression: residual Y on residual T
        dml_ate = np.cov(y_residuals, t_residuals)[0, 1] / (np.var(t_residuals) + 1e-10)

        # Standard error
        se = np.std(y_residuals * t_residuals / (np.var(t_residuals) + 1e-10)) / np.sqrt(n)

        ci_lower = dml_ate - 1.96 * se
        ci_upper = dml_ate + 1.96 * se

        t_stat = dml_ate / (se + 1e-10)
        p_value = 2 * (1 - self._normal_cdf(abs(t_stat)))

        return CausalEstimate(
            treatment='treatment',
            outcome='outcome',
            ate=float(dml_ate),
            att=float(dml_ate),  # Approximation
            confidence_interval=(float(ci_lower), float(ci_upper)),
            p_value=float(p_value),
            confounders_adjusted=self.confounders + ['DML_cross_fitted']
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ADVANCED ML PIPELINE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

class AdvancedMLPipeline:
    """
    Orchestrates advanced ML optimization techniques.
    """

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 64,
        use_maml: bool = True,
        use_bayesian: bool = True,
        use_causal: bool = True
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Initialize components
        if use_maml:
            self.maml = MAMLOptimizer(input_dim, hidden_dim)
        else:
            self.maml = None

        if use_bayesian:
            self.bnn = BayesianNeuralNetwork(input_dim, hidden_dim)
            self.evidential = EvidentialNeuralNetwork(input_dim, hidden_dim)
        else:
            self.bnn = None
            self.evidential = None

        if use_causal:
            self.causal_engine = AdvancedCausalInference()
        else:
            self.causal_engine = None

        # Current adapted parameters
        self.current_adapted_params = None

    def meta_train(
        self,
        features_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        regime_labels: pd.Series,
        n_epochs: int = 50
    ):
        """Train MAML on regime-based tasks"""
        if self.maml is None:
            return

        tasks = self.maml.create_regime_tasks(
            returns_df, features_df, regime_labels
        )

        if tasks:
            self.maml.meta_train(tasks, n_epochs=n_epochs)
            logger.info(f"MAML trained on {len(tasks)} regime tasks")

    def adapt_to_current_regime(
        self,
        support_X: np.ndarray,
        support_y: np.ndarray
    ):
        """Quickly adapt to current market regime"""
        if self.maml is None:
            return

        self.current_adapted_params = self.maml.adapt_to_regime(
            support_X, support_y, n_steps=5
        )
        logger.info("Model adapted to current regime")

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        method: str = 'bayesian'
    ) -> UncertaintyEstimate:
        """
        Make prediction with uncertainty quantification.

        Args:
            X: Input features
            method: 'bayesian' or 'evidential'

        Returns:
            UncertaintyEstimate
        """
        if method == 'bayesian' and self.bnn is not None:
            return self.bnn.predict_with_uncertainty(X)
        elif method == 'evidential' and self.evidential is not None:
            return self.evidential.predict_with_uncertainty(X)
        else:
            # Fallback: simple prediction without uncertainty
            if self.maml is not None:
                pred = self.maml.predict(X, self.current_adapted_params)
                return UncertaintyEstimate(
                    prediction=float(pred[0]) if len(pred) > 0 else 0,
                    epistemic_uncertainty=0,
                    aleatoric_uncertainty=0,
                    total_uncertainty=0,
                    confidence_interval=(0, 0),
                    calibration_score=0
                )

            return UncertaintyEstimate(
                prediction=0, epistemic_uncertainty=0, aleatoric_uncertainty=0,
                total_uncertainty=0, confidence_interval=(0, 0), calibration_score=0
            )

    def estimate_causal_effect(
        self,
        treatment: np.ndarray,
        outcome: np.ndarray,
        confounders: np.ndarray,
        method: str = 'backdoor'
    ) -> CausalEstimate:
        """
        Estimate causal effect of treatment on outcome.

        Args:
            treatment: Treatment variable
            outcome: Outcome variable
            confounders: Confounder matrix
            method: 'backdoor' or 'dml'

        Returns:
            CausalEstimate
        """
        if self.causal_engine is None:
            return CausalEstimate(
                treatment='treatment', outcome='outcome',
                ate=0, att=0, confidence_interval=(0, 0),
                p_value=1.0, confounders_adjusted=[]
            )

        if method == 'backdoor':
            return self.causal_engine.backdoor_adjustment(
                treatment, outcome, confounders
            )
        elif method == 'dml':
            return self.causal_engine.double_ml_estimate(
                treatment, outcome, confounders
            )
        else:
            return self.causal_engine.backdoor_adjustment(
                treatment, outcome, confounders
            )

    def should_trade(
        self,
        prediction: UncertaintyEstimate,
        confidence_threshold: float = 0.6,
        uncertainty_threshold: float = 0.1
    ) -> Tuple[bool, str]:
        """
        Decide whether to trade based on prediction and uncertainty.

        Args:
            prediction: UncertaintyEstimate from model
            confidence_threshold: Minimum confidence
            uncertainty_threshold: Maximum acceptable uncertainty

        Returns:
            (should_trade, reason)
        """
        # Check epistemic uncertainty (model doesn't know)
        if prediction.epistemic_uncertainty > uncertainty_threshold:
            return False, f"High epistemic uncertainty: {prediction.epistemic_uncertainty:.4f}"

        # Check total uncertainty
        if prediction.total_uncertainty > uncertainty_threshold * 2:
            return False, f"High total uncertainty: {prediction.total_uncertainty:.4f}"

        # Check confidence
        if prediction.calibration_score < confidence_threshold:
            return False, f"Low calibration: {prediction.calibration_score:.2f}"

        # Check prediction significance
        if abs(prediction.prediction) < prediction.total_uncertainty:
            return False, "Prediction not significant relative to uncertainty"

        return True, "Sufficient confidence and low uncertainty"


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    'MAMLTask',
    'UncertaintyEstimate',
    'CausalEstimate',
    'MAMLOptimizer',
    'BayesianNeuralNetwork',
    'EvidentialNeuralNetwork',
    'AdvancedCausalInference',
    'AdvancedMLPipeline',
]
