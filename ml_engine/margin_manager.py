"""
Margin Manager - Liquidation Risk Monitoring and Leverage Optimization
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MarginStatus:
    """Margin account status"""
    equity: float
    margin_used: float
    margin_available: float
    margin_ratio: float  # equity / margin_used
    maintenance_margin: float
    liquidation_price: Dict[str, float]
    leverage_used: float
    max_leverage: float
    at_risk: bool
    warning_level: str  # 'safe', 'caution', 'danger'


class MarginManager:
    """
    Manages margin and leverage for trading.

    Features:
    - Liquidation risk monitoring
    - Leverage optimization
    - Margin requirement calculation
    - Position sizing based on margin
    """

    def __init__(
        self,
        max_leverage: float = 3.0,
        target_margin_ratio: float = 3.0,
        warning_margin_ratio: float = 2.0,
        danger_margin_ratio: float = 1.5,
        maintenance_margin_pct: float = 0.05,
        config=None
    ):
        """
        Initialize margin manager.

        Args:
            max_leverage: Maximum allowed leverage
            target_margin_ratio: Target margin ratio to maintain
            warning_margin_ratio: Ratio that triggers warning
            danger_margin_ratio: Ratio that triggers danger
            maintenance_margin_pct: Maintenance margin percentage
            config: Optional configuration
        """
        self.max_leverage = max_leverage
        self.target_margin_ratio = target_margin_ratio
        self.warning_margin_ratio = warning_margin_ratio
        self.danger_margin_ratio = danger_margin_ratio
        self.maintenance_margin_pct = maintenance_margin_pct
        self.config = config

    def calculate_margin_status(
        self,
        equity: float,
        positions: Dict[str, Dict],
        prices: Dict[str, float]
    ) -> MarginStatus:
        """
        Calculate current margin status.

        Args:
            equity: Account equity
            positions: Dict of positions {symbol: {side, quantity, entry_price}}
            prices: Current prices

        Returns:
            MarginStatus object
        """
        # Calculate margin used
        total_position_value = 0
        for symbol, pos in positions.items():
            price = prices.get(symbol, pos.get('entry_price', 0))
            value = abs(pos['quantity']) * price
            total_position_value += value

        margin_used = total_position_value / self.max_leverage if self.max_leverage > 0 else total_position_value
        margin_available = max(0, equity - margin_used)
        margin_ratio = equity / margin_used if margin_used > 0 else float('inf')

        # Calculate maintenance margin
        maintenance_margin = total_position_value * self.maintenance_margin_pct

        # Calculate liquidation prices
        liquidation_prices = {}
        for symbol, pos in positions.items():
            liq_price = self._calculate_liquidation_price(
                pos['side'],
                pos['quantity'],
                pos['entry_price'],
                equity,
                total_position_value
            )
            liquidation_prices[symbol] = liq_price

        # Determine warning level
        if margin_ratio >= self.target_margin_ratio:
            warning_level = 'safe'
        elif margin_ratio >= self.warning_margin_ratio:
            warning_level = 'caution'
        else:
            warning_level = 'danger'

        leverage_used = total_position_value / equity if equity > 0 else 0

        return MarginStatus(
            equity=equity,
            margin_used=margin_used,
            margin_available=margin_available,
            margin_ratio=margin_ratio,
            maintenance_margin=maintenance_margin,
            liquidation_price=liquidation_prices,
            leverage_used=leverage_used,
            max_leverage=self.max_leverage,
            at_risk=margin_ratio < self.danger_margin_ratio,
            warning_level=warning_level
        )

    def _calculate_liquidation_price(
        self,
        side: str,
        quantity: float,
        entry_price: float,
        equity: float,
        total_position_value: float
    ) -> float:
        """Calculate liquidation price for a position"""
        if quantity == 0:
            return 0

        position_value = quantity * entry_price
        margin_per_position = equity * (position_value / total_position_value) if total_position_value > 0 else equity

        # Simplified liquidation calculation
        # Liquidation occurs when loss = margin (minus maintenance)
        available_loss = margin_per_position * (1 - self.maintenance_margin_pct)

        if side == 'long':
            liq_price = entry_price - (available_loss / quantity)
        else:
            liq_price = entry_price + (available_loss / quantity)

        return max(0, liq_price)

    def calculate_optimal_leverage(
        self,
        volatility: float,
        target_volatility: float = 0.15,
        kelly_fraction: float = 0.5
    ) -> float:
        """
        Calculate optimal leverage based on volatility.

        Args:
            volatility: Asset/portfolio volatility
            target_volatility: Target portfolio volatility
            kelly_fraction: Fraction of Kelly criterion

        Returns:
            Optimal leverage
        """
        if volatility <= 0:
            return 1.0

        # Target leverage = target_vol / asset_vol
        target_leverage = target_volatility / volatility

        # Apply Kelly fraction
        optimal_leverage = target_leverage * kelly_fraction

        # Cap at max leverage
        return min(optimal_leverage, self.max_leverage)

    def calculate_position_size_margin(
        self,
        equity: float,
        margin_available: float,
        entry_price: float,
        max_position_pct: float = 0.2
    ) -> float:
        """
        Calculate maximum position size based on margin.

        Args:
            equity: Account equity
            margin_available: Available margin
            entry_price: Entry price for position
            max_position_pct: Maximum position as % of equity

        Returns:
            Maximum position size (quantity)
        """
        # Position value limit from equity
        max_from_equity = equity * max_position_pct

        # Position value limit from margin (with leverage)
        max_from_margin = margin_available * self.max_leverage

        # Use smaller limit
        max_position_value = min(max_from_equity, max_from_margin)

        if entry_price <= 0:
            return 0

        return max_position_value / entry_price

    def should_reduce_positions(
        self,
        margin_status: MarginStatus
    ) -> bool:
        """Check if positions should be reduced"""
        return margin_status.warning_level in ['caution', 'danger']

    def calculate_deleveraging_amount(
        self,
        margin_status: MarginStatus,
        positions: Dict[str, Dict]
    ) -> Dict[str, float]:
        """
        Calculate how much to reduce each position.

        Args:
            margin_status: Current margin status
            positions: Current positions

        Returns:
            Dict of {symbol: reduction_percentage}
        """
        if margin_status.warning_level == 'safe':
            return {}

        # Calculate target reduction
        current_leverage = margin_status.leverage_used
        target_leverage = self.max_leverage * 0.7  # Target 70% of max

        if current_leverage <= target_leverage:
            return {}

        reduction_pct = 1 - (target_leverage / current_leverage)

        # Apply uniform reduction
        reductions = {}
        for symbol in positions:
            reductions[symbol] = reduction_pct

        return reductions

    def get_margin_requirements(
        self,
        symbol: str,
        quantity: float,
        price: float
    ) -> Dict[str, float]:
        """
        Get margin requirements for a position.

        Args:
            symbol: Trading symbol
            quantity: Position quantity
            price: Entry price

        Returns:
            Dict with margin requirements
        """
        notional = abs(quantity) * price
        initial_margin = notional / self.max_leverage
        maintenance_margin = notional * self.maintenance_margin_pct

        return {
            'notional': notional,
            'initial_margin': initial_margin,
            'maintenance_margin': maintenance_margin,
            'max_leverage': self.max_leverage
        }
