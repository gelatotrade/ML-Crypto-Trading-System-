"""
Position Manager - Real-time Position Tracking and P&L Calculation
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Individual position representation"""
    symbol: str
    side: str  # 'long' or 'short'
    quantity: float
    entry_price: float
    current_price: float = 0
    unrealized_pnl: float = 0
    realized_pnl: float = 0
    opened_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    def update_price(self, price: float):
        """Update position with new price"""
        self.current_price = price
        self.updated_at = datetime.utcnow()

        if self.side == 'long':
            self.unrealized_pnl = (price - self.entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.entry_price - price) * self.quantity

    @property
    def market_value(self) -> float:
        """Current market value of position"""
        return abs(self.quantity * self.current_price)

    @property
    def pnl_percent(self) -> float:
        """P&L as percentage"""
        cost = self.quantity * self.entry_price
        if cost == 0:
            return 0
        return self.unrealized_pnl / abs(cost) * 100


@dataclass
class PortfolioMetrics:
    """Portfolio-level metrics"""
    total_value: float
    cash: float
    positions_value: float
    total_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    long_exposure: float
    short_exposure: float
    net_exposure: float
    gross_exposure: float
    num_positions: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float


class PositionManager:
    """
    Manages portfolio positions and calculates P&L.

    Features:
    - Real-time position tracking
    - P&L calculation (realized & unrealized)
    - Exposure monitoring
    - Performance metrics
    - Stop-loss and take-profit checking
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        config=None
    ):
        """
        Initialize position manager.

        Args:
            initial_capital: Starting capital
            config: Optional configuration
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.config = config

        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.pnl_history: List[Dict] = []

    def open_position(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Position:
        """
        Open a new position.

        Args:
            symbol: Trading symbol
            side: 'long' or 'short'
            quantity: Position size
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price

        Returns:
            Created position
        """
        # Check if position already exists
        if symbol in self.positions:
            return self.modify_position(symbol, quantity, entry_price)

        cost = quantity * entry_price
        if side == 'long' and cost > self.cash:
            logger.warning(f"Insufficient cash for {symbol} long position")
            return None

        position = Position(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            current_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

        self.positions[symbol] = position
        if side == 'long':
            self.cash -= cost

        logger.info(f"Opened {side} position: {quantity} {symbol} @ {entry_price}")
        return position

    def modify_position(
        self,
        symbol: str,
        quantity_change: float,
        price: float
    ) -> Optional[Position]:
        """Modify existing position"""
        if symbol not in self.positions:
            return None

        position = self.positions[symbol]

        # Calculate new quantity
        if position.side == 'long':
            new_quantity = position.quantity + quantity_change
        else:
            new_quantity = position.quantity - quantity_change

        if new_quantity <= 0:
            return self.close_position(symbol, price)

        # Update average entry price
        if quantity_change > 0:
            total_cost = position.quantity * position.entry_price + abs(quantity_change) * price
            position.entry_price = total_cost / (position.quantity + abs(quantity_change))

        position.quantity = abs(new_quantity)
        position.update_price(price)

        return position

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        partial_quantity: Optional[float] = None
    ) -> Optional[Position]:
        """
        Close a position (fully or partially).

        Args:
            symbol: Trading symbol
            exit_price: Exit price
            partial_quantity: Quantity to close (None for full close)

        Returns:
            Closed position
        """
        if symbol not in self.positions:
            return None

        position = self.positions[symbol]
        close_quantity = partial_quantity or position.quantity

        # Calculate realized P&L
        if position.side == 'long':
            realized_pnl = (exit_price - position.entry_price) * close_quantity
            self.cash += close_quantity * exit_price
        else:
            realized_pnl = (position.entry_price - exit_price) * close_quantity
            self.cash += realized_pnl + position.entry_price * close_quantity

        position.realized_pnl += realized_pnl

        # Record P&L
        self.pnl_history.append({
            'timestamp': datetime.utcnow(),
            'symbol': symbol,
            'side': position.side,
            'quantity': close_quantity,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'pnl': realized_pnl
        })

        # Handle partial vs full close
        if partial_quantity and partial_quantity < position.quantity:
            position.quantity -= close_quantity
            logger.info(f"Partially closed {symbol}: {close_quantity} @ {exit_price}, P&L: {realized_pnl:.2f}")
            return position
        else:
            self.closed_positions.append(position)
            del self.positions[symbol]
            logger.info(f"Closed position {symbol} @ {exit_price}, P&L: {realized_pnl:.2f}")
            return position

    def update_prices(self, prices: Dict[str, float]):
        """Update all position prices"""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].update_price(price)

    def check_stop_loss_take_profit(
        self,
        prices: Dict[str, float]
    ) -> List[str]:
        """
        Check for stop loss or take profit triggers.

        Args:
            prices: Current prices

        Returns:
            List of symbols to close
        """
        to_close = []

        for symbol, position in self.positions.items():
            price = prices.get(symbol, position.current_price)

            if position.stop_loss:
                if position.side == 'long' and price <= position.stop_loss:
                    to_close.append(symbol)
                    logger.info(f"Stop loss triggered for {symbol} at {price}")
                elif position.side == 'short' and price >= position.stop_loss:
                    to_close.append(symbol)
                    logger.info(f"Stop loss triggered for {symbol} at {price}")

            if position.take_profit:
                if position.side == 'long' and price >= position.take_profit:
                    to_close.append(symbol)
                    logger.info(f"Take profit triggered for {symbol} at {price}")
                elif position.side == 'short' and price <= position.take_profit:
                    to_close.append(symbol)
                    logger.info(f"Take profit triggered for {symbol} at {price}")

        return to_close

    def get_portfolio_metrics(self) -> PortfolioMetrics:
        """Calculate portfolio metrics"""
        positions_value = sum(p.market_value for p in self.positions.values())
        unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values())
        realized_pnl = sum(p['pnl'] for p in self.pnl_history)

        long_value = sum(p.market_value for p in self.positions.values() if p.side == 'long')
        short_value = sum(p.market_value for p in self.positions.values() if p.side == 'short')

        total_value = self.cash + positions_value + unrealized_pnl

        # Calculate win rate
        if self.pnl_history:
            wins = sum(1 for p in self.pnl_history if p['pnl'] > 0)
            win_rate = wins / len(self.pnl_history)

            gross_profit = sum(p['pnl'] for p in self.pnl_history if p['pnl'] > 0)
            gross_loss = abs(sum(p['pnl'] for p in self.pnl_history if p['pnl'] < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        else:
            win_rate = 0
            profit_factor = 0

        # Simplified Sharpe ratio
        if len(self.pnl_history) > 1:
            pnls = [p['pnl'] for p in self.pnl_history]
            sharpe = np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(252 * 24)
        else:
            sharpe = 0

        return PortfolioMetrics(
            total_value=total_value,
            cash=self.cash,
            positions_value=positions_value,
            total_pnl=realized_pnl + unrealized_pnl,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            long_exposure=long_value / total_value if total_value > 0 else 0,
            short_exposure=short_value / total_value if total_value > 0 else 0,
            net_exposure=(long_value - short_value) / total_value if total_value > 0 else 0,
            gross_exposure=(long_value + short_value) / total_value if total_value > 0 else 0,
            num_positions=len(self.positions),
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe
        )

    def get_position_weights(self) -> Dict[str, float]:
        """Get current position weights"""
        total_value = self.cash + sum(p.market_value for p in self.positions.values())
        if total_value == 0:
            return {}

        weights = {}
        for symbol, position in self.positions.items():
            weight = position.market_value / total_value
            weights[symbol] = weight if position.side == 'long' else -weight

        return weights

    def save_state(self, filepath: str):
        """Save position manager state to file"""
        state = {
            'cash': self.cash,
            'initial_capital': self.initial_capital,
            'positions': {
                s: {
                    'side': p.side,
                    'quantity': p.quantity,
                    'entry_price': p.entry_price,
                    'stop_loss': p.stop_loss,
                    'take_profit': p.take_profit
                }
                for s, p in self.positions.items()
            },
            'pnl_history': self.pnl_history[-1000:],  # Keep last 1000
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, default=str)

    def load_state(self, filepath: str):
        """Load position manager state from file"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)

            self.cash = state['cash']
            self.initial_capital = state.get('initial_capital', self.initial_capital)
            self.pnl_history = state.get('pnl_history', [])

            for symbol, pos_data in state.get('positions', {}).items():
                self.positions[symbol] = Position(
                    symbol=symbol,
                    side=pos_data['side'],
                    quantity=pos_data['quantity'],
                    entry_price=pos_data['entry_price'],
                    stop_loss=pos_data.get('stop_loss'),
                    take_profit=pos_data.get('take_profit')
                )

            logger.info(f"Loaded state with {len(self.positions)} positions")

        except FileNotFoundError:
            logger.info("No saved state found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading state: {e}")
