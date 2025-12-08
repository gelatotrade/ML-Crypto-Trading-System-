"""
Execution Engine - Smart Order Routing and Execution
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    TWAP = "twap"
    ICEBERG = "iceberg"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Order representation"""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0
    average_price: float = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    exchange_order_id: Optional[str] = None
    slippage: float = 0
    fees: float = 0


@dataclass
class ExecutionStats:
    """Execution statistics"""
    total_orders: int = 0
    filled_orders: int = 0
    rejected_orders: int = 0
    total_volume: float = 0
    total_slippage: float = 0
    total_fees: float = 0
    average_fill_time: float = 0


class ExecutionEngine:
    """
    Smart order execution engine.

    Features:
    - Multiple order types (market, limit, TWAP)
    - Slippage control
    - Order retry logic
    - Execution statistics
    """

    def __init__(
        self,
        dex_client=None,
        slippage_tolerance: float = 0.002,
        max_retries: int = 3,
        order_timeout: int = 30,
        config=None
    ):
        """
        Initialize execution engine.

        Args:
            dex_client: DEX client for order execution
            slippage_tolerance: Maximum acceptable slippage
            max_retries: Maximum order retry attempts
            order_timeout: Order timeout in seconds
            config: Optional configuration
        """
        self.dex_client = dex_client
        self.slippage_tolerance = slippage_tolerance
        self.max_retries = max_retries
        self.order_timeout = order_timeout
        self.config = config

        self.orders: Dict[str, Order] = {}
        self.stats = ExecutionStats()
        self._order_counter = 0
        self._paper_mode = True if config is None else config.is_paper_mode()

    def _generate_order_id(self) -> str:
        """Generate unique order ID"""
        self._order_counter += 1
        return f"ORD-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{self._order_counter:04d}"

    async def execute_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None
    ) -> Order:
        """
        Execute a single order.

        Args:
            symbol: Trading symbol
            side: Buy or sell
            quantity: Order quantity
            order_type: Type of order
            price: Limit price (for limit orders)

        Returns:
            Executed order
        """
        order = Order(
            id=self._generate_order_id(),
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price
        )

        self.orders[order.id] = order
        self.stats.total_orders += 1

        try:
            if self._paper_mode:
                order = await self._execute_paper(order)
            else:
                order = await self._execute_live(order)

            if order.status == OrderStatus.FILLED:
                self.stats.filled_orders += 1
                self.stats.total_volume += order.quantity * order.average_price
                self.stats.total_slippage += order.slippage
                self.stats.total_fees += order.fees

        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            order.status = OrderStatus.REJECTED
            self.stats.rejected_orders += 1

        return order

    async def _execute_paper(self, order: Order) -> Order:
        """Execute order in paper trading mode"""
        # Simulate execution delay
        await asyncio.sleep(0.1)

        # Simulate market price (would come from market data in real impl)
        simulated_price = order.price if order.price else 100.0

        # Simulate slippage
        slippage_pct = 0.001 if order.order_type == OrderType.MARKET else 0
        if order.side == OrderSide.BUY:
            fill_price = simulated_price * (1 + slippage_pct)
        else:
            fill_price = simulated_price * (1 - slippage_pct)

        order.filled_quantity = order.quantity
        order.average_price = fill_price
        order.slippage = abs(fill_price - simulated_price) / simulated_price
        order.fees = order.quantity * fill_price * 0.001  # 0.1% fee
        order.status = OrderStatus.FILLED
        order.updated_at = datetime.utcnow()

        logger.info(f"Paper order filled: {order.id} {order.side.value} {order.quantity} {order.symbol} @ {fill_price:.4f}")

        return order

    async def _execute_live(self, order: Order) -> Order:
        """Execute order on live exchange"""
        if self.dex_client is None:
            raise ValueError("DEX client not configured for live trading")

        for attempt in range(self.max_retries):
            try:
                if order.order_type == OrderType.MARKET:
                    result = await self._execute_market_order(order)
                elif order.order_type == OrderType.LIMIT:
                    result = await self._execute_limit_order(order)
                elif order.order_type == OrderType.TWAP:
                    result = await self._execute_twap_order(order)
                else:
                    result = await self._execute_market_order(order)

                return result

            except Exception as e:
                logger.warning(f"Order attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))

        order.status = OrderStatus.REJECTED
        return order

    async def _execute_market_order(self, order: Order) -> Order:
        """Execute market order"""
        result = await self.dex_client.place_market_order(
            symbol=order.symbol,
            side=order.side.value,
            size=order.quantity
        )

        order.exchange_order_id = result.get('order_id')
        order.filled_quantity = result.get('filled_size', order.quantity)
        order.average_price = result.get('average_price', 0)
        order.fees = result.get('fees', 0)
        order.status = OrderStatus.FILLED if order.filled_quantity >= order.quantity else OrderStatus.PARTIAL
        order.updated_at = datetime.utcnow()

        return order

    async def _execute_limit_order(self, order: Order) -> Order:
        """Execute limit order"""
        result = await self.dex_client.place_limit_order(
            symbol=order.symbol,
            side=order.side.value,
            size=order.quantity,
            price=order.price
        )

        order.exchange_order_id = result.get('order_id')
        order.status = OrderStatus.SUBMITTED
        order.updated_at = datetime.utcnow()

        # Wait for fill or timeout
        start_time = time.time()
        while time.time() - start_time < self.order_timeout:
            status = await self.dex_client.get_order_status(order.exchange_order_id)
            if status.get('status') == 'filled':
                order.filled_quantity = status.get('filled_size', order.quantity)
                order.average_price = status.get('average_price', order.price)
                order.status = OrderStatus.FILLED
                break
            await asyncio.sleep(1)

        if order.status != OrderStatus.FILLED:
            await self.dex_client.cancel_order(order.exchange_order_id)
            order.status = OrderStatus.CANCELLED

        return order

    async def _execute_twap_order(
        self,
        order: Order,
        num_slices: int = 5,
        interval: int = 60
    ) -> Order:
        """Execute TWAP (Time-Weighted Average Price) order"""
        slice_size = order.quantity / num_slices
        total_filled = 0
        total_value = 0

        for i in range(num_slices):
            slice_order = Order(
                id=f"{order.id}-{i}",
                symbol=order.symbol,
                side=order.side,
                order_type=OrderType.MARKET,
                quantity=slice_size
            )

            result = await self._execute_market_order(slice_order)

            if result.status == OrderStatus.FILLED:
                total_filled += result.filled_quantity
                total_value += result.filled_quantity * result.average_price

            if i < num_slices - 1:
                await asyncio.sleep(interval)

        order.filled_quantity = total_filled
        order.average_price = total_value / total_filled if total_filled > 0 else 0
        order.status = OrderStatus.FILLED if total_filled >= order.quantity * 0.95 else OrderStatus.PARTIAL
        order.updated_at = datetime.utcnow()

        return order

    async def execute_trades(
        self,
        trades: Dict[str, float],
        prices: Dict[str, float]
    ) -> List[Order]:
        """
        Execute multiple trades.

        Args:
            trades: Dict of symbol -> target weight change
            prices: Current prices

        Returns:
            List of executed orders
        """
        orders = []

        for symbol, weight_change in trades.items():
            if abs(weight_change) < 0.001:
                continue

            side = OrderSide.BUY if weight_change > 0 else OrderSide.SELL
            quantity = abs(weight_change)  # Simplified - would calculate from portfolio value

            order = await self.execute_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=OrderType.MARKET
            )
            orders.append(order)

        return orders

    def get_execution_stats(self) -> ExecutionStats:
        """Get execution statistics"""
        return self.stats

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self.orders.get(order_id)

    def get_open_orders(self) -> List[Order]:
        """Get all open orders"""
        return [o for o in self.orders.values()
                if o.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL]]

    async def cancel_all_orders(self) -> int:
        """Cancel all open orders"""
        cancelled = 0
        for order in self.get_open_orders():
            try:
                if self.dex_client and order.exchange_order_id:
                    await self.dex_client.cancel_order(order.exchange_order_id)
                order.status = OrderStatus.CANCELLED
                cancelled += 1
            except Exception as e:
                logger.error(f"Failed to cancel order {order.id}: {e}")
        return cancelled
