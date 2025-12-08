"""
Lighter Exchange Client - Fast Execution DEX Integration
"""

import aiohttp
import asyncio
import hmac
import hashlib
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
import json

logger = logging.getLogger(__name__)


class LighterClient:
    """
    Client for Lighter Exchange DEX.

    Features:
    - Fast execution mode (1.5x fee for priority)
    - Market and limit orders
    - Position management
    - Low latency order placement
    """

    BASE_URL = "https://api.lighter.xyz"
    TESTNET_URL = "https://testnet-api.lighter.xyz"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        enable_fast_execution: bool = True,
        fast_execution_fee_multiplier: float = 1.5,
        testnet: bool = False
    ):
        """
        Initialize Lighter client.

        Args:
            api_key: API key
            api_secret: API secret
            enable_fast_execution: Enable fast execution mode
            fast_execution_fee_multiplier: Fee multiplier for fast execution
            testnet: Use testnet
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.enable_fast_execution = enable_fast_execution
        self.fast_execution_fee_multiplier = fast_execution_fee_multiplier
        self.testnet = testnet

        self.base_url = self.TESTNET_URL if testnet else self.BASE_URL

    def ping(self) -> bool:
        """Test connection to exchange"""
        try:
            return asyncio.run(self._ping_async())
        except Exception:
            return False

    async def _ping_async(self) -> bool:
        """Async ping test"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/health") as response:
                    return response.status == 200
        except Exception:
            return False

    def _generate_signature(self, timestamp: str, method: str, endpoint: str, body: str = "") -> str:
        """Generate HMAC signature for authenticated requests"""
        if not self.api_secret:
            return ""

        message = f"{timestamp}{method}{endpoint}{body}"
        signature = hmac.new(
            self.api_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()

        return signature

    def _get_headers(self, method: str, endpoint: str, body: str = "") -> Dict[str, str]:
        """Get headers for authenticated request"""
        timestamp = str(int(time.time() * 1000))
        signature = self._generate_signature(timestamp, method, endpoint, body)

        headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key or "",
            "X-Timestamp": timestamp,
            "X-Signature": signature
        }

        return headers

    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        endpoint = "/v1/account"
        headers = self._get_headers("GET", endpoint)

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}{endpoint}",
                headers=headers
            ) as response:
                if response.status == 200:
                    return await response.json()
                return {"error": f"Status {response.status}"}

    async def get_positions(self) -> List[Dict]:
        """Get current positions"""
        endpoint = "/v1/positions"
        headers = self._get_headers("GET", endpoint)

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}{endpoint}",
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("positions", [])
                return []

    async def get_balance(self) -> Dict[str, float]:
        """Get account balance"""
        account = await self.get_account_info()

        if "error" in account:
            return {}

        return {
            "equity": float(account.get("equity", 0)),
            "available": float(account.get("availableMargin", 0)),
            "margin_used": float(account.get("marginUsed", 0)),
            "unrealized_pnl": float(account.get("unrealizedPnl", 0))
        }

    async def get_orderbook(self, symbol: str, depth: int = 20) -> Dict[str, List]:
        """Get order book"""
        endpoint = f"/v1/orderbook/{symbol}"

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}{endpoint}",
                params={"depth": depth}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "bids": [[float(b["price"]), float(b["size"])] for b in data.get("bids", [])],
                        "asks": [[float(a["price"]), float(a["size"])] for a in data.get("asks", [])]
                    }
                return {"bids": [], "asks": []}

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get ticker data"""
        endpoint = f"/v1/ticker/{symbol}"

        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}{endpoint}") as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "symbol": symbol,
                        "last_price": float(data.get("lastPrice", 0)),
                        "bid": float(data.get("bestBid", 0)),
                        "ask": float(data.get("bestAsk", 0)),
                        "volume_24h": float(data.get("volume24h", 0)),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                return {}

    async def place_market_order(
        self,
        symbol: str,
        side: str,
        size: float,
        reduce_only: bool = False
    ) -> Dict[str, Any]:
        """
        Place a market order.

        Args:
            symbol: Trading symbol (e.g., 'BTC-PERP')
            side: 'buy' or 'sell'
            size: Order size
            reduce_only: Reduce only flag

        Returns:
            Order result
        """
        endpoint = "/v1/orders"

        order_data = {
            "symbol": symbol,
            "side": side.upper(),
            "type": "MARKET",
            "size": str(size),
            "reduceOnly": reduce_only,
            "fastExecution": self.enable_fast_execution
        }

        body = json.dumps(order_data)
        headers = self._get_headers("POST", endpoint, body)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}{endpoint}",
                headers=headers,
                data=body
            ) as response:
                data = await response.json()

                if response.status in [200, 201]:
                    logger.info(
                        f"Market order placed: {side} {size} {symbol}"
                        f"{' (fast)' if self.enable_fast_execution else ''}"
                    )
                    return {
                        "order_id": data.get("orderId"),
                        "status": "filled",
                        "filled_size": float(data.get("filledSize", size)),
                        "average_price": float(data.get("averagePrice", 0)),
                        "fees": float(data.get("fees", 0))
                    }

                logger.error(f"Order failed: {data}")
                return {"error": data.get("message", "Unknown error")}

    async def place_limit_order(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float,
        reduce_only: bool = False,
        post_only: bool = False,
        time_in_force: str = "GTC"
    ) -> Dict[str, Any]:
        """
        Place a limit order.

        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            size: Order size
            price: Limit price
            reduce_only: Reduce only flag
            post_only: Post-only flag
            time_in_force: GTC, IOC, FOK

        Returns:
            Order result
        """
        endpoint = "/v1/orders"

        order_data = {
            "symbol": symbol,
            "side": side.upper(),
            "type": "LIMIT",
            "size": str(size),
            "price": str(price),
            "reduceOnly": reduce_only,
            "postOnly": post_only,
            "timeInForce": time_in_force,
            "fastExecution": self.enable_fast_execution
        }

        body = json.dumps(order_data)
        headers = self._get_headers("POST", endpoint, body)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}{endpoint}",
                headers=headers,
                data=body
            ) as response:
                data = await response.json()

                if response.status in [200, 201]:
                    return {
                        "order_id": data.get("orderId"),
                        "status": data.get("status", "submitted"),
                        "filled_size": float(data.get("filledSize", 0)),
                        "average_price": float(data.get("averagePrice", 0))
                    }

                return {"error": data.get("message", "Unknown error")}

    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order"""
        endpoint = f"/v1/orders/{order_id}"
        headers = self._get_headers("DELETE", endpoint)

        async with aiohttp.ClientSession() as session:
            async with session.delete(
                f"{self.base_url}{endpoint}",
                headers=headers
            ) as response:
                if response.status == 200:
                    return {"status": "cancelled", "order_id": order_id}
                data = await response.json()
                return {"error": data.get("message", "Cancel failed")}

    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get order status"""
        endpoint = f"/v1/orders/{order_id}"
        headers = self._get_headers("GET", endpoint)

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}{endpoint}",
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "order_id": order_id,
                        "status": data.get("status"),
                        "filled_size": float(data.get("filledSize", 0)),
                        "average_price": float(data.get("averagePrice", 0)),
                        "remaining_size": float(data.get("remainingSize", 0))
                    }
                return {"error": "Order not found"}

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get open orders"""
        endpoint = "/v1/orders"
        params = {"status": "OPEN"}
        if symbol:
            params["symbol"] = symbol

        headers = self._get_headers("GET", endpoint)

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}{endpoint}",
                headers=headers,
                params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("orders", [])
                return []

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Cancel all open orders"""
        endpoint = "/v1/orders"
        params = {}
        if symbol:
            params["symbol"] = symbol

        headers = self._get_headers("DELETE", endpoint)

        async with aiohttp.ClientSession() as session:
            async with session.delete(
                f"{self.base_url}{endpoint}",
                headers=headers,
                params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {"cancelled_count": data.get("cancelledCount", 0)}
                return {"error": "Cancel all failed"}

    async def set_leverage(self, symbol: str, leverage: float) -> Dict[str, Any]:
        """Set leverage for a symbol"""
        endpoint = "/v1/leverage"

        body = json.dumps({"symbol": symbol, "leverage": leverage})
        headers = self._get_headers("POST", endpoint, body)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}{endpoint}",
                headers=headers,
                data=body
            ) as response:
                if response.status == 200:
                    return {"leverage": leverage, "symbol": symbol}
                data = await response.json()
                return {"error": data.get("message")}

    # Sync wrappers
    def get_positions_sync(self) -> List[Dict]:
        return asyncio.run(self.get_positions())

    def get_balance_sync(self) -> Dict[str, float]:
        return asyncio.run(self.get_balance())

    def place_market_order_sync(
        self,
        symbol: str,
        side: str,
        size: float
    ) -> Dict[str, Any]:
        return asyncio.run(self.place_market_order(symbol, side, size))

    def ping_sync(self) -> bool:
        return self.ping()
