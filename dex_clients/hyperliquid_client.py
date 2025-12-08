"""
Hyperliquid DEX Client - Integration with Hyperliquid perpetual DEX
"""

import aiohttp
import asyncio
import hashlib
import hmac
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
from eth_account import Account
from eth_account.messages import encode_defunct

logger = logging.getLogger(__name__)


class HyperliquidClient:
    """
    Client for Hyperliquid DEX.

    Features:
    - Market and limit orders
    - Position management
    - Account info and balances
    - Order book data
    """

    MAINNET_URL = "https://api.hyperliquid.xyz"
    TESTNET_URL = "https://api.hyperliquid-testnet.xyz"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        private_key: Optional[str] = None,
        testnet: bool = True
    ):
        """
        Initialize Hyperliquid client.

        Args:
            api_key: API key (optional, for authenticated endpoints)
            api_secret: API secret
            private_key: Ethereum private key for signing
            testnet: Use testnet if True
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.private_key = private_key
        self.testnet = testnet

        self.base_url = self.TESTNET_URL if testnet else self.MAINNET_URL

        self._account = None
        if private_key:
            try:
                self._account = Account.from_key(private_key)
                self.address = self._account.address
            except Exception as e:
                logger.error(f"Failed to load private key: {e}")
                self.address = None
        else:
            self.address = None

    def ping(self) -> bool:
        """Test connection to exchange"""
        try:
            return asyncio.run(self._ping_async())
        except Exception:
            return False

    async def _ping_async(self) -> bool:
        """Async ping"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/info",
                    json={"type": "meta"}
                ) as response:
                    return response.status == 200
        except Exception:
            return False

    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        if not self.address:
            return {"error": "No address configured"}

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/info",
                json={"type": "clearinghouseState", "user": self.address}
            ) as response:
                return await response.json()

    async def get_positions(self) -> List[Dict]:
        """Get current positions"""
        account_info = await self.get_account_info()

        if "error" in account_info:
            return []

        positions = []
        for asset_position in account_info.get("assetPositions", []):
            pos = asset_position.get("position", {})
            if float(pos.get("szi", 0)) != 0:
                positions.append({
                    "symbol": pos.get("coin"),
                    "size": float(pos.get("szi", 0)),
                    "entry_price": float(pos.get("entryPx", 0)),
                    "unrealized_pnl": float(pos.get("unrealizedPnl", 0)),
                    "margin_used": float(pos.get("marginUsed", 0)),
                    "liquidation_price": float(pos.get("liquidationPx", 0) or 0)
                })

        return positions

    async def get_balance(self) -> Dict[str, float]:
        """Get account balance"""
        account_info = await self.get_account_info()

        if "error" in account_info:
            return {}

        margin_summary = account_info.get("marginSummary", {})
        return {
            "equity": float(margin_summary.get("accountValue", 0)),
            "available": float(margin_summary.get("totalRawUsd", 0)),
            "margin_used": float(margin_summary.get("totalMarginUsed", 0))
        }

    async def get_orderbook(self, symbol: str, depth: int = 20) -> Dict[str, List]:
        """Get order book for a symbol"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/info",
                json={"type": "l2Book", "coin": symbol}
            ) as response:
                data = await response.json()

                if "levels" not in data:
                    return {"bids": [], "asks": []}

                levels = data["levels"]
                return {
                    "bids": [[float(l["px"]), float(l["sz"])] for l in levels[0][:depth]],
                    "asks": [[float(l["px"]), float(l["sz"])] for l in levels[1][:depth]]
                }

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get ticker for a symbol"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/info",
                json={"type": "allMids"}
            ) as response:
                data = await response.json()

                if symbol in data:
                    return {
                        "symbol": symbol,
                        "mid_price": float(data[symbol]),
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
            symbol: Trading symbol
            side: 'buy' or 'sell'
            size: Order size
            reduce_only: Only reduce position

        Returns:
            Order result
        """
        if not self._account:
            return {"error": "No private key configured"}

        is_buy = side.lower() == "buy"

        # Get current price for slippage
        ticker = await self.get_ticker(symbol)
        if not ticker:
            return {"error": "Could not get price"}

        # Add slippage for market order
        slippage = 0.005  # 0.5%
        price = ticker["mid_price"]
        if is_buy:
            price = price * (1 + slippage)
        else:
            price = price * (1 - slippage)

        order = {
            "a": self._get_asset_id(symbol),
            "b": is_buy,
            "p": str(round(price, 2)),
            "s": str(size),
            "r": reduce_only,
            "t": {"limit": {"tif": "Ioc"}}  # Immediate or Cancel for market-like behavior
        }

        return await self._place_order(order)

    async def place_limit_order(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float,
        reduce_only: bool = False,
        post_only: bool = False
    ) -> Dict[str, Any]:
        """
        Place a limit order.

        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            size: Order size
            price: Limit price
            reduce_only: Only reduce position
            post_only: Post-only order

        Returns:
            Order result
        """
        if not self._account:
            return {"error": "No private key configured"}

        is_buy = side.lower() == "buy"
        tif = "Alo" if post_only else "Gtc"  # Add Liquidity Only or Good Till Cancel

        order = {
            "a": self._get_asset_id(symbol),
            "b": is_buy,
            "p": str(round(price, 2)),
            "s": str(size),
            "r": reduce_only,
            "t": {"limit": {"tif": tif}}
        }

        return await self._place_order(order)

    async def _place_order(self, order: Dict) -> Dict[str, Any]:
        """Internal order placement"""
        timestamp = int(time.time() * 1000)

        action = {
            "type": "order",
            "orders": [order],
            "grouping": "na"
        }

        # Sign the action
        signature = self._sign_action(action, timestamp)

        payload = {
            "action": action,
            "nonce": timestamp,
            "signature": signature
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/exchange",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                return await response.json()

    async def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Cancel an order"""
        if not self._account:
            return {"error": "No private key configured"}

        timestamp = int(time.time() * 1000)

        action = {
            "type": "cancel",
            "cancels": [{"a": self._get_asset_id(symbol), "o": int(order_id)}]
        }

        signature = self._sign_action(action, timestamp)

        payload = {
            "action": action,
            "nonce": timestamp,
            "signature": signature
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/exchange",
                json=payload
            ) as response:
                return await response.json()

    async def get_open_orders(self) -> List[Dict]:
        """Get all open orders"""
        if not self.address:
            return []

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/info",
                json={"type": "openOrders", "user": self.address}
            ) as response:
                data = await response.json()
                return data if isinstance(data, list) else []

    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get status of an order"""
        orders = await self.get_open_orders()
        for order in orders:
            if str(order.get("oid")) == str(order_id):
                return {"status": "open", **order}
        return {"status": "filled_or_cancelled"}

    def _sign_action(self, action: Dict, timestamp: int) -> Dict[str, str]:
        """Sign an action for the exchange"""
        if not self._account:
            return {}

        # Construct message to sign
        message = json.dumps({"action": action, "nonce": timestamp}, separators=(",", ":"))
        message_hash = encode_defunct(text=message)

        # Sign with private key
        signed = self._account.sign_message(message_hash)

        return {
            "r": hex(signed.r),
            "s": hex(signed.s),
            "v": signed.v
        }

    def _get_asset_id(self, symbol: str) -> int:
        """Convert symbol to asset ID"""
        # Simplified mapping - in production, fetch from exchange
        asset_map = {
            "BTC": 0, "ETH": 1, "SOL": 2, "AVAX": 3,
            "MATIC": 4, "LINK": 5, "UNI": 6, "AAVE": 7,
            "ARB": 8, "OP": 9
        }
        base = symbol.replace("/USDT", "").replace("-PERP", "")
        return asset_map.get(base, 0)

    # Sync wrappers
    def get_positions_sync(self) -> List[Dict]:
        return asyncio.run(self.get_positions())

    def get_balance_sync(self) -> Dict[str, float]:
        return asyncio.run(self.get_balance())

    def place_market_order_sync(self, symbol: str, side: str, size: float) -> Dict:
        return asyncio.run(self.place_market_order(symbol, side, size))
