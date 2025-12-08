"""
Options Data Collector - Deribit Options Data Integration
"""

import aiohttp
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import time

from .options_flow import OptionData

logger = logging.getLogger(__name__)


class OptionsDataCollector:
    """
    Collects options data from Deribit exchange.

    Features:
    - Options chain fetching
    - DVOL (Deribit Volatility Index) tracking
    - Real-time IV data
    - Historical volatility data
    """

    BASE_URL = "https://www.deribit.com/api/v2"
    TESTNET_URL = "https://test.deribit.com/api/v2"

    SUPPORTED_UNDERLYINGS = ['BTC', 'ETH']

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        testnet: bool = True,
        config=None
    ):
        """
        Initialize options data collector.

        Args:
            client_id: Deribit API client ID
            client_secret: Deribit API client secret
            testnet: Use testnet if True
            config: Optional configuration
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.testnet = testnet
        self.config = config

        self.base_url = self.TESTNET_URL if testnet else self.BASE_URL
        self._access_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None

    async def _authenticate(self) -> bool:
        """Authenticate with Deribit API"""
        if not self.client_id or not self.client_secret:
            logger.warning("No Deribit credentials provided, using public endpoints")
            return False

        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    'client_id': self.client_id,
                    'client_secret': self.client_secret,
                    'grant_type': 'client_credentials'
                }
                async with session.get(
                    f"{self.base_url}/public/auth",
                    params=params
                ) as response:
                    data = await response.json()

                    if 'result' in data:
                        self._access_token = data['result']['access_token']
                        expires_in = data['result'].get('expires_in', 3600)
                        self._token_expiry = datetime.utcnow() + timedelta(seconds=expires_in - 60)
                        return True

                    logger.error(f"Auth failed: {data.get('error')}")
                    return False

        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False

    async def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        authenticated: bool = False
    ) -> Optional[Dict]:
        """Make API request"""
        try:
            headers = {}
            if authenticated and self._access_token:
                headers['Authorization'] = f"Bearer {self._access_token}"

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/{endpoint}",
                    params=params,
                    headers=headers
                ) as response:
                    data = await response.json()

                    if 'result' in data:
                        return data['result']
                    elif 'error' in data:
                        logger.error(f"API error: {data['error']}")
                        return None

                    return data

        except Exception as e:
            logger.error(f"Request error: {e}")
            return None

    async def fetch_options_chain(
        self,
        underlying: str,
        currency: str = 'USD'
    ) -> List[OptionData]:
        """
        Fetch complete options chain for an underlying.

        Args:
            underlying: 'BTC' or 'ETH'
            currency: Quote currency

        Returns:
            List of OptionData objects
        """
        if underlying not in self.SUPPORTED_UNDERLYINGS:
            logger.warning(f"Unsupported underlying: {underlying}")
            return []

        # Get all instruments
        instruments = await self._make_request(
            'public/get_instruments',
            {'currency': underlying, 'kind': 'option'}
        )

        if not instruments:
            return []

        options = []
        for inst in instruments:
            try:
                # Parse instrument name: BTC-31DEC21-50000-C
                parts = inst['instrument_name'].split('-')
                if len(parts) < 4:
                    continue

                expiry_str = parts[1]
                strike = float(parts[2])
                option_type = 'call' if parts[3] == 'C' else 'put'

                # Parse expiry
                try:
                    expiry = datetime.strptime(expiry_str, '%d%b%y')
                except ValueError:
                    continue

                # Get ticker data
                ticker = await self._make_request(
                    'public/ticker',
                    {'instrument_name': inst['instrument_name']}
                )

                if not ticker:
                    continue

                option = OptionData(
                    symbol=inst['instrument_name'],
                    underlying=underlying,
                    strike=strike,
                    expiry=expiry,
                    option_type=option_type,
                    price=ticker.get('last_price', 0) or 0,
                    bid=ticker.get('best_bid_price', 0) or 0,
                    ask=ticker.get('best_ask_price', 0) or 0,
                    volume=int(ticker.get('stats', {}).get('volume', 0) or 0),
                    open_interest=int(ticker.get('open_interest', 0) or 0),
                    implied_volatility=ticker.get('mark_iv', 0) / 100 if ticker.get('mark_iv') else 0,
                    delta=ticker.get('greeks', {}).get('delta', 0) or 0,
                    gamma=ticker.get('greeks', {}).get('gamma', 0) or 0,
                    theta=ticker.get('greeks', {}).get('theta', 0) or 0,
                    vega=ticker.get('greeks', {}).get('vega', 0) or 0
                )

                options.append(option)

            except Exception as e:
                logger.debug(f"Error parsing instrument: {e}")
                continue

        logger.info(f"Fetched {len(options)} options for {underlying}")
        return options

    async def fetch_dvol(self, underlying: str = 'BTC') -> Dict[str, float]:
        """
        Fetch DVOL (Deribit Volatility Index).

        Args:
            underlying: 'BTC' or 'ETH'

        Returns:
            Dict with DVOL data
        """
        index_name = f"{underlying.lower()}_usd"

        data = await self._make_request(
            'public/get_volatility_index_data',
            {'currency': underlying, 'resolution': '1D', 'count': 30}
        )

        if not data or 'data' not in data:
            return {'dvol': 0, 'dvol_24h_change': 0}

        # Get current DVOL
        current_data = data['data'][-1] if data['data'] else [0, 0, 0, 0, 0]

        return {
            'dvol': current_data[4] / 100 if len(current_data) > 4 else 0,  # Annualized vol
            'dvol_open': current_data[1] / 100 if len(current_data) > 1 else 0,
            'dvol_high': current_data[2] / 100 if len(current_data) > 2 else 0,
            'dvol_low': current_data[3] / 100 if len(current_data) > 3 else 0,
            'timestamp': current_data[0] if current_data else 0
        }

    async def fetch_index_price(self, underlying: str = 'BTC') -> float:
        """Get current index price"""
        index_name = f"{underlying.lower()}_usd"

        data = await self._make_request(
            'public/get_index_price',
            {'index_name': index_name}
        )

        return data.get('index_price', 0) if data else 0

    async def fetch_historical_volatility(
        self,
        underlying: str = 'BTC',
        period: int = 30
    ) -> Dict[str, float]:
        """
        Fetch historical volatility data.

        Args:
            underlying: 'BTC' or 'ETH'
            period: Period in days

        Returns:
            Dict with volatility metrics
        """
        data = await self._make_request(
            'public/get_historical_volatility',
            {'currency': underlying}
        )

        if not data:
            return {'hv': 0}

        # Data format: [[timestamp, volatility], ...]
        vols = [v[1] / 100 for v in data if len(v) > 1]

        return {
            'hv_current': vols[-1] if vols else 0,
            'hv_mean': sum(vols) / len(vols) if vols else 0,
            'hv_min': min(vols) if vols else 0,
            'hv_max': max(vols) if vols else 0
        }

    async def get_atm_options(
        self,
        underlying: str = 'BTC',
        days_to_expiry: int = 7
    ) -> Dict[str, OptionData]:
        """
        Get ATM call and put options.

        Args:
            underlying: 'BTC' or 'ETH'
            days_to_expiry: Target days to expiry

        Returns:
            Dict with 'call' and 'put' OptionData
        """
        spot_price = await self.fetch_index_price(underlying)
        options = await self.fetch_options_chain(underlying)

        if not options or spot_price == 0:
            return {}

        target_expiry = datetime.utcnow() + timedelta(days=days_to_expiry)

        # Find closest expiry
        expiries = sorted(set(o.expiry for o in options))
        closest_expiry = min(expiries, key=lambda x: abs((x - target_expiry).days))

        # Filter to closest expiry
        expiry_options = [o for o in options if o.expiry == closest_expiry]

        # Find ATM options
        atm_call = min(
            [o for o in expiry_options if o.option_type == 'call'],
            key=lambda x: abs(x.strike - spot_price),
            default=None
        )
        atm_put = min(
            [o for o in expiry_options if o.option_type == 'put'],
            key=lambda x: abs(x.strike - spot_price),
            default=None
        )

        result = {}
        if atm_call:
            result['call'] = atm_call
        if atm_put:
            result['put'] = atm_put

        return result

    def fetch_options_chain_sync(self, underlying: str) -> List[OptionData]:
        """Synchronous wrapper for fetch_options_chain"""
        return asyncio.run(self.fetch_options_chain(underlying))

    def fetch_dvol_sync(self, underlying: str = 'BTC') -> Dict[str, float]:
        """Synchronous wrapper for fetch_dvol"""
        return asyncio.run(self.fetch_dvol(underlying))
