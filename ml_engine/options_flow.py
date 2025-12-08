"""
Options Flow Analyzer - Black-Scholes + IV Analysis
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class OptionData:
    """Single option contract data"""
    symbol: str
    underlying: str
    strike: float
    expiry: datetime
    option_type: str  # 'call' or 'put'
    price: float
    bid: float
    ask: float
    volume: int
    open_interest: int
    implied_volatility: float = 0
    delta: float = 0
    gamma: float = 0
    theta: float = 0
    vega: float = 0


@dataclass
class OptionsChainAnalysis:
    """Analysis results for options chain"""
    underlying: str
    spot_price: float
    atm_iv: float
    put_call_ratio: float
    iv_skew: float
    term_structure: Dict[str, float]
    max_pain: float
    total_call_oi: int
    total_put_oi: int
    sentiment: str  # 'bullish', 'bearish', 'neutral'


class OptionsFlowAnalyzer:
    """
    Options flow analysis with Black-Scholes pricing and IV calculation.

    Features:
    - Black-Scholes option pricing
    - Implied volatility calculation
    - Greeks calculation
    - Put/Call ratio analysis
    - IV skew and term structure
    - Max pain calculation
    """

    def __init__(self, risk_free_rate: float = 0.05, config=None):
        """
        Initialize options analyzer.

        Args:
            risk_free_rate: Annual risk-free rate
            config: Optional configuration
        """
        self.risk_free_rate = risk_free_rate
        self.config = config

    def black_scholes_price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = 'call'
    ) -> float:
        """
        Calculate Black-Scholes option price.

        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'

        Returns:
            Option price
        """
        if T <= 0 or sigma <= 0:
            return max(0, S - K) if option_type == 'call' else max(0, K - S)

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return price

    def calculate_implied_volatility(
        self,
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str = 'call'
    ) -> float:
        """
        Calculate implied volatility using Brent's method.

        Args:
            market_price: Market price of option
            S: Spot price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            option_type: 'call' or 'put'

        Returns:
            Implied volatility
        """
        if T <= 0 or market_price <= 0:
            return 0

        def objective(sigma):
            return self.black_scholes_price(S, K, T, r, sigma, option_type) - market_price

        try:
            iv = brentq(objective, 0.001, 5.0, xtol=1e-6)
            return iv
        except ValueError:
            # If no solution found, use Newton-Raphson approximation
            return self._newton_iv(market_price, S, K, T, r, option_type)

    def _newton_iv(
        self,
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str,
        max_iter: int = 100
    ) -> float:
        """Newton-Raphson IV calculation"""
        sigma = 0.3  # Initial guess

        for _ in range(max_iter):
            price = self.black_scholes_price(S, K, T, r, sigma, option_type)
            vega = self.calculate_vega(S, K, T, r, sigma)

            if vega < 1e-10:
                break

            sigma = sigma - (price - market_price) / vega

            if sigma <= 0:
                sigma = 0.001
            if abs(price - market_price) < 1e-6:
                break

        return max(0.001, sigma)

    def calculate_greeks(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = 'call'
    ) -> Dict[str, float]:
        """
        Calculate option Greeks.

        Returns:
            Dictionary with delta, gamma, theta, vega, rho
        """
        if T <= 0 or sigma <= 0:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # Delta
        if option_type == 'call':
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1

        # Gamma
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

        # Theta (per day)
        theta_term1 = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        if option_type == 'call':
            theta_term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
        else:
            theta_term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
        theta = (theta_term1 + theta_term2) / 365

        # Vega (per 1% change in vol)
        vega = S * np.sqrt(T) * norm.pdf(d1) / 100

        # Rho (per 1% change in rate)
        if option_type == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }

    def calculate_vega(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float
    ) -> float:
        """Calculate Vega for IV calculations"""
        if T <= 0 or sigma <= 0:
            return 0

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return S * np.sqrt(T) * norm.pdf(d1)

    def analyze_options_chain(
        self,
        options: List[OptionData],
        spot_price: float
    ) -> OptionsChainAnalysis:
        """
        Analyze an options chain.

        Args:
            options: List of option data
            spot_price: Current spot price

        Returns:
            OptionsChainAnalysis results
        """
        if not options:
            return OptionsChainAnalysis(
                underlying="",
                spot_price=spot_price,
                atm_iv=0,
                put_call_ratio=0,
                iv_skew=0,
                term_structure={},
                max_pain=spot_price,
                total_call_oi=0,
                total_put_oi=0,
                sentiment='neutral'
            )

        underlying = options[0].underlying
        calls = [o for o in options if o.option_type == 'call']
        puts = [o for o in options if o.option_type == 'put']

        # Calculate ATM IV
        atm_iv = self._calculate_atm_iv(options, spot_price)

        # Put/Call ratio
        total_call_oi = sum(o.open_interest for o in calls)
        total_put_oi = sum(o.open_interest for o in puts)
        pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 1

        # IV Skew (25-delta put IV - 25-delta call IV)
        iv_skew = self._calculate_iv_skew(options, spot_price)

        # Term structure
        term_structure = self._calculate_term_structure(options, spot_price)

        # Max pain
        max_pain = self._calculate_max_pain(options)

        # Sentiment
        if pcr > 1.2 or iv_skew > 0.05:
            sentiment = 'bearish'
        elif pcr < 0.8 or iv_skew < -0.05:
            sentiment = 'bullish'
        else:
            sentiment = 'neutral'

        return OptionsChainAnalysis(
            underlying=underlying,
            spot_price=spot_price,
            atm_iv=atm_iv,
            put_call_ratio=pcr,
            iv_skew=iv_skew,
            term_structure=term_structure,
            max_pain=max_pain,
            total_call_oi=total_call_oi,
            total_put_oi=total_put_oi,
            sentiment=sentiment
        )

    def _calculate_atm_iv(
        self,
        options: List[OptionData],
        spot_price: float
    ) -> float:
        """Calculate ATM implied volatility"""
        # Find options closest to spot
        min_dist = float('inf')
        atm_iv = 0

        for opt in options:
            dist = abs(opt.strike - spot_price)
            if dist < min_dist and opt.implied_volatility > 0:
                min_dist = dist
                atm_iv = opt.implied_volatility

        return atm_iv

    def _calculate_iv_skew(
        self,
        options: List[OptionData],
        spot_price: float
    ) -> float:
        """Calculate IV skew (OTM put IV - OTM call IV)"""
        otm_puts = [o for o in options if o.option_type == 'put' and o.strike < spot_price * 0.95]
        otm_calls = [o for o in options if o.option_type == 'call' and o.strike > spot_price * 1.05]

        put_iv = np.mean([o.implied_volatility for o in otm_puts]) if otm_puts else 0
        call_iv = np.mean([o.implied_volatility for o in otm_calls]) if otm_calls else 0

        return put_iv - call_iv

    def _calculate_term_structure(
        self,
        options: List[OptionData],
        spot_price: float
    ) -> Dict[str, float]:
        """Calculate IV term structure by expiry"""
        term_structure = {}

        # Group by expiry
        expiries = {}
        for opt in options:
            expiry_str = opt.expiry.strftime('%Y-%m-%d')
            if expiry_str not in expiries:
                expiries[expiry_str] = []
            expiries[expiry_str].append(opt)

        for expiry_str, opts in expiries.items():
            atm_iv = self._calculate_atm_iv(opts, spot_price)
            if atm_iv > 0:
                term_structure[expiry_str] = atm_iv

        return term_structure

    def _calculate_max_pain(self, options: List[OptionData]) -> float:
        """Calculate max pain strike"""
        strikes = sorted(set(o.strike for o in options))
        if not strikes:
            return 0

        min_pain = float('inf')
        max_pain_strike = strikes[0]

        for strike in strikes:
            pain = 0
            for opt in options:
                if opt.option_type == 'call' and opt.strike < strike:
                    pain += (strike - opt.strike) * opt.open_interest
                elif opt.option_type == 'put' and opt.strike > strike:
                    pain += (opt.strike - strike) * opt.open_interest

            if pain < min_pain:
                min_pain = pain
                max_pain_strike = strike

        return max_pain_strike

    def get_volatility_surface(
        self,
        options: List[OptionData],
        spot_price: float
    ) -> Dict[str, Dict[float, float]]:
        """
        Build implied volatility surface.

        Returns:
            Dict of {expiry: {strike: IV}}
        """
        surface = {}

        for opt in options:
            expiry_str = opt.expiry.strftime('%Y-%m-%d')
            if expiry_str not in surface:
                surface[expiry_str] = {}

            moneyness = opt.strike / spot_price
            surface[expiry_str][moneyness] = opt.implied_volatility

        return surface
