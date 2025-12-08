#!/usr/bin/env python3
"""
Options Flow Analysis Example

Demonstrates:
- Fetching options data from Deribit
- Calculating implied volatility
- Analyzing put/call ratios
- Using options for regime detection
"""

import asyncio
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from ml_engine.options_flow import OptionsFlowAnalyzer, OptionData
from ml_engine.options_data_collector import OptionsDataCollector

load_dotenv()


def demonstrate_black_scholes():
    """Demonstrate Black-Scholes pricing"""
    print("\n" + "="*50)
    print("BLACK-SCHOLES PRICING DEMO")
    print("="*50)

    analyzer = OptionsFlowAnalyzer(risk_free_rate=0.05)

    # Example: BTC at $50,000, 3-month call at $52,000 strike
    spot = 50000
    strike = 52000
    time_to_expiry = 0.25  # 3 months
    volatility = 0.60  # 60% annual vol

    # Price call and put
    call_price = analyzer.black_scholes_price(
        spot, strike, time_to_expiry, 0.05, volatility, 'call'
    )
    put_price = analyzer.black_scholes_price(
        spot, strike, time_to_expiry, 0.05, volatility, 'put'
    )

    print(f"\nSpot Price: ${spot:,}")
    print(f"Strike: ${strike:,}")
    print(f"Time to Expiry: {time_to_expiry*12:.0f} months")
    print(f"Volatility: {volatility:.0%}")
    print(f"\nCall Price: ${call_price:,.2f}")
    print(f"Put Price: ${put_price:,.2f}")

    # Calculate Greeks
    greeks = analyzer.calculate_greeks(
        spot, strike, time_to_expiry, 0.05, volatility, 'call'
    )

    print(f"\nCall Greeks:")
    print(f"  Delta: {greeks['delta']:.4f}")
    print(f"  Gamma: {greeks['gamma']:.6f}")
    print(f"  Theta: ${greeks['theta']:.2f}/day")
    print(f"  Vega: ${greeks['vega']:.2f}/1% vol")

    # Implied Volatility calculation
    market_price = 4000  # Assume market price
    implied_vol = analyzer.calculate_implied_volatility(
        market_price, spot, strike, time_to_expiry, 0.05, 'call'
    )
    print(f"\nImplied Vol from ${market_price:,} market price: {implied_vol:.1%}")


def demonstrate_chain_analysis():
    """Demonstrate options chain analysis"""
    print("\n" + "="*50)
    print("OPTIONS CHAIN ANALYSIS DEMO")
    print("="*50)

    analyzer = OptionsFlowAnalyzer()

    # Create sample options chain
    from datetime import datetime, timedelta
    expiry = datetime.utcnow() + timedelta(days=30)
    spot = 50000

    # Sample options
    options = [
        OptionData("BTC-CALL-48000", "BTC", 48000, expiry, "call", 3500, 3400, 3600, 100, 500, 0.58),
        OptionData("BTC-CALL-50000", "BTC", 50000, expiry, "call", 2500, 2400, 2600, 150, 800, 0.55),
        OptionData("BTC-CALL-52000", "BTC", 52000, expiry, "call", 1800, 1700, 1900, 80, 400, 0.52),
        OptionData("BTC-CALL-55000", "BTC", 55000, expiry, "call", 1000, 900, 1100, 50, 200, 0.58),
        OptionData("BTC-PUT-45000", "BTC", 45000, expiry, "put", 800, 750, 850, 120, 600, 0.62),
        OptionData("BTC-PUT-48000", "BTC", 48000, expiry, "put", 1500, 1400, 1600, 200, 1000, 0.60),
        OptionData("BTC-PUT-50000", "BTC", 50000, expiry, "put", 2500, 2400, 2600, 180, 900, 0.55),
        OptionData("BTC-PUT-52000", "BTC", 52000, expiry, "put", 3800, 3700, 3900, 90, 350, 0.53),
    ]

    # Analyze chain
    analysis = analyzer.analyze_options_chain(options, spot)

    print(f"\nAnalysis for {analysis.underlying} @ ${spot:,}")
    print(f"-" * 40)
    print(f"ATM Implied Volatility: {analysis.atm_iv:.1%}")
    print(f"Put/Call Ratio: {analysis.put_call_ratio:.2f}")
    print(f"IV Skew (OTM Put - OTM Call): {analysis.iv_skew:.1%}")
    print(f"Max Pain Strike: ${analysis.max_pain:,}")
    print(f"Total Call OI: {analysis.total_call_oi:,}")
    print(f"Total Put OI: {analysis.total_put_oi:,}")
    print(f"Sentiment: {analysis.sentiment.upper()}")

    # Interpretation
    print(f"\nInterpretation:")
    if analysis.put_call_ratio > 1.2:
        print("  - High P/C ratio suggests bearish sentiment")
    elif analysis.put_call_ratio < 0.8:
        print("  - Low P/C ratio suggests bullish sentiment")
    else:
        print("  - Neutral P/C ratio")

    if analysis.iv_skew > 0.05:
        print("  - Put skew indicates demand for downside protection")
    elif analysis.iv_skew < -0.05:
        print("  - Call skew indicates speculative upside interest")


async def demonstrate_live_data():
    """Demonstrate live Deribit data (requires API keys)"""
    print("\n" + "="*50)
    print("LIVE DERIBIT DATA DEMO")
    print("="*50)

    client_id = os.getenv('DERIBIT_CLIENT_ID')
    client_secret = os.getenv('DERIBIT_CLIENT_SECRET')

    if not client_id:
        print("\nNo Deribit credentials found.")
        print("Set DERIBIT_CLIENT_ID and DERIBIT_CLIENT_SECRET in .env")
        print("Using simulated data instead...")
        return

    collector = OptionsDataCollector(
        client_id=client_id,
        client_secret=client_secret,
        testnet=os.getenv('DERIBIT_TESTNET', 'true').lower() == 'true'
    )

    print("\nFetching BTC index price...")
    btc_price = await collector.fetch_index_price('BTC')
    print(f"BTC Index: ${btc_price:,.2f}")

    print("\nFetching DVOL...")
    dvol = await collector.fetch_dvol('BTC')
    print(f"BTC DVOL: {dvol.get('dvol', 0):.1%}")

    print("\nFetching historical volatility...")
    hv = await collector.fetch_historical_volatility('BTC')
    print(f"30-day HV: {hv.get('hv_current', 0):.1%}")

    print("\nFetching ATM options...")
    atm = await collector.get_atm_options('BTC', days_to_expiry=7)

    if 'call' in atm:
        print(f"\nATM Call:")
        print(f"  Strike: ${atm['call'].strike:,}")
        print(f"  IV: {atm['call'].implied_volatility:.1%}")
        print(f"  Delta: {atm['call'].delta:.3f}")

    if 'put' in atm:
        print(f"\nATM Put:")
        print(f"  Strike: ${atm['put'].strike:,}")
        print(f"  IV: {atm['put'].implied_volatility:.1%}")
        print(f"  Delta: {atm['put'].delta:.3f}")


def main():
    """Run all demonstrations"""
    print("="*50)
    print("OPTIONS FLOW ANALYSIS EXAMPLES")
    print("="*50)

    # Black-Scholes demo
    demonstrate_black_scholes()

    # Chain analysis demo
    demonstrate_chain_analysis()

    # Live data demo (if credentials available)
    asyncio.run(demonstrate_live_data())

    print("\n" + "="*50)
    print("DEMO COMPLETE")
    print("="*50)


if __name__ == "__main__":
    main()
