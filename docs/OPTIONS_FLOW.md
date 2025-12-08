# Options Flow Analysis Guide

How the system uses options data for improved trading decisions.

## Overview

The system integrates BTC/ETH options data from Deribit to:
- Gauge market sentiment via put/call ratios
- Use implied volatility for position sizing
- Detect regime changes early
- Identify market stress levels

## Components

### OptionsFlowAnalyzer

Black-Scholes pricing and Greeks calculation:

```python
from ml_engine.options_flow import OptionsFlowAnalyzer

analyzer = OptionsFlowAnalyzer(risk_free_rate=0.05)

# Price an option
price = analyzer.black_scholes_price(
    S=50000,      # Spot price
    K=52000,      # Strike
    T=0.25,       # Time to expiry (years)
    r=0.05,       # Risk-free rate
    sigma=0.6,    # Volatility
    option_type='call'
)

# Calculate IV from market price
iv = analyzer.calculate_implied_volatility(
    market_price=3500,
    S=50000, K=52000, T=0.25, r=0.05,
    option_type='call'
)

# Full Greeks
greeks = analyzer.calculate_greeks(
    S=50000, K=52000, T=0.25, r=0.05, sigma=0.6
)
# Returns: delta, gamma, theta, vega, rho
```

### OptionsDataCollector

Fetches live options data from Deribit:

```python
from ml_engine.options_data_collector import OptionsDataCollector

collector = OptionsDataCollector(
    client_id='your_id',
    client_secret='your_secret',
    testnet=True
)

# Fetch BTC options chain
options = await collector.fetch_options_chain('BTC')

# Get DVOL (Deribit Volatility Index)
dvol = await collector.fetch_dvol('BTC')
print(f"DVOL: {dvol['dvol']:.1%}")

# Get ATM options
atm = await collector.get_atm_options('BTC', days_to_expiry=7)
```

## Key Metrics

### Put/Call Ratio (PCR)
- PCR > 1.2: Bearish sentiment (fear)
- PCR < 0.8: Bullish sentiment (greed)
- Used in regime detection

### IV Skew
- Positive skew: OTM puts expensive (crash protection demand)
- Negative skew: OTM calls expensive (upside speculation)
- Signals market sentiment asymmetry

### ATM Implied Volatility
- Compared to historical volatility
- High IV/HV ratio suggests overpriced options
- Used in volatility targeting

## Integration with Trading

### Volatility Weighting

The system uses a weighted combination of implied and historical volatility:

```python
# From config
IV_WEIGHT = 0.6  # 60% options IV, 40% historical

combined_vol = IV_WEIGHT * options_iv + (1 - IV_WEIGHT) * historical_vol
```

### Regime Detection

Options metrics feed into regime detection:

```python
def _analyze_options_sentiment(options_analysis):
    score = 0

    # Put/Call ratio
    pcr = options_analysis['put_call_ratio']
    if pcr > 1.5:
        score -= 0.5  # Bearish
    elif pcr < 0.7:
        score += 0.5  # Bullish

    # IV skew
    if iv_skew > 0.1:
        score -= 0.3  # Put skew = bearish

    return score
```

### Position Sizing

High IV environments trigger smaller positions:

```python
if atm_iv > 0.8:  # Very high IV
    position_size *= 0.5  # Reduce exposure
```

## Example Output

```
Options Analysis - BTC:
  ATM IV: 62.5%
  Put/Call Ratio: 1.15
  IV Skew: +3.2% (puts expensive)
  Max Pain: $48,000
  Sentiment: Slightly Bearish

Recommendation: Reduce long exposure, consider put protection
```

## Configuration

```env
OPTIONS_ENABLED=true
IV_WEIGHT=0.6
DERIBIT_CLIENT_ID=xxx
DERIBIT_CLIENT_SECRET=xxx
DERIBIT_TESTNET=true
```

## Limitations

- Options data only for BTC/ETH
- DVOL available hourly (not tick-by-tick)
- Weekend/holiday liquidity gaps
- Requires Deribit API access
