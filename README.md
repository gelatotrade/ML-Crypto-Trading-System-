# ML Crypto Trading System

A comprehensive machine learning-based crypto trading system implementing a market-neutral long/short strategy with risk-adjusted position sizing.

## Key Features

- **ML-Driven Predictions**: LSTM + Random Forest + Gradient Boosting ensemble
- **Market Neutrality**: Net exposure maintained < 10%
- **Dynamic Beta**: Regime-based beta adjustment (+0.3 risk-on, -0.3 risk-off)
- **Correlation Screening**: Prevents longing highly correlated assets
- **Options Flow Analysis**: Black-Scholes IV, put/call ratios, sentiment
- **Target Sharpe Ratio**: > 2.5 optimization
- **100+ Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, OBV, VWAP, and more

## Project Structure

```
ML-Crypto-Trading-System/
├── ml_engine/                      # Core ML components
│   ├── config.py                   # Configuration management
│   ├── data_collector.py           # CCXT data collection
│   ├── market_data_aggregator.py   # Multi-source aggregation
│   ├── feature_engineering.py      # 100+ technical indicators
│   ├── risk_models.py              # VaR, CVaR, GARCH volatility
│   ├── portfolio_optimizer.py      # Sharpe optimization + beta
│   ├── ml_models.py                # LSTM + ensemble models
│   ├── signal_generator.py         # Kelly criterion signals
│   ├── execution_engine.py         # Smart order routing
│   ├── position_manager.py         # Position tracking & P&L
│   ├── margin_manager.py           # Leverage optimization
│   ├── correlation_screener.py     # Prevent correlated positions
│   ├── options_flow.py             # Black-Scholes + Greeks
│   ├── options_data_collector.py   # Deribit options data
│   └── regime_detector.py          # Risk-on/off detection
├── dex_clients/
│   ├── hyperliquid_client.py       # Hyperliquid DEX
│   └── lighter_client.py           # Lighter Exchange (fast exec)
├── docs/
│   ├── API_SETUP.md                # API key setup guide
│   ├── OPTIONS_FLOW.md             # Options analysis docs
│   └── DYNAMIC_BETA.md             # Regime-based beta docs
├── examples/
│   └── options_flow_example.py     # Options analysis demo
├── ml_trading_bot.py               # Main orchestrator
├── .env.example                    # Configuration template
└── requirements.txt                # Dependencies
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.example .env
nano .env
# Add your API keys (see docs/API_SETUP.md)
```

Minimum required:
```
TRADING_MODE=paper
YAHOO_FINANCE_ENABLED=true
```

### 3. Run in Paper Mode

```bash
python ml_trading_bot.py
```

## Trading Strategy

### Market Neutrality
- Net Exposure: < 10%
- Gross Exposure: < 200%
- Long/Short: Balanced positions

### Dynamic Beta Management

| Regime | Beta | Strategy |
|--------|------|----------|
| Risk-On | +0.3 | Long bias, capture upside |
| Risk-Off | -0.3 | Short bias, hedge downside |
| Neutral | 0.0 | Pure market neutral |

### Correlation Screening

The system prevents taking same-direction positions in highly correlated assets:
- Maximum correlation threshold: 0.7
- Clusters highly correlated assets
- Keeps strongest signal in correlated groups
- Suggests hedges with negative correlations

### Asset Universe

```python
# Major assets (full position limits: 20%)
"BTC/USDT", "ETH/USDT"

# Mid-cap altcoins (10% limits)
"SOL/USDT", "AVAX/USDT", "MATIC/USDT", "LINK/USDT"

# Smaller altcoins (10% limits, vol-adjusted)
"UNI/USDT", "AAVE/USDT", "SUSHI/USDT", "CRV/USDT",
"LDO/USDT", "ARB/USDT", "OP/USDT", "IMX/USDT"

# Hedge asset (30% limits)
"HYPE/USDT"
```

## Risk Parameters

```python
MAX_LEVERAGE = 3.0
MAX_NET_EXPOSURE = 0.1          # 10% max for neutrality
MAX_GROSS_EXPOSURE = 2.0        # 200% max
TARGET_VOLATILITY = 0.15        # 15% annualized
MAX_DRAWDOWN = 0.15
MIN_SHARPE_RATIO = 2.5
MAX_CORRELATION = 0.7           # For same-direction positions
```

## Data Sources

### Trading Exchanges
- **Lighter Exchange** (Primary): Fast execution with 1.5x fee priority
- **Hyperliquid** (Backup): Reliable perpetual DEX

### Market Data
- Yahoo Finance (free, no key)
- Polygon.io (generous free tier)
- Alpha Vantage (500 req/day free)
- Financial Modeling Prep

### Options Data
- Deribit: BTC/ETH options chains, DVOL index

## Testing

```bash
# Test market data
python -c "
from ml_engine.market_data_aggregator import MarketDataAggregator
import os
from dotenv import load_dotenv
load_dotenv()
agg = MarketDataAggregator(dict(os.environ))
df = agg.fetch_aggregated_data('BTC/USDT')
print(f'Fetched {len(df)} rows')
"

# Run options example
python examples/options_flow_example.py
```

## Configuration

See `.env.example` for all configuration options:
- Trading mode (paper/live)
- Risk parameters
- API keys
- ML model settings

## Documentation

- [API Setup Guide](docs/API_SETUP.md)
- [Options Flow Analysis](docs/OPTIONS_FLOW.md)
- [Dynamic Beta Management](docs/DYNAMIC_BETA.md)

## Risk Warnings

- Start with paper trading
- High Sharpe target (2.5+) is ambitious
- Max 3x leverage - use carefully
- Monitor correlation exposures
- Options data requires Deribit API

## License

MIT License
