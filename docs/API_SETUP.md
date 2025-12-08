# API Setup Guide

Complete guide for setting up API keys for the ML Crypto Trading System.

## Quick Start

```bash
cp .env.example .env
nano .env  # Add your API keys
```

## Trading Exchanges

### Lighter Exchange (Primary)

**Why Lighter?**
- Fast execution with priority queue (1.5x fee)
- Low latency critical for ML strategies
- Good liquidity for major pairs

**Setup:**
1. Go to https://app.lighter.xyz/apikeys
2. Create new API key
3. Add to `.env`:
```
LIGHTER_API_KEY=your_key
LIGHTER_API_SECRET=your_secret
LIGHTER_ENABLE_FAST_EXECUTION=true
```

### Hyperliquid (Backup)

**Setup:**
1. Go to https://app.hyperliquid.xyz/API
2. Generate API credentials
3. Export your private key (for signing)
4. Add to `.env`:
```
HYPERLIQUID_API_KEY=your_key
HYPERLIQUID_API_SECRET=your_secret
DEX_PRIVATE_KEY=0x...your_private_key
```

## Market Data Sources

### Yahoo Finance (Recommended - Free)
No API key needed! Enabled by default.
```
YAHOO_FINANCE_ENABLED=true
```

### Alpha Vantage
- Free tier: 5 requests/minute, 500/day
- Get key: https://www.alphavantage.co/support/#api-key
```
ALPHA_VANTAGE_API_KEY=your_key
```

### Polygon.io
- Generous free tier
- Real-time + historical data
- Get key: https://polygon.io/
```
POLYGON_API_KEY=your_key
```

### Financial Modeling Prep
- Free tier: 250 requests/day
- Get key: https://site.financialmodelingprep.com/developer/docs
```
FMP_API_KEY=your_key
```

## Options Data (Deribit)

For IV analysis and options flow:
1. Create account at https://www.deribit.com
2. Go to API settings
3. Create new API key with read permissions
```
DERIBIT_CLIENT_ID=your_client_id
DERIBIT_CLIENT_SECRET=your_client_secret
DERIBIT_TESTNET=true  # Use false for mainnet
```

## Minimum Required Setup

For basic functionality, you only need:
```
TRADING_MODE=paper
YAHOO_FINANCE_ENABLED=true
```

This allows paper trading with Yahoo Finance data.

## Recommended Setup

For full functionality:
```
# Trading
LIGHTER_API_KEY=xxx
LIGHTER_API_SECRET=xxx

# Data (multiple sources for redundancy)
YAHOO_FINANCE_ENABLED=true
POLYGON_API_KEY=xxx

# Options
DERIBIT_CLIENT_ID=xxx
DERIBIT_CLIENT_SECRET=xxx
```

## Testing Your Setup

```python
# Test market data
python -c "
from ml_engine.market_data_aggregator import MarketDataAggregator
import os
from dotenv import load_dotenv
load_dotenv()
agg = MarketDataAggregator(dict(os.environ))
df = agg.fetch_aggregated_data('BTC/USDT')
print(f'✅ Fetched {len(df)} rows')
"

# Test Lighter connection
python -c "
from dex_clients.lighter_client import LighterClient
import os
from dotenv import load_dotenv
load_dotenv()
client = LighterClient(
    os.getenv('LIGHTER_API_KEY'),
    os.getenv('LIGHTER_API_SECRET')
)
print('✅ Connected' if client.ping() else '❌ Failed')
"
```

## Security Notes

- Never commit `.env` to git
- Use environment variables in production
- Rotate API keys periodically
- Use read-only keys where possible
- Enable IP whitelisting if available
