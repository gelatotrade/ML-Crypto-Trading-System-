# ML Crypto Trading System

A comprehensive machine learning-based crypto trading system implementing a market-neutral long/short strategy with advanced tokenomics analysis.

## Key Features

- **ML-Driven Predictions**: LSTM + Random Forest + Gradient Boosting ensemble
- **Market Neutrality**: Net exposure maintained < 10%
- **Dynamic Beta**: Regime-based beta adjustment (+0.3 risk-on, -0.3 risk-off)
- **Inflation/Deflation Strategy**: Short high-inflation tokens, long deflationary assets
- **Token Unlock Analysis**: Track team/VC unlocks for short opportunities
- **Buyback Tracking**: Monitor HYPE buybacks for position sizing
- **Orderbook Analysis**: Real-time depth and whale activity detection
- **Correlation Screening**: Prevents longing highly correlated assets
- **Options Flow Analysis**: Black-Scholes IV, put/call ratios, sentiment
- **Target Sharpe Ratio**: > 2.5 optimization
- **100+ Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, OBV, VWAP, and more

---

## Project Structure

```
ML-Crypto-Trading-System/
├── ml_engine/                      # Core ML & strategy components
│   ├── config.py                   # Configuration & token universe
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
│   ├── regime_detector.py          # Risk-on/off detection
│   │
│   │   # NEW PIPELINES
│   ├── token_unlock_pipeline.py    # Token unlock analysis
│   ├── buyback_analyzer.py         # HYPE buyback tracking
│   ├── orderbook_pipeline.py       # Multi-exchange orderbook
│   ├── inflation_strategy.py       # Inflation/deflation signals
│   └── drift_protection.py         # Drift detection & mitigation
│
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

---

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

For full functionality:
```
# Token unlock data
TOKENOMIST_API_KEY=your_key
CRYPTORANK_API_KEY=your_key

# Trading
LIGHTER_API_KEY=your_key
LIGHTER_API_SECRET=your_secret
```

### 3. Run in Paper Mode

```bash
python ml_trading_bot.py
```

---

## Trading Strategy Overview

### Signal Combination (Multi-Factor)

The system combines multiple signal sources with configurable weights:

| Factor | Weight | Description |
|--------|--------|-------------|
| ML Predictions | 70% | LSTM + ensemble model predictions |
| Inflation/Deflation | 30% | Token unlock & buyback analysis |

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

---

## New Pipelines (Detailed)

### 1. Token Unlock Pipeline

**Purpose**: Identify tokens with upcoming large unlocks for short opportunities.

**Data Sources**:
- [Tokenomist.ai](https://tokenomist.ai) - Primary API for unlock schedules
- [CryptoRank.io](https://cryptorank.io/token-unlock) - Backup data source

**What It Tracks**:
- Team unlocks (founder, advisor vesting)
- Investor unlocks (Seed, Series A/B/C)
- Ecosystem/treasury unlocks
- Cliff endings (large one-time unlocks)

**How It Works**:

```python
from ml_engine.token_unlock_pipeline import TokenUnlockPipeline

pipeline = TokenUnlockPipeline(
    tokenomist_api_key="your_key",
    cryptorank_api_key="your_key"
)

# Fetch unlock data for a token
profile = await pipeline.fetch_token_unlocks("ARB")

# Analyze short opportunity
analysis = pipeline.analyze_short_opportunity(profile, current_price=1.50)

print(f"Short signal: {analysis.short_signal}")
print(f"7-day unlock: {analysis.unlock_7d_pct}%")
print(f"Investor ROI: {analysis.investor_roi * 100}%")
```

**Signal Generation**:

| Factor | Weight | Signal |
|--------|--------|--------|
| 7-day unlock > 5% | 40% | Strong short |
| 30-day unlock > 10% | 25% | Short bias |
| Investor ROI > 500% | 25% | Sell pressure likely |
| Team unlock soon | 10% | Additional pressure |

---

### 2. Buyback Analyzer Pipeline

**Purpose**: Track token buyback programs to optimize long positions.

**Primary Focus**: HYPE (Hyperliquid) buybacks from Assistance Fund

**Data Sources**:
- [HypurrScan.io](https://hypurrscan.io/dashboard) - HYPE buyback tracking
- [Tokenomist.ai](https://tokenomist.ai/hyperliquid/buyback) - Buyback history

**What It Tracks**:
- 24-hour buyback volume
- 7-day buyback trend
- Buyback vs average activity
- Fee-to-buyback conversion rate

**How It Works**:

```python
from ml_engine.buyback_analyzer import BuybackAnalyzer

analyzer = BuybackAnalyzer()

# Fetch HYPE buyback data
profile = await analyzer.fetch_hype_buybacks()

# Analyze for trading
analysis = analyzer.analyze_for_trading(profile, current_price=25.0)

print(f"Sentiment: {analysis.sentiment.value}")
print(f"Long signal: {analysis.long_signal}")
print(f"Position multiplier: {analysis.position_multiplier}x")
```

**Sentiment Levels**:

| Sentiment | Buyback Activity | Position Multiplier |
|-----------|------------------|---------------------|
| Very Bullish | 2x+ average | 1.5x |
| Bullish | Above average | 1.25x |
| Neutral | Normal | 1.0x |
| Bearish | Below average | 0.75x |
| Very Bearish | No activity | 0.5x |

**Important Note**: HYPE is NOT used as a hedge asset. Position sizing is based on buyback activity, with higher exposure during active buyback periods.

---

### 3. Orderbook Pipeline

**Purpose**: Real-time orderbook analysis for whale activity and support/resistance.

**Data Sources** (WebSocket + REST):
- **Bybit**: `wss://stream.bybit.com/v5/public/linear`
- **Binance**: `wss://fstream.binance.com/stream`
- **Hyperliquid**: `wss://api.hyperliquid.xyz/ws`
- **Lighter**: REST API polling

**What It Detects**:
- Large order blocks (> $100K)
- Whale orders (> $1M)
- Bid/ask imbalance
- Support/resistance walls
- Potential spoofing

**How It Works**:

```python
from ml_engine.orderbook_pipeline import OrderbookPipeline, Exchange

pipeline = OrderbookPipeline(
    symbols=["BTC/USDT", "ETH/USDT"],
    exchanges=[Exchange.BYBIT, Exchange.BINANCE],
    depth_levels=50
)

# Start WebSocket connections
await pipeline.start()

# Get current analysis
analysis = pipeline.get_current_analysis("BTC/USDT")

print(f"Bid depth: ${analysis.bid_depth_usd:,.0f}")
print(f"Ask depth: ${analysis.ask_depth_usd:,.0f}")
print(f"Imbalance signal: {analysis.imbalance_signal.value}")

if analysis.largest_bid_wall:
    print(f"Bid wall: ${analysis.largest_bid_wall.value_usd:,.0f} at {analysis.largest_bid_wall.price}")
```

**Imbalance Signals**:

| Signal | Depth Imbalance | Interpretation |
|--------|-----------------|----------------|
| Strong Buy | > +50% | Heavy bid support |
| Buy | > +20% | Moderate buy pressure |
| Neutral | -20% to +20% | Balanced book |
| Sell | < -20% | Moderate sell pressure |
| Strong Sell | < -50% | Heavy ask pressure |

---

### 4. Inflation/Deflation Strategy

**Purpose**: Combine unlock and buyback data for low-risk directional trades.

**Core Principle**:
- **SHORT** high-inflation tokens (meme coins, large unlocks)
- **LONG** deflationary tokens (active buybacks)

**How It Works**:

```python
from ml_engine.inflation_strategy import InflationDeflationStrategy

strategy = InflationDeflationStrategy()

# Analyze a symbol
signal = await strategy.analyze_symbol("ARB/USDT", current_price=1.50)

print(f"Type: {signal.inflation_type.value}")
print(f"Direction: {signal.direction.value}")
print(f"Combined signal: {signal.combined_signal}")
print(f"Recommended weight: {signal.recommended_weight}")
```

**Signal Weights**:

| Component | Weight | Description |
|-----------|--------|-------------|
| Unlock pressure | 40% | Token unlock sell pressure |
| Buyback support | 35% | Buyback program activity |
| Orderbook | 25% | Real-time depth analysis |

**Token Classification**:

| Type | Characteristics | Trading Approach |
|------|-----------------|------------------|
| `highly_inflationary` | > 5% monthly inflation | Strong short bias |
| `inflationary` | 2-5% monthly inflation | Short bias |
| `neutral` | < 2% inflation | ML-driven |
| `deflationary` | Active buybacks | Long during buybacks |
| `highly_deflationary` | Large buyback activity | Strong long bias |

---

## Asset Universe

### Token Classifications

```python
# Major assets (20% max position)
"BTC/USDT", "ETH/USDT"

# Lighter.xyz Perpetuals (10% max position)
"SOL/USDT", "AVAX/USDT", "LINK/USDT", "NEAR/USDT",
"DOT/USDT", "TON/USDT", "TAO/USDT", "POL/USDT"

# Meme Tokens - SHORT CANDIDATES (5% max position)
"DOGE/USDT", "PEPE/USDT", "WLD/USDT", "SHIB/USDT",
"BONK/USDT", "WIF/USDT", "FLOKI/USDT"

# High Unlock Tokens - SHORT CANDIDATES (8% max position)
"ARB/USDT", "OP/USDT", "APT/USDT", "SUI/USDT",
"SEI/USDT", "TIA/USDT", "JUP/USDT", "STRK/USDT"

# Deflationary Tokens - LONG CANDIDATES (15% max position)
"HYPE/USDT", "BNB/USDT"

# Other Altcoins (10% max position)
"UNI/USDT", "AAVE/USDT", "CRV/USDT", "LDO/USDT",
"IMX/USDT", "INJ/USDT", "FTM/USDT", "ATOM/USDT",
"FIL/USDT", "RUNE/USDT"
```

### Adding Custom Tokens

You can add custom tokens programmatically:

```python
from ml_engine.config import Config, AssetType

# Add a new token
Config.add_custom_token("NEWTOKEN/USDT", AssetType.ALTCOIN)

# Or add as a specific type
Config.add_custom_token("MEMETOKEN/USDT", AssetType.MEME)
Config.add_custom_token("UNLOCKING/USDT", AssetType.HIGH_UNLOCK)

# Remove a custom token
Config.remove_custom_token("NEWTOKEN/USDT")

# Get all tokens (default + custom)
all_tokens = Config.get_all_tokens()
```

---

## Risk Parameters

```python
# Position Limits by Asset Type
MAX_POSITION_MAJOR = 0.20        # 20% for BTC, ETH
MAX_POSITION_ALTCOIN = 0.10      # 10% for altcoins
MAX_POSITION_MEME = 0.05         # 5% for meme tokens (shorts only)
MAX_POSITION_DEFLATIONARY = 0.15 # 15% for buyback tokens
MAX_POSITION_HIGH_UNLOCK = 0.08  # 8% for high unlock tokens

# Portfolio Limits
MAX_LEVERAGE = 3.0
MAX_NET_EXPOSURE = 0.1           # 10% max for neutrality
MAX_GROSS_EXPOSURE = 2.0         # 200% max
TARGET_VOLATILITY = 0.15         # 15% annualized
MAX_DRAWDOWN = 0.15
MIN_SHARPE_RATIO = 2.5
MAX_CORRELATION = 0.7            # For same-direction positions
```

---

## Data Sources

### Market Data
| Source | Type | Rate Limit | Key Required |
|--------|------|------------|--------------|
| Yahoo Finance | Free | Unlimited | No |
| Polygon.io | Backup | Generous | Yes |
| Alpha Vantage | Fallback | 500/day | Yes |
| FMP | Fallback | 250/day | Yes |

### Token Unlock Data
| Source | Type | API Key |
|--------|------|---------|
| Tokenomist.ai | Primary | Required |
| CryptoRank.io | Backup | Required |

### Buyback Data
| Source | Token | API Key |
|--------|-------|---------|
| HypurrScan.io | HYPE | No |
| Tokenomist.ai | Various | Required |

### Orderbook Data
| Exchange | Protocol | Connection |
|----------|----------|------------|
| Bybit | WebSocket | Direct |
| Binance | WebSocket | Direct |
| Hyperliquid | WebSocket | Direct |
| Lighter | REST | Polling |

### Trading Exchanges
| Exchange | Type | Primary Use |
|----------|------|-------------|
| Lighter | Perp DEX | Execution |
| Hyperliquid | Perp DEX | Backup |

### Options Data
| Source | Tokens | Use |
|--------|--------|-----|
| Deribit | BTC, ETH | IV, sentiment |

---

## Configuration

### Environment Variables

```bash
# Trading Mode
TRADING_MODE=paper              # paper, live, backtest
PRIMARY_EXCHANGE=lighter
INITIAL_CAPITAL=100000

# Lighter Exchange
LIGHTER_API_KEY=your_key
LIGHTER_API_SECRET=your_secret

# Hyperliquid (Backup)
HYPERLIQUID_API_KEY=your_key
DEX_PRIVATE_KEY=0x...

# Token Unlock Data
TOKENOMIST_API_KEY=your_key
CRYPTORANK_API_KEY=your_key

# Market Data
YAHOO_FINANCE_ENABLED=true
POLYGON_API_KEY=your_key
ALPHA_VANTAGE_API_KEY=your_key

# Options Data
DERIBIT_CLIENT_ID=your_id
DERIBIT_CLIENT_SECRET=your_secret
OPTIONS_ENABLED=true

# Risk Parameters
MAX_LEVERAGE=3.0
MAX_NET_EXPOSURE=0.1
MIN_SHARPE_RATIO=2.5
```

---

## Testing

### Test Market Data

```bash
python -c "
from ml_engine.market_data_aggregator import MarketDataAggregator
import os
from dotenv import load_dotenv
load_dotenv()
agg = MarketDataAggregator(dict(os.environ))
df = agg.fetch_aggregated_data('BTC/USDT')
print(f'Fetched {len(df)} rows')
"
```

### Test Token Unlock Pipeline

```bash
python -c "
import asyncio
from ml_engine.token_unlock_pipeline import TokenUnlockPipeline

async def test():
    pipeline = TokenUnlockPipeline()
    profile = await pipeline.fetch_token_unlocks('ARB')
    print(f'Unlock risk: {profile.unlock_risk.value}')
    print(f'30-day inflation: {profile.inflation_30d}%')

asyncio.run(test())
"
```

### Test Buyback Analyzer

```bash
python -c "
import asyncio
from ml_engine.buyback_analyzer import BuybackAnalyzer

async def test():
    analyzer = BuybackAnalyzer()
    profile = await analyzer.fetch_hype_buybacks()
    print(f'24h buyback: \${profile.buyback_24h_usd:,.0f}')
    print(f'Sentiment: {profile.sentiment.value}')

asyncio.run(test())
"
```

---

## Trading Flow

```
1. MARKET DATA (async)
   ├─ Yahoo Finance / Polygon / FMP / Alpha Vantage
   ├─ CCXT (OHLCV, orderbook, ticker)
   └─ Deribit (options, IV, DVOL)

2. FEATURE ENGINEERING
   └─ 100+ technical indicators

3. REGIME DETECTION
   ├─ Trend (25%)
   ├─ Volatility (25%)
   ├─ Momentum (20%)
   ├─ Options sentiment (20%)
   └─ Correlation (10%)

4. ML PREDICTIONS (70% weight)
   ├─ LSTM neural network
   ├─ Random Forest
   └─ Gradient Boosting

5. INFLATION/DEFLATION ANALYSIS (30% weight)
   ├─ Token unlock pressure
   ├─ Buyback activity
   └─ Orderbook imbalance

6. SIGNAL GENERATION
   ├─ Combined ML + Inflation signals
   ├─ Kelly criterion sizing
   └─ Asset-type constraints

7. CORRELATION SCREENING
   └─ Block correlated same-direction positions

8. PORTFOLIO OPTIMIZATION
   ├─ Maximize Sharpe (target > 2.5)
   ├─ Constrain to target beta
   └─ Enforce exposure limits

9. EXECUTION
   ├─ Lighter (primary)
   └─ Hyperliquid (fallback)

10. POSITION MANAGEMENT
    └─ P&L tracking, metrics

11. DRIFT PROTECTION
    ├─ Statistical drift detection
    ├─ Performance monitoring
    └─ Automatic risk adjustment
```

---

## Drift Protection Pipeline (Detailed)

### What is Model Drift?

Model drift occurs when the relationship between input features and predictions changes over time, causing model performance to degrade. In crypto markets, drift is common due to:

- **Regime Changes**: Bull/bear market transitions
- **Structural Changes**: New market participants, regulatory changes
- **Black Swan Events**: Flash crashes, major hacks
- **Seasonality**: Different behavior in different market cycles

### Types of Drift

| Type | Description | Detection Method |
|------|-------------|------------------|
| **Data Drift** | Feature distributions change | KS-Test, PSI |
| **Concept Drift** | Feature-target relationship changes | Performance monitoring |
| **Performance Drift** | Model accuracy degrades | Sharpe, hit-rate tracking |
| **Regime Drift** | Market regime shifts | Regime detection |

### Drift Protection Components

#### 1. Statistical Drift Detection

The pipeline uses multiple statistical tests:

```python
from ml_engine.drift_protection import DriftProtectionPipeline, DriftConfig

# Initialize with custom thresholds
config = DriftConfig(
    ks_test_threshold=0.1,      # KS-Test threshold
    psi_threshold=0.2,          # Population Stability Index threshold
    sharpe_min_threshold=0.5,   # Minimum acceptable Sharpe
    drawdown_threshold=0.10,    # Max drawdown trigger
)

pipeline = DriftProtectionPipeline(config=config)

# Set baseline for a feature
pipeline.set_feature_baseline("BTC_returns", historical_returns)

# Update current values
pipeline.update_current_features("BTC_returns", recent_returns)

# Check for drift
report = pipeline.check_all_drift()

print(f"Severity: {report.overall_severity.value}")
print(f"Action: {report.recommended_action.value}")
print(f"Drifted features: {report.drifted_features}")
```

**Kolmogorov-Smirnov Test (KS-Test)**:
- Compares baseline vs. current feature distributions
- Threshold: 0.1 (configurable)
- Detects significant distribution shifts

**Population Stability Index (PSI)**:
- Measures distribution stability over time
- PSI < 0.1: Stable (no action)
- 0.1 ≤ PSI < 0.2: Moderate drift (monitor)
- PSI ≥ 0.2: Significant drift (retrain)

**CUSUM Test**:
- Detects change points in time series
- Good for sudden regime changes

#### 2. Performance Monitoring

Real-time tracking of key metrics:

```python
# Add return observations
pipeline.performance_monitor.add_return(0.02, datetime.utcnow())

# Add completed trade
pipeline.performance_monitor.add_trade({
    'timestamp': datetime.utcnow(),
    'symbol': 'BTC/USDT',
    'side': 'long',
    'pnl': 150.0,
    'pnl_pct': 0.015
})

# Calculate metrics
metrics = pipeline.performance_monitor.calculate_metrics(window_days=7)

print(f"Sharpe: {metrics.sharpe_ratio:.2f}")
print(f"Drawdown: {metrics.current_drawdown:.1%}")
print(f"Hit rate: {metrics.hit_rate:.1%}")
```

**Monitored Metrics**:
| Metric | Threshold | Action on Breach |
|--------|-----------|------------------|
| Sharpe Ratio | < 0.5 | Reduce positions |
| Sharpe Drop | > 0.5 from baseline | Retrain model |
| Max Drawdown | > 10% | Reduce positions |
| Hit Rate | < 45% | Increase monitoring |

#### 3. Automatic Risk Controls

When drift is detected, the system automatically adjusts:

```python
# Get position multiplier
multiplier = pipeline.get_position_multiplier()

# Severity -> Multiplier mapping:
# NONE:     1.0   (100% normal positions)
# LOW:      1.0   (continue monitoring)
# MEDIUM:   0.75  (25% reduction)
# HIGH:     0.5   (50% reduction)
# CRITICAL: 0.0   (halt trading)

# Apply to signals
adjusted_position = original_position * multiplier
```

**Severity Levels**:

| Severity | Position Multiplier | Recommended Action |
|----------|--------------------|--------------------|
| NONE | 1.0 | Continue normal trading |
| LOW | 1.0 | Monitor more frequently |
| MEDIUM | 0.75 | Schedule retraining |
| HIGH | 0.5 | Reduce exposure, retrain |
| CRITICAL | 0.0 | Halt trading immediately |

#### 4. Rolling Window Retraining

Models are automatically retrained when:
- Scheduled interval reached (default: 7 days)
- Performance drift detected
- Significant feature drift detected

```python
# Check if retraining needed
if pipeline.should_retrain():
    # Get recent data
    features, targets = pipeline.online_adapter.get_recent_data(
        window_hours=168  # 1 week
    )

    # Retrain models (implementation-specific)
    model.fit(features, targets)

    # Reset baseline
    pipeline.on_retraining_complete()
```

#### 5. Robust Feature Engineering

The pipeline includes drift-resistant features:

```python
from ml_engine.drift_protection import RobustFeatureEngineer

engineer = RobustFeatureEngineer()

# Create drift-resistant features
robust_features = engineer.create_robust_features(ohlcv_df, symbol="BTC/USDT")

# Features created:
# - risk_adj_return: Returns / volatility (more stable than absolute)
# - return_zscore: Z-normalized returns
# - return_percentile: Percentile rank (0-1, very stable)
# - vol_percentile: Volatility percentile
# - risk_adj_momentum: Momentum / volatility
```

**Why Robust Features?**
- Absolute returns drift with volatility regimes
- Risk-adjusted returns are more stable
- Percentile ranks invariant to distribution shifts
- Z-scores normalize for changing means/variances

#### 6. Regime-Specific Models

Train specialized models for different regimes:

```python
# Register regime-specific models
pipeline.register_regime_model("bull", bull_market_model)
pipeline.register_regime_model("bear", bear_market_model)
pipeline.register_regime_model("sideways", sideways_model)

# Switch regime when detected
pipeline.switch_regime("bear")

# Get current regime model
model = pipeline.get_regime_model(current_regime)
```

#### 7. Online Learning

Incremental model updates without full retraining:

```python
from ml_engine.drift_protection import OnlineLearningAdapter

adapter = OnlineLearningAdapter(config, base_model=model)

# Add new observation
adapter.add_observation(
    features=new_features,
    target=actual_return,
    timestamp=datetime.utcnow()
)

# Learning rate decays over time
lr = adapter.get_effective_learning_rate()

# Check if update needed
if adapter.should_update(datetime.utcnow()):
    recent_data = adapter.get_recent_data()
    # Perform incremental update
```

### Drift Monitoring Dashboard

Get a summary of drift status:

```python
summary = pipeline.create_monitoring_summary()

# Returns:
{
    "timestamp": "2024-12-09T10:30:00",
    "overall_severity": "medium",
    "recommended_action": "retrain",
    "position_multiplier": 0.75,
    "data_drift_score": 0.15,
    "drifted_features_count": 3,
    "stable_features_count": 12,
    "alerts_count": 2,
    "current_sharpe": 1.2,
    "current_drawdown": 0.05,
    "retraining_scheduled": true,
    "current_regime": "neutral",
    "recommendations": [
        "MEDIUM: Schedule model retraining",
        "Increase monitoring frequency",
        "Re-evaluate features: BTC_returns, ETH_volatility"
    ]
}
```

### Best Practices for Drift Prevention

1. **Use Robust Features**: Risk-adjusted and normalized features drift less
2. **Monitor Continuously**: Check drift at every iteration
3. **Retrain Regularly**: Even without drift, retrain weekly
4. **Multiple Models**: Use regime-specific models
5. **Hard Risk Limits**: Always have stop-losses as last defense
6. **Conservative on Drift**: Reduce positions early, don't wait for critical

### Configuration

```bash
# .env configuration for drift protection

# Statistical thresholds
DRIFT_KS_THRESHOLD=0.1
DRIFT_PSI_THRESHOLD=0.2

# Performance thresholds
DRIFT_SHARPE_MIN=0.5
DRIFT_SHARPE_DROP=0.5
DRIFT_DRAWDOWN_THRESHOLD=0.10
DRIFT_HIT_RATE_MIN=0.45

# Monitoring windows
DRIFT_BASELINE_DAYS=30
DRIFT_DETECTION_DAYS=7
DRIFT_RETRAIN_DAYS=7

# Risk controls
DRIFT_MAX_REDUCTION=0.5
DRIFT_HALT_ON_CRITICAL=true

# Online learning
DRIFT_ONLINE_LEARNING=true
DRIFT_LEARNING_RATE=0.01
```

---

## Documentation

- [API Setup Guide](docs/API_SETUP.md)
- [Options Flow Analysis](docs/OPTIONS_FLOW.md)
- [Dynamic Beta Management](docs/DYNAMIC_BETA.md)

---

## Risk Warnings

- Start with paper trading
- High Sharpe target (2.5+) is ambitious
- Max 3x leverage - use carefully
- Monitor correlation exposures
- Token unlock data may have delays
- Meme tokens are high volatility - small positions only
- Buyback analysis is supplementary, not guaranteed
- **Model drift is inevitable** - monitor drift alerts closely
- Halt trading if drift protection triggers CRITICAL severity
- Regular retraining is essential for long-term performance

---

## License

MIT License
