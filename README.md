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
│   │   # STRATEGY PIPELINES
│   ├── token_unlock_pipeline.py    # Token unlock analysis
│   ├── buyback_analyzer.py         # HYPE buyback tracking
│   ├── orderbook_pipeline.py       # Multi-exchange orderbook
│   ├── inflation_strategy.py       # Inflation/deflation signals
│   ├── drift_protection.py         # Drift detection & mitigation
│   │
│   │   # ADVANCED ML PIPELINES
│   ├── meta_pipeline.py            # HMM regime detection, Bayesian optimization
│   ├── advanced_data_pipeline.py   # Microstructure, NLP, SABR, stationarity
│   ├── alpha_pipeline.py           # GP regression, QRF, causal inference
│   ├── advanced_risk_pipeline.py   # Kalman betas, copulas, L-VaR
│   ├── advanced_portfolio_optimizer.py  # Black-Litterman, online convex
│   ├── execution_pipeline.py       # Multi-agent, RL execution
│   ├── monitoring_pipeline.py      # Attribution, PBO, stress testing
│   └── advanced_ml_optimizations.py # MAML, Bayesian DL, evidential
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

## Advanced ML Trading Pipelines

The system includes 8 advanced ML pipelines implementing state-of-the-art quantitative finance techniques based on the Adaptive Market Hypothesis (AMH). These pipelines work together to provide a comprehensive trading system that adapts to changing market conditions.

### Theoretical Foundation: Adaptive Market Hypothesis

Unlike the Efficient Market Hypothesis (EMH), AMH recognizes that market efficiency varies over time. Our system implements:

- **Regime-dependent strategies** that adapt to market conditions
- **Multi-timescale analysis** capturing different market dynamics
- **Continuous learning** to adapt to structural changes
- **Robust uncertainty quantification** for reliable risk assessment

---

### Pipeline 0: Meta-Pipeline (System Control & Adaptivity)

**File**: `ml_engine/meta_pipeline.py`

**Purpose**: Coordinates all subsystems and adapts the overall trading strategy based on market regimes.

#### Components

##### HMM Regime Detector (Hidden Markov Model)

Detects 7 market regimes using a 5-state Hidden Markov Model:

| Regime | Characteristics | Trading Approach |
|--------|-----------------|------------------|
| Bull Trending | High returns, moderate vol | Aggressive long |
| Bear Trending | Negative returns, high vol | Defensive/short |
| High Volatility | Extreme volatility | Reduce exposure |
| Low Volatility | Compressed volatility | Mean reversion |
| Neutral | Normal conditions | Balanced |

```python
from ml_engine.meta_pipeline import HMMRegimeDetector

detector = HMMRegimeDetector(n_states=5, n_features=3)

# Train on historical data
detector.fit(returns, volatility, volume)

# Detect current regime
regime, probabilities = detector.predict_regime(current_features)
print(f"Current regime: {regime}")
print(f"Regime probabilities: {probabilities}")
```

**Mathematical Foundation**:
- Forward-Backward (Baum-Welch) algorithm for training
- Viterbi algorithm for state sequence estimation
- Features: returns, realized volatility, volume ratios

##### Bayesian Hyperparameter Optimizer

Uses Gaussian Process optimization with Upper Confidence Bound (UCB) acquisition:

```python
from ml_engine.meta_pipeline import BayesianHyperparameterOptimizer

optimizer = BayesianHyperparameterOptimizer(
    param_bounds={
        'lookback': (10, 100),
        'threshold': (0.01, 0.1),
        'leverage': (1.0, 3.0)
    }
)

# Suggest next parameters to try
next_params = optimizer.suggest_next()

# Update with observed performance
optimizer.update(next_params, observed_sharpe=2.1)
```

##### Capital Allocation Engine

Combines Kelly Criterion with Risk-Parity allocation:

```python
from ml_engine.meta_pipeline import CapitalAllocationEngine

allocator = CapitalAllocationEngine(
    kelly_fraction=0.5,  # Half-Kelly for safety
    risk_parity_weight=0.5  # Blend with risk-parity
)

# Get optimal allocations
allocations = allocator.allocate(
    expected_returns={'BTC': 0.02, 'ETH': 0.015},
    covariance_matrix=cov_matrix,
    current_regime='bull'
)
```

**Kelly Criterion Formula**:
```
f* = (μ - r) / σ² × kelly_fraction
```

Where:
- μ = expected return
- r = risk-free rate
- σ² = variance
- kelly_fraction = fractional Kelly (0.5 recommended)

---

### Pipeline 1: Advanced Data Pipeline

**File**: `ml_engine/advanced_data_pipeline.py`

**Purpose**: Multi-layer data processing with microstructure analysis, NLP sentiment, and stationarity transformations.

#### Components

##### Microstructure Analyzer

Extracts market microstructure features:

```python
from ml_engine.advanced_data_pipeline import MicrostructureAnalyzer

analyzer = MicrostructureAnalyzer(
    vpin_window=50,
    flow_imbalance_window=100
)

# Analyze trade data
features = analyzer.analyze(trades_df, orderbook_df)

print(f"VPIN: {features['vpin']}")  # Volume-Synchronized Probability of Informed Trading
print(f"Order Flow Imbalance: {features['ofi']}")
print(f"Kyle's Lambda: {features['kyle_lambda']}")  # Price impact coefficient
```

**Key Metrics**:

| Metric | Description | Use |
|--------|-------------|-----|
| VPIN | Probability of informed trading | Detect information asymmetry |
| OFI | Order flow imbalance | Predict short-term direction |
| Kyle's Lambda | Market impact coefficient | Execution optimization |
| Bid-Ask Spread | Liquidity indicator | Transaction cost estimation |

##### NLP Sentiment Pipeline

Processes text data for sentiment analysis:

```python
from ml_engine.advanced_data_pipeline import NLPSentimentPipeline

nlp = NLPSentimentPipeline(
    use_lexicon=True,
    topic_modeling=True,
    n_topics=10
)

# Analyze news/social media
sentiment = nlp.analyze([
    "Bitcoin breaks new all-time high",
    "SEC delays ETF decision again"
])

print(f"Aggregate sentiment: {sentiment['aggregate_score']}")
print(f"Topic distribution: {sentiment['topic_weights']}")
```

**Features**:
- Lexicon-based sentiment (bullish/bearish word counts)
- LDA topic modeling for theme extraction
- Temporal sentiment aggregation

##### SABR Volatility Surface

Calibrates the SABR stochastic volatility model:

```python
from ml_engine.advanced_data_pipeline import SABRVolatilitySurface

sabr = SABRVolatilitySurface()

# Calibrate to market data
params = sabr.calibrate(
    forward_price=50000,
    strikes=[45000, 47500, 50000, 52500, 55000],
    market_vols=[0.65, 0.60, 0.58, 0.61, 0.66],
    time_to_expiry=0.25
)

print(f"Alpha (vol of vol): {params['alpha']}")
print(f"Beta (CEV exponent): {params['beta']}")
print(f"Rho (correlation): {params['rho']}")
print(f"Nu (vol of vol): {params['nu']}")

# Get implied vol for any strike
iv = sabr.get_implied_vol(strike=48000, expiry=0.25)
```

##### Stationarity Transformer

Transforms non-stationary data for ML models:

```python
from ml_engine.advanced_data_pipeline import StationarityTransformer

transformer = StationarityTransformer()

# Apply transformations
stationary_data = transformer.transform(price_series, method='zscore')

# Available methods:
# - 'zscore': Z-score normalization
# - 'quantile': Percentile transformation
# - 'fractional_diff': Fractional differencing (preserves memory)
# - 'log_returns': Log returns
```

---

### Pipeline 2: Alpha Pipeline (Deep Signal Generation)

**File**: `ml_engine/alpha_pipeline.py`

**Purpose**: Generates alpha signals using hierarchical models, Gaussian Processes, and causal inference.

#### Components

##### Multi-Timeframe Feature Generators

**Momentum Features**:
```python
from ml_engine.alpha_pipeline import MomentumFeatures

momentum = MomentumFeatures(lookbacks=[5, 10, 21, 63, 126, 252])
features = momentum.generate(price_series)

# Features include:
# - Multi-timeframe momentum
# - Momentum acceleration
# - Cross-sectional momentum rank
```

**Mean Reversion Features**:
```python
from ml_engine.alpha_pipeline import MeanReversionFeatures

mr = MeanReversionFeatures(
    half_lives=[5, 10, 21],
    zscore_window=20
)
features = mr.generate(price_series)

# Features include:
# - Z-scores at multiple timeframes
# - Half-life estimation
# - Hurst exponent (mean-reversion strength)
```

**Carry Features**:
```python
from ml_engine.alpha_pipeline import CarryFeatures

carry = CarryFeatures()
features = carry.generate(spot_prices, futures_prices, funding_rates)

# Features include:
# - Futures basis
# - Funding rate signals
# - Roll yield
```

##### Gaussian Process Regressor

Provides predictions with uncertainty quantification:

```python
from ml_engine.alpha_pipeline import GaussianProcessRegressor

gp = GaussianProcessRegressor(
    kernel='rbf',
    length_scale=1.0,
    noise_level=0.1
)

# Fit and predict with uncertainty
gp.fit(X_train, y_train)
mean, std = gp.predict(X_test, return_std=True)

# Use uncertainty for position sizing
confidence = 1 / std
position_size = signal * confidence
```

**Key Advantage**: GP provides calibrated uncertainty estimates, enabling:
- Confidence-weighted position sizing
- Detection of out-of-distribution inputs
- Adaptive risk management

##### Quantile Random Forest

Predicts full return distribution:

```python
from ml_engine.alpha_pipeline import QuantileRandomForest

qrf = QuantileRandomForest(
    n_estimators=100,
    quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]
)

# Get quantile predictions
qrf.fit(X_train, y_train)
predictions = qrf.predict(X_test)

print(f"5th percentile (VaR): {predictions['q_0.05']}")
print(f"Median prediction: {predictions['q_0.5']}")
print(f"95th percentile: {predictions['q_0.95']}")

# Calculate prediction intervals
interval_width = predictions['q_0.95'] - predictions['q_0.05']
```

##### Causal Inference Engine

Estimates causal effects using Do-Calculus:

```python
from ml_engine.alpha_pipeline import CausalInferenceEngine

causal = CausalInferenceEngine()

# Define causal graph
causal.define_graph({
    'BTC_return': ['ETH_return', 'funding_rate'],
    'ETH_return': ['funding_rate'],
    'funding_rate': []
})

# Estimate Average Treatment Effect
ate = causal.estimate_ate(
    data=df,
    treatment='funding_rate',
    outcome='BTC_return',
    method='inverse_propensity_weighting'
)

print(f"Causal effect of funding rate on BTC: {ate}")
```

##### Hierarchical Alpha Model

Combines multiple alpha sources with adaptive weighting:

```python
from ml_engine.alpha_pipeline import HierarchicalAlphaModel

model = HierarchicalAlphaModel(
    alpha_sources=['momentum', 'mean_reversion', 'carry', 'ml'],
    decay_factor=0.94
)

# Update with new observations
model.update(predictions={'momentum': 0.02, 'mean_reversion': -0.01},
             actual_return=0.015)

# Get blended signal with adaptive weights
signal = model.get_combined_signal()
print(f"Blended alpha: {signal}")
print(f"Current weights: {model.weights}")
```

---

### Pipeline 3: Advanced Risk Pipeline

**File**: `ml_engine/advanced_risk_pipeline.py`

**Purpose**: Comprehensive risk modeling with dynamic betas, copulas, and liquidity-adjusted VaR.

#### Components

##### Kalman Filter Beta Estimation

Tracks time-varying betas dynamically:

```python
from ml_engine.advanced_risk_pipeline import KalmanFilterBeta

kalman = KalmanFilterBeta(
    process_variance=0.001,  # How fast beta can change
    measurement_variance=0.01  # Observation noise
)

# Update with new observations
for asset_return, market_return in zip(asset_returns, market_returns):
    beta, beta_std = kalman.update(asset_return, market_return)

print(f"Current beta: {beta:.3f} ± {beta_std:.3f}")
```

**Kalman Filter Equations**:
```
State: β_t = β_{t-1} + ε_t  (random walk)
Observation: r_asset = α + β_t × r_market + η_t

Predict: β̂_t|t-1 = β̂_{t-1}
Update: β̂_t = β̂_t|t-1 + K_t × (r_asset - β̂_t|t-1 × r_market)
```

##### Multi-Factor Risk Model

Decomposes risk into systematic and idiosyncratic components:

```python
from ml_engine.advanced_risk_pipeline import MultiFactorRiskModel

risk_model = MultiFactorRiskModel(
    factors=['market', 'size', 'momentum', 'volatility', 'liquidity']
)

# Fit model
risk_model.fit(returns_df, factor_returns_df)

# Get risk decomposition
decomp = risk_model.decompose_risk(portfolio_weights)

print(f"Systematic risk: {decomp['systematic']:.2%}")
print(f"Idiosyncratic risk: {decomp['idiosyncratic']:.2%}")
print(f"Factor contributions: {decomp['factor_contributions']}")
```

##### Cornish-Fisher VaR

VaR with higher moments (skewness, kurtosis):

```python
from ml_engine.advanced_risk_pipeline import CornishFisherVaR

cf_var = CornishFisherVaR(confidence=0.99)

# Calculate VaR with skewness/kurtosis adjustment
var = cf_var.calculate(
    returns=returns,
    portfolio_value=1000000
)

print(f"99% VaR: ${var['var']:,.0f}")
print(f"99% CVaR: ${var['cvar']:,.0f}")
print(f"Skewness: {var['skewness']:.2f}")
print(f"Kurtosis: {var['kurtosis']:.2f}")
```

**Cornish-Fisher Expansion**:
```
z_cf = z + (z² - 1)×S/6 + (z³ - 3z)×K/24 - (2z³ - 5z)×S²/36
```

Where S = skewness, K = excess kurtosis, z = normal quantile

##### Copula Risk Model

Models tail dependencies between assets:

```python
from ml_engine.advanced_risk_pipeline import CopulaRiskModel

copula = CopulaRiskModel(copula_type='student_t')

# Fit copula
copula.fit(returns_df)

print(f"Tail dependence: {copula.tail_dependence}")
print(f"Degrees of freedom: {copula.df}")

# Simulate correlated returns preserving tail dependence
simulated = copula.simulate(n_scenarios=10000)

# Calculate joint VaR
joint_var = copula.joint_var(portfolio_weights, confidence=0.99)
```

**Why Copulas?**
- Gaussian correlation underestimates tail risk
- Student-t copula captures "correlation breakdown" in crashes
- Better risk estimates during market stress

##### Liquidity-Adjusted VaR (L-VaR)

Incorporates liquidation costs:

```python
from ml_engine.advanced_risk_pipeline import LiquidityAdjustedRisk

lvar = LiquidityAdjustedRisk(
    base_spread=0.001,
    kyle_lambda=0.0001,  # Price impact coefficient
    liquidation_time=1.0  # Days to liquidate
)

# Calculate L-VaR
result = lvar.calculate(
    position_value=500000,
    daily_volume=10000000,
    var_estimate=25000
)

print(f"Base VaR: ${result['base_var']:,.0f}")
print(f"Liquidity cost: ${result['liquidity_cost']:,.0f}")
print(f"L-VaR: ${result['lvar']:,.0f}")
```

**L-VaR Formula**:
```
L-VaR = VaR + Liquidation Cost
Liquidation Cost = Position × (spread/2 + λ × sqrt(Position/ADV))
```

---

### Pipeline 4: Advanced Portfolio Optimizer

**File**: `ml_engine/advanced_portfolio_optimizer.py`

**Purpose**: Robust portfolio construction using Black-Litterman and online optimization.

#### Components

##### Black-Litterman Optimizer

Combines market equilibrium with active views:

```python
from ml_engine.advanced_portfolio_optimizer import BlackLittermanOptimizer

bl = BlackLittermanOptimizer(
    risk_aversion=2.5,
    tau=0.05  # Uncertainty in equilibrium
)

# Set market cap weights (equilibrium)
bl.set_market_weights({
    'BTC': 0.50,
    'ETH': 0.30,
    'SOL': 0.10,
    'Others': 0.10
})

# Add active views
bl.add_view(
    assets=['BTC'],
    expected_return=0.02,  # BTC returns 2%
    confidence=0.8
)
bl.add_view(
    assets=['ETH', 'SOL'],
    expected_return=0.01,  # ETH-SOL spread returns 1%
    confidence=0.6,
    view_type='relative'
)

# Get posterior weights
weights = bl.optimize(covariance_matrix=cov)
print(f"Optimal weights: {weights}")
```

**Black-Litterman Formula**:
```
Posterior Return = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ × [(τΣ)⁻¹π + P'Ω⁻¹Q]
```

Where:
- π = equilibrium returns
- P = view matrix
- Q = view returns
- Ω = view uncertainty

##### Online Convex Optimizer

Adaptive portfolio optimization:

```python
from ml_engine.advanced_portfolio_optimizer import OnlineConvexOptimizer

online = OnlineConvexOptimizer(
    algorithm='ons',  # Online Newton Step
    learning_rate=0.1,
    regularization=0.01
)

# Update portfolio iteratively
for returns in daily_returns:
    weights = online.get_weights()
    online.update(returns)

print(f"Final weights: {online.get_weights()}")
print(f"Cumulative wealth: {online.wealth}")
```

**Available Algorithms**:

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| OGD | Online Gradient Descent | Simple, fast |
| FTRL | Follow The Regularized Leader | Sparse solutions |
| ONS | Online Newton Step | Best regret bounds |

##### Transaction Cost Optimizer

Optimizes with realistic transaction costs:

```python
from ml_engine.advanced_portfolio_optimizer import TransactionCostOptimizer

tc_opt = TransactionCostOptimizer(
    spread_cost=0.001,
    market_impact_coeff=0.0001,
    fixed_cost=1.0
)

# Optimize considering turnover
new_weights = tc_opt.optimize(
    current_weights=current,
    target_weights=target,
    portfolio_value=1000000
)

print(f"Adjusted weights: {new_weights}")
print(f"Estimated transaction cost: ${tc_opt.last_cost:,.0f}")
```

##### Almgren-Chriss Executor

Optimal execution scheduling:

```python
from ml_engine.advanced_portfolio_optimizer import AlmgrenChrissExecutor

executor = AlmgrenChrissExecutor(
    risk_aversion=1e-6,
    volatility=0.02,
    market_impact=0.0001
)

# Get optimal execution schedule
schedule = executor.get_schedule(
    total_shares=10000,
    time_horizon=24,  # hours
    n_periods=12
)

for period, shares in enumerate(schedule):
    print(f"Period {period}: Trade {shares:.0f} shares")
```

**Almgren-Chriss Model**:
```
Optimal trajectory minimizes: E[Cost] + λ × Var[Cost]
Cost = temporary impact + permanent impact + volatility risk
```

---

### Pipeline 5: Execution Pipeline

**File**: `ml_engine/execution_pipeline.py`

**Purpose**: High-performance execution with multi-agent system and RL optimization.

#### Components

##### Market Maker Agent

Implements Stoikov's optimal market making:

```python
from ml_engine.execution_pipeline import MarketMakerAgent

mm = MarketMakerAgent(
    risk_aversion=0.1,
    inventory_target=0,
    max_inventory=100
)

# Get optimal quotes
mid_price = 50000
volatility = 0.02
quotes = mm.get_quotes(
    mid_price=mid_price,
    volatility=volatility,
    current_inventory=10,
    time_remaining=3600
)

print(f"Bid: {quotes['bid_price']} ({quotes['bid_size']} units)")
print(f"Ask: {quotes['ask_price']} ({quotes['ask_size']} units)")
```

**Stoikov's Optimal Spread**:
```
reservation_price = mid - q × γ × σ² × T
optimal_spread = γ × σ² × T + (2/γ) × ln(1 + γ/k)
```

Where q = inventory, γ = risk aversion, T = time remaining

##### Smart Order Router

Routes orders to optimal venues:

```python
from ml_engine.execution_pipeline import SmartOrderRouter

router = SmartOrderRouter(
    venues=['binance', 'bybit', 'hyperliquid'],
    selection_method='ucb'  # Upper Confidence Bound
)

# Update venue performance
router.update_venue('binance', execution_quality=0.998, latency=50)

# Get optimal routing
routing = router.route_order(
    symbol='BTC/USDT',
    size=1.0,
    side='buy'
)

print(f"Route to: {routing['venue']}")
print(f"Expected slippage: {routing['expected_slippage']:.4f}")
```

##### Execution Manager

Manages order execution with multiple strategies:

```python
from ml_engine.execution_pipeline import ExecutionManager

manager = ExecutionManager()

# Execute with TWAP
await manager.execute_twap(
    symbol='BTC/USDT',
    total_size=10,
    duration_seconds=3600,
    n_slices=12
)

# Execute with VWAP
await manager.execute_vwap(
    symbol='ETH/USDT',
    total_size=50,
    volume_profile=historical_volume_profile
)

# Execute with Almgren-Chriss
await manager.execute_optimal(
    symbol='SOL/USDT',
    total_size=100,
    urgency=0.5  # 0=passive, 1=aggressive
)
```

##### RL Execution Agent

Learns optimal execution from experience:

```python
from ml_engine.execution_pipeline import RLExecutionAgent

agent = RLExecutionAgent(
    state_dim=10,
    action_dim=5,
    hidden_dim=64
)

# Training loop
for episode in range(1000):
    state = env.reset()
    done = False

    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state

# Use trained agent
action = agent.select_action(current_state, deterministic=True)
```

**State Features**:
- Remaining inventory
- Time remaining
- Recent price changes
- Order book imbalance
- Recent execution quality

---

### Pipeline 6: Monitoring Pipeline

**File**: `ml_engine/monitoring_pipeline.py`

**Purpose**: Comprehensive monitoring with performance attribution and overfitting detection.

#### Components

##### Performance Attributor

Brinson-Fachler style attribution:

```python
from ml_engine.monitoring_pipeline import PerformanceAttributor

attributor = PerformanceAttributor(
    benchmark_weights={'BTC': 0.5, 'ETH': 0.3, 'Others': 0.2}
)

# Calculate attribution
attribution = attributor.attribute(
    portfolio_weights={'BTC': 0.6, 'ETH': 0.25, 'SOL': 0.15},
    portfolio_returns={'BTC': 0.05, 'ETH': 0.03, 'SOL': 0.08},
    benchmark_returns={'BTC': 0.05, 'ETH': 0.03, 'Others': 0.02}
)

print(f"Allocation effect: {attribution['allocation']:.4f}")
print(f"Selection effect: {attribution['selection']:.4f}")
print(f"Interaction effect: {attribution['interaction']:.4f}")
print(f"Total active return: {attribution['total']:.4f}")
```

**Brinson Attribution**:
```
Allocation = Σ(w_p - w_b) × (r_b,s - r_b)
Selection = Σw_b × (r_p,s - r_b,s)
Interaction = Σ(w_p - w_b) × (r_p,s - r_b,s)
```

##### PBO Analyzer (Probability of Backtest Overfitting)

Detects strategy overfitting:

```python
from ml_engine.monitoring_pipeline import PBOAnalyzer

pbo = PBOAnalyzer(n_partitions=16)

# Analyze strategy performance matrix
# Each row = strategy variant, each column = time period
pbo_result = pbo.analyze(performance_matrix)

print(f"PBO: {pbo_result['pbo']:.2%}")
print(f"Performance degradation: {pbo_result['degradation']:.2%}")

if pbo_result['pbo'] > 0.5:
    print("WARNING: High probability of overfitting!")
```

**CSCV Method (Combinatorially Symmetric Cross-Validation)**:
1. Split data into S partitions
2. For each combination, train on half, test on other half
3. Compare in-sample vs out-of-sample performance
4. PBO = P(OOS rank < IS rank)

##### Stress Testing Framework

Tests portfolio under extreme scenarios:

```python
from ml_engine.monitoring_pipeline import StressTestingFramework

stress = StressTestingFramework()

# Run predefined scenarios
results = stress.run_scenarios(
    portfolio_weights={'BTC': 0.4, 'ETH': 0.3, 'SOL': 0.3},
    portfolio_value=1000000
)

for scenario, result in results.items():
    print(f"{scenario}: ${result['pnl']:,.0f} ({result['pnl_pct']:.1%})")
```

**Built-in Scenarios**:

| Scenario | Description |
|----------|-------------|
| `market_crash_2020` | COVID crash (-40% BTC) |
| `luna_collapse` | Terra/Luna event |
| `ftx_contagion` | FTX bankruptcy |
| `flash_crash` | -20% in 1 hour |
| `correlation_breakdown` | Correlations go to 1 |
| `liquidity_crisis` | 90% volume drop |
| `rate_shock` | Interest rate spike |
| `regulatory_ban` | Major country ban |
| `stablecoin_depeg` | USDT/USDC depeg |

##### Model Diagnostics Engine

Real-time model health monitoring:

```python
from ml_engine.monitoring_pipeline import ModelDiagnosticsEngine

diagnostics = ModelDiagnosticsEngine()

# Add prediction and actual
diagnostics.add_observation(
    prediction=0.02,
    actual=0.015,
    features=current_features,
    timestamp=datetime.now()
)

# Get diagnostics report
report = diagnostics.get_report()

print(f"Prediction bias: {report['bias']:.4f}")
print(f"RMSE: {report['rmse']:.4f}")
print(f"Hit rate: {report['hit_rate']:.1%}")
print(f"Information coefficient: {report['ic']:.3f}")
print(f"Feature drift detected: {report['drift_features']}")
```

---

### Pipeline 7: Advanced ML Optimizations

**File**: `ml_engine/advanced_ml_optimizations.py`

**Purpose**: State-of-the-art ML techniques including meta-learning and Bayesian deep learning.

#### Components

##### MAML Optimizer (Model-Agnostic Meta-Learning)

Fast adaptation to new market regimes:

```python
from ml_engine.advanced_ml_optimizations import MAMLOptimizer

maml = MAMLOptimizer(
    model=base_model,
    inner_lr=0.01,
    outer_lr=0.001,
    n_inner_steps=5
)

# Create regime-specific tasks
tasks = [
    {'features': bull_features, 'targets': bull_returns},
    {'features': bear_features, 'targets': bear_returns},
    {'features': sideways_features, 'targets': sideways_returns}
]

# Meta-train
maml.meta_train(tasks, n_epochs=100)

# Fast adaptation to new regime (few-shot learning)
adapted_model = maml.adapt(
    new_features=new_regime_data,
    new_targets=new_regime_returns,
    n_steps=5
)
```

**MAML Objective**:
```
θ* = argmin_θ Σ L(f_{θ - α∇L(θ, D_train)}, D_test)
```

Meta-learns initialization that can quickly adapt to any regime.

##### Bayesian Neural Network

Uncertainty quantification via MC Dropout:

```python
from ml_engine.advanced_ml_optimizations import BayesianNeuralNetwork

bnn = BayesianNeuralNetwork(
    input_dim=50,
    hidden_dims=[64, 32],
    dropout_rate=0.2,
    n_samples=100
)

# Train (with dropout)
bnn.fit(X_train, y_train, epochs=100)

# Predict with uncertainty
mean, std = bnn.predict(X_test, return_uncertainty=True)

# Use uncertainty for position sizing
confidence = 1 / (1 + std)
position = signal * confidence
```

**Epistemic vs Aleatoric Uncertainty**:
- Epistemic: Model uncertainty (reducible with more data)
- Aleatoric: Data uncertainty (irreducible)

##### Evidential Neural Network

Direct uncertainty estimation without sampling:

```python
from ml_engine.advanced_ml_optimizations import EvidentialNeuralNetwork

enn = EvidentialNeuralNetwork(
    input_dim=50,
    hidden_dims=[64, 32]
)

# Outputs Normal-Inverse-Gamma parameters
enn.fit(X_train, y_train)

# Get prediction with uncertainties
result = enn.predict(X_test)

print(f"Mean prediction: {result['mean']}")
print(f"Aleatoric uncertainty: {result['aleatoric']}")
print(f"Epistemic uncertainty: {result['epistemic']}")
print(f"Total uncertainty: {result['total']}")
```

**Evidential Output**:
```
Output = (γ, ν, α, β) parametrizing Normal-Inverse-Gamma
Mean = γ
Aleatoric = β / (α - 1)
Epistemic = β / (ν × (α - 1))
```

##### Advanced Causal Inference

Double ML for causal effect estimation:

```python
from ml_engine.advanced_ml_optimizations import AdvancedCausalInference

causal = AdvancedCausalInference(
    treatment_model='random_forest',
    outcome_model='gradient_boosting'
)

# Estimate causal effect with confounding control
ate = causal.estimate_ate_double_ml(
    data=df,
    treatment='funding_rate',
    outcome='next_day_return',
    confounders=['volatility', 'volume', 'trend']
)

print(f"Average Treatment Effect: {ate['ate']:.4f}")
print(f"Standard Error: {ate['se']:.4f}")
print(f"95% CI: [{ate['ci_lower']:.4f}, {ate['ci_upper']:.4f}]")
```

**Double ML**:
1. Estimate E[Y|X] with ML model
2. Estimate E[T|X] with ML model
3. Regress residuals to get unbiased ATE

---

### Pipeline Integration: Full System

All pipelines work together in the main trading loop:

```python
from ml_engine import (
    MetaPipeline,
    AdvancedDataPipeline,
    AlphaPipeline,
    AdvancedRiskPipeline,
    AdvancedPortfolioPipeline,
    MultiAgentExecutionSystem,
    MonitoringPipeline,
    AdvancedMLPipeline
)

# Initialize all pipelines
meta = MetaPipeline()
data = AdvancedDataPipeline()
alpha = AlphaPipeline()
risk = AdvancedRiskPipeline()
portfolio = AdvancedPortfolioPipeline()
execution = MultiAgentExecutionSystem()
monitoring = MonitoringPipeline()
ml_advanced = AdvancedMLPipeline()

# Main trading loop
async def trading_iteration():
    # 1. Meta: Detect regime and adjust parameters
    regime = meta.detect_regime(market_data)
    params = meta.get_regime_parameters(regime)

    # 2. Data: Process raw data
    processed = data.process(raw_data)

    # 3. Alpha: Generate signals
    signals = alpha.generate_signals(processed, regime)

    # 4. Risk: Assess portfolio risk
    risk_metrics = risk.analyze(current_positions, signals)

    # 5. Portfolio: Optimize allocation
    target_weights = portfolio.optimize(
        signals=signals,
        risk_constraints=risk_metrics,
        regime=regime
    )

    # 6. Execution: Execute trades
    trades = await execution.execute(
        current_weights=current_weights,
        target_weights=target_weights
    )

    # 7. Monitoring: Track performance
    monitoring.update(trades, risk_metrics)

    # 8. ML: Adapt models if needed
    if monitoring.should_retrain():
        ml_advanced.adapt_to_regime(regime, recent_data)
```

### Performance Expectations

| Metric | Target | Description |
|--------|--------|-------------|
| Sharpe Ratio | > 2.5 | Risk-adjusted return |
| Max Drawdown | < 15% | Worst peak-to-trough |
| Win Rate | > 55% | Profitable trades |
| Profit Factor | > 1.5 | Gross profit / Gross loss |
| Information Ratio | > 1.0 | Active return / Tracking error |

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
