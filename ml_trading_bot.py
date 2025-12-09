#!/usr/bin/env python3
"""
ML Crypto Trading Bot - Main Orchestrator

Market-neutral long/short strategy with:
- ML-driven predictions (LSTM + ensemble)
- Dynamic beta management based on regime detection
- Correlation screening to prevent correlated positions
- Target Sharpe ratio > 2.5
- Options flow analysis for sentiment
"""

import asyncio
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, Optional

from dotenv import load_dotenv

# ML Engine imports
from ml_engine.config import Config, TradingMode, AssetType
from ml_engine.data_collector import DataCollector
from ml_engine.market_data_aggregator import MarketDataAggregator
from ml_engine.feature_engineering import FeatureEngineer
from ml_engine.risk_models import RiskModels
from ml_engine.portfolio_optimizer import PortfolioOptimizer
from ml_engine.ml_models import MLModels
from ml_engine.signal_generator import SignalGenerator
from ml_engine.execution_engine import ExecutionEngine
from ml_engine.position_manager import PositionManager
from ml_engine.margin_manager import MarginManager
from ml_engine.correlation_screener import CorrelationScreener
from ml_engine.options_flow import OptionsFlowAnalyzer
from ml_engine.options_data_collector import OptionsDataCollector
from ml_engine.regime_detector import RegimeDetector

# New pipelines for inflation/deflation strategy
from ml_engine.token_unlock_pipeline import TokenUnlockPipeline, UnlockRisk
from ml_engine.buyback_analyzer import BuybackAnalyzer, BuybackSentiment
from ml_engine.orderbook_pipeline import OrderbookPipeline, Exchange
from ml_engine.inflation_strategy import InflationDeflationStrategy, InflationSignal

# DEX clients
from dex_clients.lighter_client import LighterClient
from dex_clients.hyperliquid_client import HyperliquidClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/trading.log')
    ]
)
logger = logging.getLogger(__name__)


class MLTradingBot:
    """
    Main trading bot orchestrating all components.

    Trading Flow:
    1. Collect market data
    2. Generate features
    3. Detect market regime
    4. Generate ML predictions
    5. Screen for correlations
    6. Optimize portfolio
    7. Execute trades
    8. Monitor positions
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize trading bot with all components"""
        load_dotenv(config_path)

        # Configuration
        self.config = Config(config_path)
        logger.info(f"Trading mode: {self.config.trading.mode.value}")
        logger.info(f"Primary exchange: {self.config.trading.primary_exchange}")

        # Initialize components
        self._init_data_components()
        self._init_ml_components()
        self._init_trading_components()
        self._init_risk_components()

        # State
        self.is_running = False
        self.iteration_count = 0

    def _init_data_components(self):
        """Initialize data collection components"""
        self.data_collector = DataCollector(self.config)
        self.market_aggregator = MarketDataAggregator(dict(os.environ))
        self.feature_engineer = FeatureEngineer(self.config)

        # Options data (if enabled)
        if self.config.options.enabled:
            self.options_collector = OptionsDataCollector(
                client_id=self.config.api_keys['deribit'].get('client_id'),
                client_secret=self.config.api_keys['deribit'].get('client_secret'),
                testnet=self.config.api_keys['deribit'].get('testnet', True)
            )
            self.options_analyzer = OptionsFlowAnalyzer(
                risk_free_rate=self.config.options.risk_free_rate
            )
        else:
            self.options_collector = None
            self.options_analyzer = None

        # Initialize new pipelines for inflation/deflation strategy
        self._init_inflation_pipelines()

    def _init_inflation_pipelines(self):
        """Initialize token unlock, buyback, and orderbook pipelines"""
        # Get API keys from environment
        tokenomist_key = os.getenv('TOKENOMIST_API_KEY', '')
        cryptorank_key = os.getenv('CRYPTORANK_API_KEY', '')

        # Token unlock pipeline (Tokenomist + CryptoRank)
        self.unlock_pipeline = TokenUnlockPipeline(
            tokenomist_api_key=tokenomist_key if tokenomist_key else None,
            cryptorank_api_key=cryptorank_key if cryptorank_key else None
        )

        # Buyback analyzer (HypurrScan for HYPE)
        self.buyback_analyzer = BuybackAnalyzer(
            tokenomist_api_key=tokenomist_key if tokenomist_key else None
        )

        # Orderbook pipeline (Bybit, Binance, Hyperliquid, Lighter)
        self.orderbook_pipeline = OrderbookPipeline(
            symbols=self.config.get_symbols(),
            exchanges=[Exchange.BYBIT, Exchange.BINANCE, Exchange.HYPERLIQUID],
            depth_levels=50
        )

        # Combined inflation/deflation strategy
        self.inflation_strategy = InflationDeflationStrategy(
            tokenomist_api_key=tokenomist_key if tokenomist_key else None,
            cryptorank_api_key=cryptorank_key if cryptorank_key else None
        )

        logger.info("Initialized inflation/deflation analysis pipelines")

    def _init_ml_components(self):
        """Initialize ML components"""
        self.ml_models = MLModels(self.config)
        self.signal_generator = SignalGenerator(
            kelly_fraction=0.5,
            min_signal_strength=0.3,
            max_position_size=self.config.risk.position_limit_major,
            config=self.config
        )
        self.regime_detector = RegimeDetector(
            risk_on_beta=self.config.regime.risk_on_beta,
            risk_off_beta=self.config.regime.risk_off_beta,
            neutral_beta=self.config.regime.neutral_beta,
            config=self.config
        )

    def _init_trading_components(self):
        """Initialize trading components"""
        # DEX clients
        if self.config.trading.primary_exchange == 'lighter':
            self.primary_client = LighterClient(
                api_key=self.config.api_keys['lighter'].get('api_key'),
                api_secret=self.config.api_keys['lighter'].get('api_secret'),
                enable_fast_execution=self.config.api_keys['lighter'].get('enable_fast_execution', True),
                fast_execution_fee_multiplier=self.config.api_keys['lighter'].get('fast_execution_fee_multiplier', 1.5)
            )
        else:
            self.primary_client = HyperliquidClient(
                private_key=self.config.api_keys['hyperliquid'].get('private_key'),
                testnet=True
            )

        # Execution engine
        self.execution_engine = ExecutionEngine(
            dex_client=self.primary_client,
            slippage_tolerance=self.config.trading.slippage_tolerance,
            max_retries=self.config.trading.max_order_retries,
            config=self.config
        )

        # Position manager
        initial_capital = float(os.getenv('INITIAL_CAPITAL', '100000'))
        self.position_manager = PositionManager(
            initial_capital=initial_capital,
            config=self.config
        )

        # Load saved state if exists
        state_file = 'state/positions.json'
        if os.path.exists(state_file):
            self.position_manager.load_state(state_file)

    def _init_risk_components(self):
        """Initialize risk management components"""
        self.risk_models = RiskModels(self.config)

        self.portfolio_optimizer = PortfolioOptimizer(
            min_sharpe_ratio=self.config.risk.min_sharpe_ratio,
            target_beta=0.0,  # Updated dynamically based on regime
            max_net_exposure=self.config.risk.max_net_exposure,
            max_gross_exposure=self.config.risk.max_gross_exposure,
            risk_free_rate=self.config.options.risk_free_rate,
            max_position_size=self.config.risk.position_limit_major,
            config=self.config
        )

        self.margin_manager = MarginManager(
            max_leverage=self.config.risk.max_leverage,
            config=self.config
        )

        self.correlation_screener = CorrelationScreener(
            max_correlation=0.7,
            lookback_periods=168,
            cluster_threshold=0.6,
            config=self.config
        )

    async def run(self):
        """Main trading loop"""
        self.is_running = True
        logger.info("Starting ML Trading Bot...")

        # Create necessary directories
        os.makedirs('logs', exist_ok=True)
        os.makedirs('state', exist_ok=True)
        os.makedirs('models', exist_ok=True)

        # Try to load pre-trained models
        self.ml_models.load_models('models/')

        while self.is_running:
            try:
                self.iteration_count += 1
                logger.info(f"\n{'='*50}")
                logger.info(f"Iteration #{self.iteration_count} - {datetime.utcnow()}")
                logger.info(f"{'='*50}")

                # Run trading iteration
                await self._trading_iteration()

                # Save state
                self.position_manager.save_state('state/positions.json')

                # Wait for next iteration
                wait_time = self.config.trading.rebalance_interval
                logger.info(f"Waiting {wait_time}s until next iteration...")
                await asyncio.sleep(wait_time)

            except KeyboardInterrupt:
                logger.info("Shutdown requested...")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait before retry

        logger.info("Trading bot stopped")

    async def _trading_iteration(self):
        """Single trading iteration"""
        symbols = self.config.get_symbols()

        # Step 1: Collect market data
        logger.info("Collecting market data...")
        market_data = await self._collect_market_data(symbols)

        if not market_data:
            logger.warning("No market data available")
            return

        # Step 2: Generate features
        logger.info("Generating features...")
        features_data = {}
        returns_data = {}

        for symbol, data in market_data.items():
            if data is not None and not data.empty:
                features_df = self.feature_engineer.create_all_features(data)
                features_data[symbol] = features_df
                returns_data[symbol] = features_df['returns'].dropna()

        # Step 3: Detect market regime
        logger.info("Detecting market regime...")
        options_analysis = None
        if self.options_collector and self.options_analyzer:
            options_analysis = await self._get_options_analysis()

        btc_data = market_data.get('BTC/USDT')
        regime = self.regime_detector.detect_regime(
            btc_data if btc_data is not None else list(market_data.values())[0],
            options_analysis
        )
        logger.info(f"Regime: {regime.regime.value} (confidence: {regime.confidence:.2f})")
        logger.info(f"Recommended beta: {regime.recommended_beta:.2f}")

        # Update portfolio optimizer target beta
        self.portfolio_optimizer.target_beta = regime.recommended_beta

        # Step 4: Generate ML predictions
        logger.info("Generating predictions...")
        predictions = {}
        for symbol, features in features_data.items():
            pred = self.ml_models.predict(features.dropna(), symbol)
            predictions[symbol] = pred
            if pred.direction != 'neutral':
                logger.info(f"  {symbol}: {pred.direction} ({pred.predicted_return:.4f}, conf: {pred.confidence:.2f})")

        # Step 5: Get current prices
        prices = self.market_aggregator.get_current_prices(symbols)

        # Step 6: Calculate volatilities
        volatilities = {}
        for symbol, returns in returns_data.items():
            if len(returns) > 20:
                volatilities[symbol] = self.risk_models.calculate_volatility(
                    returns, method='ewma', window=20, annualize=True
                ).iloc[-1]

        # Step 7: Generate ML-based signals
        logger.info("Generating trading signals...")
        signals = self.signal_generator.generate_signals(
            predictions, volatilities, prices
        )

        # Step 7.5: Integrate inflation/deflation analysis
        # This is an ADDITIONAL factor, not the sole strategy
        logger.info("Analyzing inflation/deflation factors...")
        inflation_signals = await self._analyze_inflation_deflation(symbols, prices)
        signals = self._integrate_inflation_signals(signals, inflation_signals)

        # Step 8: Screen for correlations
        logger.info("Screening for correlations...")
        if returns_data:
            import pandas as pd
            returns_df = pd.DataFrame(returns_data)
            signals = self.signal_generator.filter_by_correlation(
                signals, self.correlation_screener, returns_df
            )

            # Log correlation report
            corr_report = self.correlation_screener.get_correlation_report(returns_df)
            if corr_report['high_correlation_pairs']:
                logger.info(f"High correlation pairs: {len(corr_report['high_correlation_pairs'])}")
                for pair in corr_report['high_correlation_pairs'][:3]:
                    logger.info(f"  {pair[0]} <-> {pair[1]}: {pair[2]:.2f}")

        # Step 9: Portfolio optimization
        if signals and returns_data:
            logger.info("Optimizing portfolio...")
            import pandas as pd

            returns_df = pd.DataFrame(returns_data)
            expected_returns = pd.Series({
                s: signals[s].target_weight * 0.1  # Scaled expected return
                for s in signals
            })
            cov_matrix = returns_df.cov() * (252 * 24)  # Annualized

            # Get asset betas
            btc_returns = returns_data.get('BTC/USDT')
            asset_betas = {}
            if btc_returns is not None:
                for symbol, returns in returns_data.items():
                    asset_betas[symbol] = self.risk_models.calculate_beta(returns, btc_returns)

            # Optimize
            result = self.portfolio_optimizer.optimize_mean_variance(
                expected_returns,
                cov_matrix,
                asset_betas=asset_betas
            )

            logger.info(f"Optimal portfolio - Sharpe: {result.sharpe_ratio:.2f}, "
                       f"Net Exposure: {result.net_exposure:.1%}")

            # Step 10: Execute trades
            if self.config.trading.mode != TradingMode.PAPER:
                await self._execute_trades(result.weights, prices)
            else:
                self._simulate_trades(result.weights, prices)

        # Step 11: Update and display portfolio status
        self._display_portfolio_status(prices)

    async def _collect_market_data(self, symbols):
        """Collect market data for all symbols"""
        market_data = {}

        for symbol in symbols:
            try:
                df = self.market_aggregator.fetch_aggregated_data(
                    symbol,
                    lookback_days=self.config.data.lookback_days,
                    interval=self.config.data.ohlcv_timeframe
                )
                if not df.empty:
                    market_data[symbol] = df
            except Exception as e:
                logger.warning(f"Failed to fetch data for {symbol}: {e}")

        logger.info(f"Collected data for {len(market_data)}/{len(symbols)} symbols")
        return market_data

    async def _analyze_inflation_deflation(self, symbols, prices):
        """
        Analyze token unlock schedules and buyback activity

        This provides ADDITIONAL signals to complement ML predictions:
        - Short bias for tokens with large upcoming unlocks
        - Long bias for deflationary tokens during active buybacks
        """
        inflation_signals = {}

        try:
            # Get orderbook analysis if available
            orderbook_analyses = self.orderbook_pipeline.get_all_analyses()

            # Screen all symbols through inflation strategy
            inflation_signals = await self.inflation_strategy.screen_universe(
                symbols, prices, orderbook_analyses
            )

            # Log significant findings
            shorts = [s for s, sig in inflation_signals.items()
                     if sig.direction.value.startswith('short')]
            longs = [s for s, sig in inflation_signals.items()
                    if sig.direction.value.startswith('long')]

            if shorts:
                logger.info(f"Inflation analysis - Short candidates: {len(shorts)}")
                for symbol in shorts[:3]:
                    sig = inflation_signals[symbol]
                    logger.info(f"  {symbol}: {sig.inflation_type.value} ({sig.combined_signal:.2f})")

            if longs:
                logger.info(f"Inflation analysis - Long candidates: {len(longs)}")
                for symbol in longs[:3]:
                    sig = inflation_signals[symbol]
                    logger.info(f"  {symbol}: {sig.inflation_type.value} ({sig.combined_signal:.2f})")

            # Log upcoming unlocks
            unlock_calendar = self.inflation_strategy.get_unlock_calendar(days_ahead=7)
            if unlock_calendar:
                logger.info(f"Upcoming unlocks (7 days): {len(unlock_calendar)}")
                for unlock in unlock_calendar[:3]:
                    logger.info(f"  {unlock['token']}: {unlock['amount_pct']:.1f}% on {unlock['date'][:10]}")

        except Exception as e:
            logger.warning(f"Inflation/deflation analysis failed: {e}")

        return inflation_signals

    def _integrate_inflation_signals(self, ml_signals, inflation_signals):
        """
        Integrate inflation/deflation signals into ML signals

        Weight distribution (configurable):
        - ML predictions: 70%
        - Inflation/deflation: 30%

        This ensures inflation analysis is a COMPONENT, not the sole driver
        """
        INFLATION_WEIGHT = 0.30  # 30% weight for inflation signals
        ML_WEIGHT = 0.70         # 70% weight for ML signals

        if not inflation_signals:
            return ml_signals

        for symbol, ml_signal in ml_signals.items():
            if symbol not in inflation_signals:
                continue

            inf_signal = inflation_signals[symbol]

            # Only adjust if inflation signal is significant
            if abs(inf_signal.combined_signal) < 0.25:
                continue

            # Adjust target weight based on inflation signal
            # Positive inflation signal (long) increases weight
            # Negative inflation signal (short) decreases weight
            inflation_adjustment = inf_signal.combined_signal * INFLATION_WEIGHT

            # Update signal weight
            original_weight = ml_signal.target_weight
            adjusted_weight = (original_weight * ML_WEIGHT) + inflation_adjustment

            # Apply position limits based on asset type
            asset_type = self.config.get_asset_type(symbol)
            max_pos = self.config.get_position_limit(symbol)

            # Meme tokens: only allow shorts
            if asset_type == AssetType.MEME and adjusted_weight > 0:
                adjusted_weight = 0  # Don't long meme tokens

            # High unlock tokens: bias toward shorts
            if asset_type == AssetType.HIGH_UNLOCK and inf_signal.unlock_signal < -0.3:
                adjusted_weight = min(adjusted_weight, -0.05)  # Ensure short bias

            # Deflationary tokens: boost longs during buybacks
            if asset_type == AssetType.DEFLATIONARY and inf_signal.buyback_signal > 0.5:
                adjusted_weight = max(adjusted_weight, original_weight * 1.2)  # 20% boost

            # Cap to position limits
            adjusted_weight = max(-max_pos, min(max_pos, adjusted_weight))

            ml_signal.target_weight = adjusted_weight

            # Log significant adjustments
            if abs(adjusted_weight - original_weight) > 0.02:
                logger.debug(f"{symbol}: Weight adjusted {original_weight:.3f} -> {adjusted_weight:.3f} "
                           f"(inflation: {inf_signal.combined_signal:.2f})")

        return ml_signals

    async def _get_options_analysis(self):
        """Get options flow analysis"""
        try:
            # Fetch BTC options
            btc_chain = await self.options_collector.fetch_options_chain('BTC')
            btc_spot = await self.options_collector.fetch_index_price('BTC')

            if btc_chain and btc_spot > 0:
                analysis = self.options_analyzer.analyze_options_chain(btc_chain, btc_spot)
                logger.info(f"Options - ATM IV: {analysis.atm_iv:.1%}, P/C Ratio: {analysis.put_call_ratio:.2f}")
                return {
                    'atm_iv': analysis.atm_iv,
                    'put_call_ratio': analysis.put_call_ratio,
                    'iv_skew': analysis.iv_skew,
                    'sentiment': analysis.sentiment
                }
        except Exception as e:
            logger.warning(f"Options analysis failed: {e}")

        return None

    async def _execute_trades(self, target_weights, prices):
        """Execute trades to reach target weights"""
        current_weights = self.position_manager.get_position_weights()

        # Calculate required trades
        trades = self.portfolio_optimizer.rebalance_portfolio(
            current_weights, target_weights, threshold=0.02
        )

        if not trades:
            logger.info("No rebalancing needed")
            return

        logger.info(f"Executing {len(trades)} trades...")
        orders = await self.execution_engine.execute_trades(trades, prices)

        for order in orders:
            logger.info(f"  {order.side.value} {order.quantity} {order.symbol} @ {order.average_price}")

    def _simulate_trades(self, target_weights, prices):
        """Simulate trades in paper mode"""
        current_weights = self.position_manager.get_position_weights()

        for symbol, target in target_weights.items():
            current = current_weights.get(symbol, 0)
            diff = target - current

            if abs(diff) < 0.02:
                continue

            price = prices.get(symbol, 100)

            if diff > 0:
                self.position_manager.open_position(
                    symbol, 'long', abs(diff) * 1000, price
                )
            else:
                self.position_manager.open_position(
                    symbol, 'short', abs(diff) * 1000, price
                )

    def _display_portfolio_status(self, prices):
        """Display current portfolio status"""
        self.position_manager.update_prices(prices)
        metrics = self.position_manager.get_portfolio_metrics()

        logger.info("\nPortfolio Status:")
        logger.info(f"  Total Value: ${metrics.total_value:,.2f}")
        logger.info(f"  Total P&L: ${metrics.total_pnl:,.2f} ({metrics.total_pnl/metrics.total_value*100:.2f}%)")
        logger.info(f"  Long Exposure: {metrics.long_exposure:.1%}")
        logger.info(f"  Short Exposure: {metrics.short_exposure:.1%}")
        logger.info(f"  Net Exposure: {metrics.net_exposure:.1%}")
        logger.info(f"  Active Positions: {metrics.num_positions}")

        if metrics.num_positions > 0:
            logger.info(f"  Win Rate: {metrics.win_rate:.1%}")
            logger.info(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")

    def stop(self):
        """Stop the trading bot"""
        self.is_running = False
        logger.info("Stop signal received")


def main():
    """Main entry point"""
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║       ML Crypto Trading Bot v1.0                      ║
    ║       Market-Neutral Long/Short Strategy              ║
    ╚═══════════════════════════════════════════════════════╝
    """)

    bot = MLTradingBot()

    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        bot.stop()
        print("\nBot stopped by user")


if __name__ == "__main__":
    main()
