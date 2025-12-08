"""
Configuration Management for ML Crypto Trading System
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from dotenv import load_dotenv


class TradingMode(Enum):
    PAPER = "paper"
    LIVE = "live"
    BACKTEST = "backtest"


class AssetType(Enum):
    MAJOR = "major"        # Full position limits (BTC, ETH)
    ALTCOIN = "altcoin"    # 50% position limits, volatility-adjusted
    HEDGE = "hedge"        # 150% position limits (HYPE)


@dataclass
class RiskConfig:
    """Risk management configuration"""
    max_leverage: float = 3.0
    max_net_exposure: float = 0.1       # 10% max net for neutrality
    max_gross_exposure: float = 2.0     # 200% max gross
    target_volatility: float = 0.15     # 15% annualized target
    max_drawdown: float = 0.15          # 15% max drawdown
    var_confidence: float = 0.95        # 95% VaR confidence
    position_limit_major: float = 0.20  # 20% max for majors
    position_limit_altcoin: float = 0.10  # 10% max for altcoins
    position_limit_hedge: float = 0.30   # 30% max for hedge assets
    min_sharpe_ratio: float = 2.5       # Minimum target Sharpe


@dataclass
class TradingConfig:
    """Trading configuration"""
    mode: TradingMode = TradingMode.PAPER
    primary_exchange: str = "lighter"
    backup_exchange: str = "hyperliquid"
    rebalance_interval: int = 3600      # 1 hour in seconds
    min_trade_size: float = 10.0        # Minimum trade size in USD
    slippage_tolerance: float = 0.002   # 0.2% slippage tolerance
    max_order_retries: int = 3
    order_timeout: int = 30             # seconds


@dataclass
class DataConfig:
    """Data collection configuration"""
    lookback_days: int = 30
    ohlcv_timeframe: str = "1h"
    orderbook_depth: int = 20
    cache_ttl: int = 300                # 5 minutes cache TTL


@dataclass
class MLConfig:
    """Machine learning configuration"""
    model_path: str = "models/"
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    sequence_length: int = 24           # 24 hours for hourly data
    prediction_horizon: int = 1         # 1 hour ahead
    min_confidence: float = 0.6         # Minimum prediction confidence
    ensemble_models: List[str] = field(default_factory=lambda: [
        "lstm", "random_forest", "gradient_boosting"
    ])


@dataclass
class OptionsConfig:
    """Options analysis configuration"""
    enabled: bool = True
    iv_weight: float = 0.6              # 60% options IV, 40% historical
    dvol_enabled: bool = True           # Deribit DVOL index
    risk_free_rate: float = 0.05        # 5% risk-free rate


@dataclass
class RegimeConfig:
    """Regime detection configuration"""
    enabled: bool = True
    risk_on_beta: float = 0.3           # Long bias in risk-on
    risk_off_beta: float = -0.3         # Short bias in risk-off
    neutral_beta: float = 0.0           # Market neutral


class Config:
    """Main configuration class"""

    # Asset universe with classifications
    ASSET_UNIVERSE: Dict[str, AssetType] = {
        # Major assets
        "BTC/USDT": AssetType.MAJOR,
        "ETH/USDT": AssetType.MAJOR,
        # Mid-cap altcoins
        "SOL/USDT": AssetType.ALTCOIN,
        "AVAX/USDT": AssetType.ALTCOIN,
        "MATIC/USDT": AssetType.ALTCOIN,
        "LINK/USDT": AssetType.ALTCOIN,
        # Smaller altcoins
        "UNI/USDT": AssetType.ALTCOIN,
        "AAVE/USDT": AssetType.ALTCOIN,
        "SUSHI/USDT": AssetType.ALTCOIN,
        "CRV/USDT": AssetType.ALTCOIN,
        "LDO/USDT": AssetType.ALTCOIN,
        "ARB/USDT": AssetType.ALTCOIN,
        "OP/USDT": AssetType.ALTCOIN,
        "IMX/USDT": AssetType.ALTCOIN,
        # Hedge asset
        "HYPE/USDT": AssetType.HEDGE,
    }

    def __init__(self, env_path: Optional[str] = None):
        """Initialize configuration from environment"""
        if env_path:
            load_dotenv(env_path)
        else:
            load_dotenv()

        self.risk = self._load_risk_config()
        self.trading = self._load_trading_config()
        self.data = self._load_data_config()
        self.ml = self._load_ml_config()
        self.options = self._load_options_config()
        self.regime = self._load_regime_config()
        self.api_keys = self._load_api_keys()

    def _load_risk_config(self) -> RiskConfig:
        """Load risk configuration from environment"""
        return RiskConfig(
            max_leverage=float(os.getenv("MAX_LEVERAGE", "3.0")),
            max_net_exposure=float(os.getenv("MAX_NET_EXPOSURE", "0.1")),
            max_gross_exposure=float(os.getenv("MAX_GROSS_EXPOSURE", "2.0")),
            target_volatility=float(os.getenv("TARGET_VOLATILITY", "0.15")),
            max_drawdown=float(os.getenv("MAX_DRAWDOWN", "0.15")),
            var_confidence=float(os.getenv("VAR_CONFIDENCE", "0.95")),
            position_limit_major=float(os.getenv("POSITION_LIMIT_MAJOR", "0.20")),
            position_limit_altcoin=float(os.getenv("POSITION_LIMIT_ALTCOIN", "0.10")),
            position_limit_hedge=float(os.getenv("POSITION_LIMIT_HEDGE", "0.30")),
            min_sharpe_ratio=float(os.getenv("MIN_SHARPE_RATIO", "2.5")),
        )

    def _load_trading_config(self) -> TradingConfig:
        """Load trading configuration from environment"""
        mode_str = os.getenv("TRADING_MODE", "paper").lower()
        mode = TradingMode(mode_str) if mode_str in [m.value for m in TradingMode] else TradingMode.PAPER

        return TradingConfig(
            mode=mode,
            primary_exchange=os.getenv("PRIMARY_EXCHANGE", "lighter"),
            backup_exchange=os.getenv("BACKUP_EXCHANGE", "hyperliquid"),
            rebalance_interval=int(os.getenv("REBALANCE_INTERVAL", "3600")),
            min_trade_size=float(os.getenv("MIN_TRADE_SIZE", "10.0")),
            slippage_tolerance=float(os.getenv("SLIPPAGE_TOLERANCE", "0.002")),
            max_order_retries=int(os.getenv("MAX_ORDER_RETRIES", "3")),
            order_timeout=int(os.getenv("ORDER_TIMEOUT", "30")),
        )

    def _load_data_config(self) -> DataConfig:
        """Load data configuration from environment"""
        return DataConfig(
            lookback_days=int(os.getenv("LOOKBACK_DAYS", "30")),
            ohlcv_timeframe=os.getenv("OHLCV_TIMEFRAME", "1h"),
            orderbook_depth=int(os.getenv("ORDERBOOK_DEPTH", "20")),
            cache_ttl=int(os.getenv("CACHE_TTL", "300")),
        )

    def _load_ml_config(self) -> MLConfig:
        """Load ML configuration from environment"""
        return MLConfig(
            model_path=os.getenv("MODEL_PATH", "models/"),
            lstm_hidden_size=int(os.getenv("LSTM_HIDDEN_SIZE", "128")),
            lstm_num_layers=int(os.getenv("LSTM_NUM_LAYERS", "2")),
            lstm_dropout=float(os.getenv("LSTM_DROPOUT", "0.2")),
            sequence_length=int(os.getenv("SEQUENCE_LENGTH", "24")),
            prediction_horizon=int(os.getenv("PREDICTION_HORIZON", "1")),
            min_confidence=float(os.getenv("MIN_CONFIDENCE", "0.6")),
        )

    def _load_options_config(self) -> OptionsConfig:
        """Load options configuration from environment"""
        return OptionsConfig(
            enabled=os.getenv("OPTIONS_ENABLED", "true").lower() == "true",
            iv_weight=float(os.getenv("IV_WEIGHT", "0.6")),
            dvol_enabled=os.getenv("DVOL_ENABLED", "true").lower() == "true",
            risk_free_rate=float(os.getenv("RISK_FREE_RATE", "0.05")),
        )

    def _load_regime_config(self) -> RegimeConfig:
        """Load regime detection configuration from environment"""
        return RegimeConfig(
            enabled=os.getenv("REGIME_DETECTION_ENABLED", "true").lower() == "true",
            risk_on_beta=float(os.getenv("RISK_ON_BETA", "0.3")),
            risk_off_beta=float(os.getenv("RISK_OFF_BETA", "-0.3")),
            neutral_beta=float(os.getenv("NEUTRAL_BETA", "0.0")),
        )

    def _load_api_keys(self) -> Dict[str, Any]:
        """Load API keys from environment"""
        return {
            # Trading exchanges
            "lighter": {
                "api_key": os.getenv("LIGHTER_API_KEY", ""),
                "api_secret": os.getenv("LIGHTER_API_SECRET", ""),
                "enable_fast_execution": os.getenv("LIGHTER_ENABLE_FAST_EXECUTION", "true").lower() == "true",
                "fast_execution_fee_multiplier": float(os.getenv("LIGHTER_FAST_EXECUTION_FEE", "1.5")),
            },
            "hyperliquid": {
                "api_key": os.getenv("HYPERLIQUID_API_KEY", ""),
                "api_secret": os.getenv("HYPERLIQUID_API_SECRET", ""),
                "private_key": os.getenv("DEX_PRIVATE_KEY", ""),
            },
            # Market data sources
            "yahoo_finance": {
                "enabled": os.getenv("YAHOO_FINANCE_ENABLED", "true").lower() == "true",
            },
            "alpha_vantage": {
                "api_key": os.getenv("ALPHA_VANTAGE_API_KEY", ""),
                "enabled": bool(os.getenv("ALPHA_VANTAGE_API_KEY", "")),
            },
            "fmp": {
                "api_key": os.getenv("FMP_API_KEY", ""),
                "enabled": bool(os.getenv("FMP_API_KEY", "")),
            },
            "polygon": {
                "api_key": os.getenv("POLYGON_API_KEY", ""),
                "enabled": bool(os.getenv("POLYGON_API_KEY", "")),
            },
            # Options data
            "deribit": {
                "client_id": os.getenv("DERIBIT_CLIENT_ID", ""),
                "client_secret": os.getenv("DERIBIT_CLIENT_SECRET", ""),
                "testnet": os.getenv("DERIBIT_TESTNET", "true").lower() == "true",
            },
        }

    def get_position_limit(self, symbol: str) -> float:
        """Get position limit for a symbol based on asset type"""
        asset_type = self.ASSET_UNIVERSE.get(symbol, AssetType.ALTCOIN)

        if asset_type == AssetType.MAJOR:
            return self.risk.position_limit_major
        elif asset_type == AssetType.HEDGE:
            return self.risk.position_limit_hedge
        else:
            return self.risk.position_limit_altcoin

    def get_asset_type(self, symbol: str) -> AssetType:
        """Get asset type for a symbol"""
        return self.ASSET_UNIVERSE.get(symbol, AssetType.ALTCOIN)

    def get_symbols(self) -> List[str]:
        """Get list of all symbols in universe"""
        return list(self.ASSET_UNIVERSE.keys())

    def is_paper_mode(self) -> bool:
        """Check if running in paper trading mode"""
        return self.trading.mode == TradingMode.PAPER

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "risk": {
                "max_leverage": self.risk.max_leverage,
                "max_net_exposure": self.risk.max_net_exposure,
                "max_gross_exposure": self.risk.max_gross_exposure,
                "target_volatility": self.risk.target_volatility,
                "max_drawdown": self.risk.max_drawdown,
                "var_confidence": self.risk.var_confidence,
                "min_sharpe_ratio": self.risk.min_sharpe_ratio,
            },
            "trading": {
                "mode": self.trading.mode.value,
                "primary_exchange": self.trading.primary_exchange,
                "rebalance_interval": self.trading.rebalance_interval,
            },
            "data": {
                "lookback_days": self.data.lookback_days,
                "ohlcv_timeframe": self.data.ohlcv_timeframe,
            },
            "ml": {
                "sequence_length": self.ml.sequence_length,
                "prediction_horizon": self.ml.prediction_horizon,
                "min_confidence": self.ml.min_confidence,
            },
            "options": {
                "enabled": self.options.enabled,
                "iv_weight": self.options.iv_weight,
            },
            "regime": {
                "enabled": self.regime.enabled,
                "risk_on_beta": self.regime.risk_on_beta,
                "risk_off_beta": self.regime.risk_off_beta,
            },
        }
