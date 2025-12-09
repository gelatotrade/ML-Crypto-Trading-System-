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
    MAJOR = "major"           # Full position limits (BTC, ETH)
    ALTCOIN = "altcoin"       # 50% position limits, volatility-adjusted
    MEME = "meme"             # High volatility, inflation-based shorts
    DEFLATIONARY = "deflationary"  # Buyback/burn tokens (HYPE)
    HIGH_UNLOCK = "high_unlock"    # Tokens with significant upcoming unlocks


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
    position_limit_meme: float = 0.05   # 5% max for meme tokens (shorts only)
    position_limit_deflationary: float = 0.15  # 15% max for deflationary (buyback) tokens
    position_limit_high_unlock: float = 0.08   # 8% max for high unlock tokens
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
    # Based on Lighter.xyz available pairs + inflation/deflation strategy tokens
    ASSET_UNIVERSE: Dict[str, AssetType] = {
        # ═══════════════════════════════════════════════════════════════
        # MAJOR ASSETS (50x leverage on Lighter)
        # ═══════════════════════════════════════════════════════════════
        "BTC/USDT": AssetType.MAJOR,
        "ETH/USDT": AssetType.MAJOR,

        # ═══════════════════════════════════════════════════════════════
        # LIGHTER.XYZ PERPETUAL PAIRS (8-15x leverage)
        # ═══════════════════════════════════════════════════════════════
        "SOL/USDT": AssetType.ALTCOIN,
        "AVAX/USDT": AssetType.ALTCOIN,
        "LINK/USDT": AssetType.ALTCOIN,
        "NEAR/USDT": AssetType.ALTCOIN,
        "DOT/USDT": AssetType.ALTCOIN,
        "TON/USDT": AssetType.ALTCOIN,
        "TAO/USDT": AssetType.ALTCOIN,
        "POL/USDT": AssetType.ALTCOIN,   # Polygon (formerly MATIC)

        # ═══════════════════════════════════════════════════════════════
        # MEME TOKENS (HIGH INFLATION - SHORT CANDIDATES)
        # High volatility, often inflationary, targets for shorts
        # ═══════════════════════════════════════════════════════════════
        "DOGE/USDT": AssetType.MEME,
        "PEPE/USDT": AssetType.MEME,
        "WLD/USDT": AssetType.MEME,      # Worldcoin - large unlock schedule
        "SHIB/USDT": AssetType.MEME,
        "BONK/USDT": AssetType.MEME,
        "WIF/USDT": AssetType.MEME,
        "FLOKI/USDT": AssetType.MEME,

        # ═══════════════════════════════════════════════════════════════
        # HIGH UNLOCK TOKENS (TEAM/VC PRESSURE - SHORT CANDIDATES)
        # Tokens with significant upcoming team/investor unlocks
        # ═══════════════════════════════════════════════════════════════
        "ARB/USDT": AssetType.HIGH_UNLOCK,    # Arbitrum - large VC unlocks
        "OP/USDT": AssetType.HIGH_UNLOCK,     # Optimism - team unlocks
        "APT/USDT": AssetType.HIGH_UNLOCK,    # Aptos - investor unlocks
        "SUI/USDT": AssetType.HIGH_UNLOCK,    # Sui - team/investor unlocks
        "SEI/USDT": AssetType.HIGH_UNLOCK,    # Sei - ecosystem unlocks
        "TIA/USDT": AssetType.HIGH_UNLOCK,    # Celestia - unlock schedule
        "JUP/USDT": AssetType.HIGH_UNLOCK,    # Jupiter - team unlocks
        "STRK/USDT": AssetType.HIGH_UNLOCK,   # Starknet - VC unlocks

        # ═══════════════════════════════════════════════════════════════
        # DEFLATIONARY TOKENS (BUYBACK/BURN - LONG CANDIDATES)
        # Tokens with active buyback programs
        # NOTE: HYPE is NOT a hedge asset - uses buyback-based sizing
        # ═══════════════════════════════════════════════════════════════
        "HYPE/USDT": AssetType.DEFLATIONARY,  # 97% fee buyback - LONG when buybacks active
        "BNB/USDT": AssetType.DEFLATIONARY,   # Quarterly burns

        # ═══════════════════════════════════════════════════════════════
        # OTHER ALTCOINS (STANDARD)
        # ═══════════════════════════════════════════════════════════════
        "UNI/USDT": AssetType.ALTCOIN,
        "AAVE/USDT": AssetType.ALTCOIN,
        "CRV/USDT": AssetType.ALTCOIN,
        "LDO/USDT": AssetType.ALTCOIN,
        "IMX/USDT": AssetType.ALTCOIN,
        "INJ/USDT": AssetType.ALTCOIN,
        "FTM/USDT": AssetType.ALTCOIN,
        "ATOM/USDT": AssetType.ALTCOIN,
        "FIL/USDT": AssetType.ALTCOIN,
        "RUNE/USDT": AssetType.ALTCOIN,
    }

    # Manual token additions - users can add custom tokens here
    # Format: {"SYMBOL/USDT": AssetType.TYPE}
    CUSTOM_TOKENS: Dict[str, AssetType] = {}

    @classmethod
    def add_custom_token(cls, symbol: str, asset_type: AssetType):
        """
        Add a custom token to the universe

        Args:
            symbol: Trading pair (e.g., 'NEW/USDT')
            asset_type: Token classification

        Example:
            Config.add_custom_token("NEWTOKEN/USDT", AssetType.ALTCOIN)
        """
        cls.CUSTOM_TOKENS[symbol] = asset_type

    @classmethod
    def remove_custom_token(cls, symbol: str):
        """Remove a custom token from the universe"""
        if symbol in cls.CUSTOM_TOKENS:
            del cls.CUSTOM_TOKENS[symbol]

    @classmethod
    def get_all_tokens(cls) -> Dict[str, AssetType]:
        """Get combined token universe (default + custom)"""
        return {**cls.ASSET_UNIVERSE, **cls.CUSTOM_TOKENS}

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
        all_tokens = self.get_all_tokens()
        asset_type = all_tokens.get(symbol, AssetType.ALTCOIN)

        if asset_type == AssetType.MAJOR:
            return self.risk.position_limit_major
        elif asset_type == AssetType.MEME:
            return self.risk.position_limit_meme
        elif asset_type == AssetType.DEFLATIONARY:
            return self.risk.position_limit_deflationary
        elif asset_type == AssetType.HIGH_UNLOCK:
            return self.risk.position_limit_high_unlock
        else:
            return self.risk.position_limit_altcoin

    def get_asset_type(self, symbol: str) -> AssetType:
        """Get asset type for a symbol"""
        all_tokens = self.get_all_tokens()
        return all_tokens.get(symbol, AssetType.ALTCOIN)

    def get_symbols(self) -> List[str]:
        """Get list of all symbols in universe (default + custom)"""
        return list(self.get_all_tokens().keys())

    def get_symbols_by_type(self, asset_type: AssetType) -> List[str]:
        """Get symbols filtered by asset type"""
        all_tokens = self.get_all_tokens()
        return [s for s, t in all_tokens.items() if t == asset_type]

    def get_meme_tokens(self) -> List[str]:
        """Get all meme tokens (short candidates)"""
        return self.get_symbols_by_type(AssetType.MEME)

    def get_high_unlock_tokens(self) -> List[str]:
        """Get all high unlock tokens (short candidates)"""
        return self.get_symbols_by_type(AssetType.HIGH_UNLOCK)

    def get_deflationary_tokens(self) -> List[str]:
        """Get all deflationary tokens (long candidates during buybacks)"""
        return self.get_symbols_by_type(AssetType.DEFLATIONARY)

    def get_short_candidates(self) -> List[str]:
        """Get all tokens suitable for shorting (meme + high_unlock)"""
        return self.get_meme_tokens() + self.get_high_unlock_tokens()

    def get_long_candidates(self) -> List[str]:
        """Get all tokens suitable for longing (deflationary)"""
        return self.get_deflationary_tokens()

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
