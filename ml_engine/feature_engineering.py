"""
Feature Engineering - 100+ Technical Indicators
Includes RSI, MACD, Bollinger Bands, ATR, OBV, VWAP, volatility metrics, and more
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering class with 100+ technical indicators.
    """

    def __init__(self, config=None):
        """Initialize feature engineer"""
        self.config = config
        self.feature_names: List[str] = []

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all technical features for a DataFrame.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with all features added
        """
        df = df.copy()

        # Price-based features
        df = self._add_price_features(df)

        # Momentum indicators
        df = self._add_momentum_indicators(df)

        # Volatility indicators
        df = self._add_volatility_indicators(df)

        # Volume indicators
        df = self._add_volume_indicators(df)

        # Trend indicators
        df = self._add_trend_indicators(df)

        # Pattern recognition
        df = self._add_pattern_features(df)

        # Statistical features
        df = self._add_statistical_features(df)

        # Time-based features
        df = self._add_time_features(df)

        # Cross-sectional features
        df = self._add_cross_features(df)

        # Store feature names
        self.feature_names = [col for col in df.columns if col not in
                             ['open', 'high', 'low', 'close', 'volume', 'symbol', 'data_source']]

        logger.info(f"Created {len(self.feature_names)} features")
        return df

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Price changes
        for period in [1, 3, 5, 10, 20]:
            df[f'return_{period}'] = df['close'].pct_change(period)
            df[f'log_return_{period}'] = np.log(df['close'] / df['close'].shift(period))

        # Price ratios
        df['hl_ratio'] = df['high'] / df['low']
        df['oc_ratio'] = df['open'] / df['close']
        df['close_open_diff'] = (df['close'] - df['open']) / df['open']

        # Gaps
        df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        df['gap_filled'] = ((df['low'] <= df['close'].shift(1)) | (df['high'] >= df['close'].shift(1))).astype(int)

        # Range features
        df['true_range'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['hl_pct'] = (df['high'] - df['low']) / df['close']

        return df

    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators"""
        # RSI (multiple periods)
        for period in [7, 14, 21]:
            df[f'rsi_{period}'] = self._calc_rsi(df['close'], period)

        # Stochastic RSI
        df['stoch_rsi'] = self._calc_stoch_rsi(df['close'], 14)

        # MACD
        macd, signal, hist = self._calc_macd(df['close'])
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
        df['macd_cross'] = np.sign(macd - signal)

        # Rate of Change (ROC)
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period) * 100

        # Momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)

        # Williams %R
        for period in [7, 14, 21]:
            df[f'williams_r_{period}'] = self._calc_williams_r(df, period)

        # Commodity Channel Index (CCI)
        for period in [14, 20]:
            df[f'cci_{period}'] = self._calc_cci(df, period)

        # Ultimate Oscillator
        df['ultimate_osc'] = self._calc_ultimate_oscillator(df)

        # Money Flow Index
        df['mfi_14'] = self._calc_mfi(df, 14)

        return df

    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators"""
        # Bollinger Bands (multiple periods)
        for period in [10, 20, 50]:
            bb_upper, bb_middle, bb_lower = self._calc_bollinger_bands(df['close'], period)
            df[f'bb_upper_{period}'] = bb_upper
            df[f'bb_middle_{period}'] = bb_middle
            df[f'bb_lower_{period}'] = bb_lower
            df[f'bb_width_{period}'] = (bb_upper - bb_lower) / bb_middle
            df[f'bb_pct_{period}'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)

        # Average True Range (ATR)
        for period in [7, 14, 21]:
            df[f'atr_{period}'] = self._calc_atr(df, period)

        # ATR Percent
        df['atr_pct'] = df['atr_14'] / df['close'] * 100

        # Keltner Channels
        kc_upper, kc_middle, kc_lower = self._calc_keltner_channels(df, 20, 2)
        df['kc_upper'] = kc_upper
        df['kc_middle'] = kc_middle
        df['kc_lower'] = kc_lower

        # Donchian Channels
        for period in [10, 20]:
            df[f'donchian_upper_{period}'] = df['high'].rolling(period).max()
            df[f'donchian_lower_{period}'] = df['low'].rolling(period).min()
            df[f'donchian_mid_{period}'] = (df[f'donchian_upper_{period}'] + df[f'donchian_lower_{period}']) / 2

        # Historical Volatility
        for period in [10, 20, 60]:
            df[f'volatility_{period}'] = df['log_returns'].rolling(period).std() * np.sqrt(252 * 24)

        # Realized Volatility (Parkinson)
        for period in [10, 20]:
            df[f'parkinson_vol_{period}'] = np.sqrt(
                (1 / (4 * np.log(2))) *
                ((np.log(df['high'] / df['low'])) ** 2).rolling(period).mean()
            ) * np.sqrt(252 * 24)

        # Garman-Klass Volatility
        df['gk_volatility'] = self._calc_garman_klass_vol(df, 20)

        return df

    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume indicators"""
        # On-Balance Volume (OBV)
        df['obv'] = self._calc_obv(df)
        df['obv_ma_20'] = df['obv'].rolling(20).mean()
        df['obv_trend'] = df['obv'] - df['obv_ma_20']

        # Volume Moving Averages
        for period in [5, 10, 20, 50]:
            df[f'volume_ma_{period}'] = df['volume'].rolling(period).mean()

        # Volume Ratio
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

        # Accumulation/Distribution Line
        df['ad_line'] = self._calc_ad_line(df)
        df['ad_line_ma'] = df['ad_line'].rolling(20).mean()

        # Chaikin Money Flow
        df['cmf'] = self._calc_cmf(df, 20)

        # VWAP
        df['vwap'] = self._calc_vwap(df)
        df['vwap_distance'] = (df['close'] - df['vwap']) / df['vwap'] * 100

        # Force Index
        for period in [2, 13]:
            df[f'force_index_{period}'] = self._calc_force_index(df, period)

        # Ease of Movement
        df['eom'] = self._calc_eom(df, 14)

        # Volume Price Trend
        df['vpt'] = self._calc_vpt(df)

        return df

    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicators"""
        # Simple Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'sma_dist_{period}'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}'] * 100

        # Exponential Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            df[f'ema_dist_{period}'] = (df['close'] - df[f'ema_{period}']) / df[f'ema_{period}'] * 100

        # Moving Average Crossovers
        df['sma_5_20_cross'] = np.sign(df['sma_5'] - df['sma_20'])
        df['sma_20_50_cross'] = np.sign(df['sma_20'] - df['sma_50'])
        df['ema_5_20_cross'] = np.sign(df['ema_5'] - df['ema_20'])

        # ADX (Average Directional Index)
        adx, plus_di, minus_di = self._calc_adx(df, 14)
        df['adx'] = adx
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        df['di_diff'] = plus_di - minus_di

        # Parabolic SAR
        df['psar'] = self._calc_parabolic_sar(df)
        df['psar_direction'] = np.sign(df['close'] - df['psar'])

        # Ichimoku Cloud
        tenkan, kijun, senkou_a, senkou_b, chikou = self._calc_ichimoku(df)
        df['ichimoku_tenkan'] = tenkan
        df['ichimoku_kijun'] = kijun
        df['ichimoku_senkou_a'] = senkou_a
        df['ichimoku_senkou_b'] = senkou_b
        df['ichimoku_chikou'] = chikou
        df['ichimoku_cloud'] = np.sign(senkou_a - senkou_b)

        # TRIX
        df['trix'] = self._calc_trix(df, 15)

        # Mass Index
        df['mass_index'] = self._calc_mass_index(df, 9, 25)

        return df

    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick pattern features"""
        # Body and wick sizes
        df['body_size'] = abs(df['close'] - df['open']) / df['close']
        df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
        df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']

        # Candle direction
        df['candle_direction'] = np.sign(df['close'] - df['open'])

        # Doji detection
        df['is_doji'] = (df['body_size'] < 0.001).astype(int)

        # Hammer/Shooting Star
        df['is_hammer'] = ((df['lower_wick'] > df['body_size'] * 2) &
                          (df['upper_wick'] < df['body_size'] * 0.5)).astype(int)
        df['is_shooting_star'] = ((df['upper_wick'] > df['body_size'] * 2) &
                                  (df['lower_wick'] < df['body_size'] * 0.5)).astype(int)

        # Engulfing patterns
        df['bullish_engulfing'] = ((df['close'] > df['open']) &
                                   (df['close'].shift(1) < df['open'].shift(1)) &
                                   (df['open'] < df['close'].shift(1)) &
                                   (df['close'] > df['open'].shift(1))).astype(int)
        df['bearish_engulfing'] = ((df['close'] < df['open']) &
                                   (df['close'].shift(1) > df['open'].shift(1)) &
                                   (df['open'] > df['close'].shift(1)) &
                                   (df['close'] < df['open'].shift(1))).astype(int)

        # Higher highs / lower lows
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)

        return df

    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features"""
        # Skewness
        for period in [10, 20, 60]:
            df[f'skewness_{period}'] = df['returns'].rolling(period).skew()

        # Kurtosis
        for period in [10, 20, 60]:
            df[f'kurtosis_{period}'] = df['returns'].rolling(period).kurt()

        # Z-score of close
        for period in [10, 20, 50]:
            mean = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            df[f'zscore_{period}'] = (df['close'] - mean) / std

        # Autocorrelation
        for lag in [1, 5, 10]:
            df[f'autocorr_{lag}'] = df['returns'].rolling(20).apply(
                lambda x: x.autocorr(lag) if len(x) > lag else np.nan
            )

        # Rolling correlation with volume
        df['price_volume_corr'] = df['returns'].rolling(20).corr(df['volume'].pct_change())

        # Hurst Exponent (simplified)
        df['hurst_proxy'] = self._calc_hurst_proxy(df['close'], 20)

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        if isinstance(df.index, pd.DatetimeIndex):
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['day_of_month'] = df.index.day
            df['month'] = df.index.month
            df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

            # Cyclical encoding
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        return df

    def _add_cross_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cross-sectional/interaction features"""
        # RSI divergence from price trend
        df['rsi_price_divergence'] = df['rsi_14'].diff(5) * np.sign(df['return_5'])

        # Volume-price relationship
        df['price_volume_trend'] = df['returns'] * df['volume_ratio']

        # Momentum vs volatility
        df['momentum_vol_ratio'] = df['momentum_10'] / (df['volatility_10'] + 1e-8)

        # Composite indicators
        df['trend_strength'] = (df['adx'] * df['di_diff']).fillna(0)
        df['momentum_composite'] = (df['rsi_14'] - 50) * df['macd_hist'] / 100

        return df

    # Helper calculation methods
    def _calc_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def _calc_stoch_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Stochastic RSI"""
        rsi = self._calc_rsi(prices, period)
        rsi_min = rsi.rolling(period).min()
        rsi_max = rsi.rolling(period).max()
        return (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10)

    def _calc_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def _calc_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        highest_high = df['high'].rolling(period).max()
        lowest_low = df['low'].rolling(period).min()
        return -100 * (highest_high - df['close']) / (highest_high - lowest_low + 1e-10)

    def _calc_cci(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Commodity Channel Index"""
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
        return (tp - sma) / (0.015 * mad + 1e-10)

    def _calc_ultimate_oscillator(self, df: pd.DataFrame, p1: int = 7, p2: int = 14, p3: int = 28) -> pd.Series:
        """Calculate Ultimate Oscillator"""
        bp = df['close'] - df[['low', 'close']].shift(1).min(axis=1)
        tr = df[['high', 'close']].shift(1).max(axis=1) - df[['low', 'close']].shift(1).min(axis=1)

        avg1 = bp.rolling(p1).sum() / tr.rolling(p1).sum()
        avg2 = bp.rolling(p2).sum() / tr.rolling(p2).sum()
        avg3 = bp.rolling(p3).sum() / tr.rolling(p3).sum()

        return 100 * (4 * avg1 + 2 * avg2 + avg3) / 7

    def _calc_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index"""
        tp = (df['high'] + df['low'] + df['close']) / 3
        mf = tp * df['volume']
        mf_pos = mf.where(tp > tp.shift(1), 0).rolling(period).sum()
        mf_neg = mf.where(tp < tp.shift(1), 0).rolling(period).sum()
        return 100 - 100 / (1 + mf_pos / (mf_neg + 1e-10))

    def _calc_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple:
        """Calculate Bollinger Bands"""
        middle = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        return upper, middle, lower

    def _calc_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _calc_keltner_channels(self, df: pd.DataFrame, period: int = 20, mult: float = 2) -> Tuple:
        """Calculate Keltner Channels"""
        middle = df['close'].ewm(span=period, adjust=False).mean()
        atr = self._calc_atr(df, period)
        upper = middle + mult * atr
        lower = middle - mult * atr
        return upper, middle, lower

    def _calc_garman_klass_vol(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Garman-Klass Volatility"""
        log_hl = np.log(df['high'] / df['low']) ** 2
        log_co = np.log(df['close'] / df['open']) ** 2
        return np.sqrt((0.5 * log_hl - (2 * np.log(2) - 1) * log_co).rolling(period).mean()) * np.sqrt(252 * 24)

    def _calc_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        obv = pd.Series(index=df.index, dtype=float)
        obv.iloc[0] = df['volume'].iloc[0]

        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]

        return obv

    def _calc_ad_line(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Accumulation/Distribution Line"""
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
        return (clv * df['volume']).cumsum()

    def _calc_cmf(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Chaikin Money Flow"""
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
        return (clv * df['volume']).rolling(period).sum() / df['volume'].rolling(period).sum()

    def _calc_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate VWAP"""
        tp = (df['high'] + df['low'] + df['close']) / 3
        return (tp * df['volume']).cumsum() / df['volume'].cumsum()

    def _calc_force_index(self, df: pd.DataFrame, period: int = 13) -> pd.Series:
        """Calculate Force Index"""
        fi = df['close'].diff() * df['volume']
        return fi.ewm(span=period, adjust=False).mean()

    def _calc_eom(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Ease of Movement"""
        dm = ((df['high'] + df['low']) / 2) - ((df['high'].shift(1) + df['low'].shift(1)) / 2)
        br = df['volume'] / (df['high'] - df['low'] + 1e-10)
        return (dm / br).rolling(period).mean()

    def _calc_vpt(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Price Trend"""
        return (df['volume'] * df['close'].pct_change()).cumsum()

    def _calc_adx(self, df: pd.DataFrame, period: int = 14) -> Tuple:
        """Calculate ADX, +DI, -DI"""
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff().multiply(-1)

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr = self._calc_atr(df, 1) * period

        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / tr)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / tr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(span=period, adjust=False).mean()

        return adx, plus_di, minus_di

    def _calc_parabolic_sar(self, df: pd.DataFrame, af_start: float = 0.02, af_max: float = 0.2) -> pd.Series:
        """Calculate Parabolic SAR (simplified)"""
        psar = df['close'].copy()
        psar.iloc[:2] = df['low'].iloc[:2]

        bull = True
        af = af_start
        ep = df['high'].iloc[0]

        for i in range(2, len(df)):
            if bull:
                psar.iloc[i] = psar.iloc[i-1] + af * (ep - psar.iloc[i-1])
                if df['low'].iloc[i] < psar.iloc[i]:
                    bull = False
                    psar.iloc[i] = ep
                    ep = df['low'].iloc[i]
                    af = af_start
                else:
                    if df['high'].iloc[i] > ep:
                        ep = df['high'].iloc[i]
                        af = min(af + af_start, af_max)
            else:
                psar.iloc[i] = psar.iloc[i-1] - af * (psar.iloc[i-1] - ep)
                if df['high'].iloc[i] > psar.iloc[i]:
                    bull = True
                    psar.iloc[i] = ep
                    ep = df['high'].iloc[i]
                    af = af_start
                else:
                    if df['low'].iloc[i] < ep:
                        ep = df['low'].iloc[i]
                        af = min(af + af_start, af_max)

        return psar

    def _calc_ichimoku(self, df: pd.DataFrame) -> Tuple:
        """Calculate Ichimoku Cloud components"""
        # Tenkan-sen (Conversion Line)
        tenkan = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2

        # Kijun-sen (Base Line)
        kijun = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2

        # Senkou Span A (Leading Span A)
        senkou_a = ((tenkan + kijun) / 2).shift(26)

        # Senkou Span B (Leading Span B)
        senkou_b = ((df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2).shift(26)

        # Chikou Span (Lagging Span)
        chikou = df['close'].shift(-26)

        return tenkan, kijun, senkou_a, senkou_b, chikou

    def _calc_trix(self, df: pd.DataFrame, period: int = 15) -> pd.Series:
        """Calculate TRIX indicator"""
        ema1 = df['close'].ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        return ema3.pct_change() * 100

    def _calc_mass_index(self, df: pd.DataFrame, period1: int = 9, period2: int = 25) -> pd.Series:
        """Calculate Mass Index"""
        ema_range = (df['high'] - df['low']).ewm(span=period1, adjust=False).mean()
        double_ema_range = ema_range.ewm(span=period1, adjust=False).mean()
        return (ema_range / double_ema_range).rolling(period2).sum()

    def _calc_hurst_proxy(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """Calculate simplified Hurst exponent proxy"""
        log_returns = np.log(prices / prices.shift(1))
        rs_list = []

        for i in range(len(prices)):
            if i < period:
                rs_list.append(np.nan)
            else:
                window = log_returns.iloc[i-period:i]
                cumsum = (window - window.mean()).cumsum()
                r = cumsum.max() - cumsum.min()
                s = window.std()
                if s > 0:
                    rs_list.append(r / s)
                else:
                    rs_list.append(np.nan)

        return pd.Series(rs_list, index=prices.index)

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        return self.feature_names

    def select_features(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Select specific features from DataFrame"""
        return df[features].copy()
