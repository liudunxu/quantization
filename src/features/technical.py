"""Technical indicator features."""

from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
from ..utils.cache import FeatureCache
from ..utils.config import get_config
from .base import BaseFeatureExtractor
from ..data_providers import fetch_stock_data


class TechnicalFeatures(BaseFeatureExtractor):
    """Extract technical indicator features."""

    def __init__(self, cache: Optional[FeatureCache] = None):
        super().__init__(cache)
        self.config = get_config().get_section('features').get('technical', {})
        self.ma_periods = self.config.get('ma_periods', [5, 10, 20, 60])
        self.rsi_period = self.config.get('rsi_period', 14)
        self.macd_fast = self.config.get('macd_fast', 12)
        self.macd_slow = self.config.get('macd_slow', 26)
        self.macd_signal = self.config.get('macd_signal', 9)
        self.bollinger_period = self.config.get('bollinger_period', 20)
        self.bollinger_std = self.config.get('bollinger_std', 2)

    @property
    def feature_type(self) -> str:
        return 'technical'

    def extract(self, stock_code: str, **kwargs) -> pd.DataFrame:
        """Extract technical features for a stock."""
        days = kwargs.get('days', 120)

        # Use multi-provider data fetcher with fallback
        data = fetch_stock_data(stock_code, days=days)
        if data.empty:
            return pd.DataFrame()

        # Ensure lowercase column names (data fetcher returns lowercase)
        data.columns = [c.lower() for c in data.columns]

        # Create df with explicit date column (not using index)
        df = pd.DataFrame()
        df['date'] = pd.to_datetime(data['date'])
        df['stock_code'] = stock_code

        # Price data - handle both uppercase and lowercase
        for col_name in ['close', 'open', 'high', 'low', 'volume']:
            if col_name in data.columns:
                val = data[col_name]
                df[col_name] = val.iloc[:, 0] if isinstance(val, pd.DataFrame) else val
            else:
                raise KeyError(f"Column '{col_name}' not found. Available: {list(data.columns)}")

        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Moving averages
        for period in self.ma_periods:
            df[f'ma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ma_{period}_ratio'] = df['close'] / df[f'ma_{period}']

        # Exponential moving averages
        for period in [12, 26]:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

        # ========== MA Cross features (金叉死叉) ==========
        # Golden cross: short MA crosses above long MA
        # Death cross: short MA crosses below long MA
        ma5_ma10_diff = df['ma_5'] - df['ma_10']
        ma10_ma20_diff = df['ma_10'] - df['ma_20']
        ma5_ma20_diff = df['ma_5'] - df['ma_20']

        # MA5-MA10 cross
        ma5_above_ma10_prev = (ma5_ma10_diff.shift(1) > 0)
        ma5_above_ma10_now = (ma5_ma10_diff > 0)
        df['golden_cross_5_10'] = (ma5_above_ma10_prev == False) & (ma5_above_ma10_now == True)
        df['golden_cross_5_10'] = df['golden_cross_5_10'].astype(int)
        df['death_cross_5_10'] = (ma5_above_ma10_prev == True) & (ma5_above_ma10_now == False)
        df['death_cross_5_10'] = df['death_cross_5_10'].astype(int)

        # MA10-MA20 cross
        ma10_above_ma20_prev = (ma10_ma20_diff.shift(1) > 0)
        ma10_above_ma20_now = (ma10_ma20_diff > 0)
        df['golden_cross_10_20'] = (ma10_above_ma20_prev == False) & (ma10_above_ma20_now == True)
        df['golden_cross_10_20'] = df['golden_cross_10_20'].astype(int)
        df['death_cross_10_20'] = (ma10_above_ma20_prev == True) & (ma10_above_ma20_now == False)
        df['death_cross_10_20'] = df['death_cross_10_20'].astype(int)

        # MA5-MA20 cross
        ma5_above_ma20_prev = (ma5_ma20_diff.shift(1) > 0)
        ma5_above_ma20_now = (ma5_ma20_diff > 0)
        df['golden_cross_5_20'] = (ma5_above_ma20_prev == False) & (ma5_above_ma20_now == True)
        df['golden_cross_5_20'] = df['golden_cross_5_20'].astype(int)
        df['death_cross_5_20'] = (ma5_above_ma20_prev == True) & (ma5_above_ma20_now == False)
        df['death_cross_5_20'] = df['death_cross_5_20'].astype(int)

        # ========== MA Arrangement features (多头/空头排列) ==========
        # Bullish arrangement (多头排列): MA5 > MA10 > MA20
        df['ma_bullish_arrange'] = ((df['ma_5'] > df['ma_10']) & (df['ma_10'] > df['ma_20'])).astype(int)
        # Bearish arrangement (空头排列): MA5 < MA10 < MA20
        df['ma_bearish_arrange'] = ((df['ma_5'] < df['ma_10']) & (df['ma_10'] < df['ma_20'])).astype(int)

        # Partial arrangements
        df['ma_5_above_10'] = (df['ma_5'] > df['ma_10']).astype(int)
        df['ma_10_above_20'] = (df['ma_10'] > df['ma_20']).astype(int)
        df['ma_5_above_20'] = (df['ma_5'] > df['ma_20']).astype(int)

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=self.macd_signal, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=self.bollinger_period).mean()
        bb_std = df['close'].rolling(window=self.bollinger_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * self.bollinger_std)
        df['bb_lower'] = df['bb_middle'] - (bb_std * self.bollinger_std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()

        # Volume indicators
        df['volume_ma5'] = df['volume'].rolling(window=5).mean()
        df['volume_ma20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma20']
        df['volume_change'] = df['volume'].pct_change()

        # Momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1

        # Volatility
        for period in [5, 10, 20]:
            df[f'volatility_{period}'] = df['returns'].rolling(window=period).std()

        # Historical high/low
        df['high_20d'] = df['close'].rolling(window=20).max()
        df['low_20d'] = df['close'].rolling(window=20).min()
        df['high_low_ratio'] = df['close'] / df['high_20d']
        df['close_low_ratio'] = df['close'] / df['low_20d']

        # Price relative to high/low
        df['price_position'] = (df['close'] - df['low_20d']) / (df['high_20d'] - df['low_20d'] + 1e-10)

        # ========== Short-term momentum features ==========
        # return_2d, return_3d: 2-day and 3-day returns
        df['return_2d'] = df['close'].pct_change(2)
        df['return_3d'] = df['close'].pct_change(3)

        # momentum_acceleration: return_5d - return_10d (动量加速/减速)
        df['momentum_acceleration'] = df['momentum_5'] - df['momentum_10']

        # ========== Risk-adjusted features ==========
        # volatility_20d: 20-day volatility (annualized)
        df['volatility_20d'] = df['returns'].rolling(window=20).std() * np.sqrt(252)

        # sharpe_like: return_5d / volatility_20d
        df['sharpe_like'] = df['momentum_5'] / (df['volatility_20d'] + 1e-8)

        # cv: coefficient of variation (std/mean) for returns stability
        df['cv'] = df['returns'].rolling(window=20).std() / (df['returns'].rolling(window=20).mean().abs() + 1e-8)

        # max_drawdown_20d: 20-day maximum drawdown
        rolling_max = df['close'].rolling(window=20).max()
        drawdown = (df['close'] - rolling_max) / rolling_max
        df['max_drawdown_20d'] = drawdown.rolling(window=20).min()

        # ========== Extra technical indicators ==========
        # ADX_14 (Average Directional Index)
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0).rolling(window=14).mean()
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0).rolling(window=14).mean()
        atr_14 = df['atr']
        plus_di = 100 * plus_dm / (atr_14 + 1e-8)
        minus_di = 100 * minus_dm / (atr_14 + 1e-8)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        df['adx'] = dx

        # Stochastic Oscillator (%K and %D, 14-period)
        lowest_low = df['low'].rolling(window=14).min()
        highest_high = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low + 1e-8)
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

        # MFI_14 (Money Flow Index, 14-period)
        typical_price = (df['high'] + df['low'] + df['close']) / 3.0
        money_flow = typical_price * df['volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=14).sum()
        mfi = 100 - (100 / (1 + positive_flow / (negative_flow + 1e-8)))
        df['mfi'] = mfi

        # CCI_14 (Commodity Channel Index, 14-period)
        tp = (df['high'] + df['low'] + df['close']) / 3.0
        sma_tp = tp.rolling(window=14).mean()
        mad = (tp - sma_tp).abs().rolling(window=14).mean()
        df['cci'] = (tp - sma_tp) / (0.015 * mad + 1e-8)

        # ========== Streak features (连续涨跌) ==========
        # Daily direction: 1 for up, -1 for down, 0 for unchanged
        daily_dir = np.sign(df['close'].diff())
        daily_dir = daily_dir.replace(0, np.nan).ffill().fillna(0)

        # Count consecutive up/down days
        # Consecutive up days
        df['streak_up_2'] = (daily_dir.shift(1) >= 1).rolling(window=2).sum() >= 2
        df['streak_up_2'] = df['streak_up_2'].astype(int)
        df['streak_up_3'] = (daily_dir.shift(1) >= 1).rolling(window=3).sum() >= 3
        df['streak_up_3'] = df['streak_up_3'].astype(int)

        # Consecutive down days
        df['streak_down_2'] = (daily_dir.shift(1) <= -1).rolling(window=2).sum() >= 2
        df['streak_down_2'] = df['streak_down_2'].astype(int)
        df['streak_down_3'] = (daily_dir.shift(1) <= -1).rolling(window=3).sum() >= 3
        df['streak_down_3'] = df['streak_down_3'].astype(int)

        # ========== Pattern features (10日内量价模式) ==========
        # Returns and volume changes over last 10 days
        ret_10d = df['close'].pct_change(10)
        vol_change_10d = df['volume'].pct_change(10)

        # Big drop then big rise (>5% drop, >5% rise in next period)
        ret_5d_later = df['close'].shift(-5).pct_change(5)
        df['pattern_drop_then_rise'] = ((ret_10d < -0.05) & (ret_5d_later > 0.05)).astype(int)

        # Big rise then big drop
        df['pattern_rise_then_drop'] = ((ret_10d > 0.05) & (ret_5d_later < -0.05)).astype(int)

        # Volume surge pattern: massive volume increase followed by continued rise
        vol_10d_ago = df['volume'].shift(10)
        vol_now = df['volume']
        df['pattern_vol_surge_rise'] = ((vol_now / vol_10d_ago > 2.0) & (ret_10d > 0)).astype(int)
        df['pattern_vol_surge_drop'] = ((vol_now / vol_10d_ago > 2.0) & (ret_10d < 0)).astype(int)

        # Recent momentum reversal: last 3 days opposite to prior 7 days
        ret_3d = df['close'].pct_change(3)
        ret_7d_prior = df['close'].shift(3).pct_change(7)
        df['pattern_reversal_up'] = ((ret_7d_prior < -0.03) & (ret_3d > 0.03)).astype(int)
        df['pattern_reversal_down'] = ((ret_7d_prior > 0.03) & (ret_3d < -0.03)).astype(int)

        # Fill NaN values with forward fill then backward fill for remaining
        df = df.ffill().bfill()

        return df
