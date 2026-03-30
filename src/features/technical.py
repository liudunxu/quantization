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

        # ========== MA120 (120日均线) ==========
        if 120 not in self.ma_periods:
            df['ma_120'] = df['close'].rolling(window=120).mean()
            df['ma_120_ratio'] = df['close'] / df['ma_120']

        # ========== MA Slope features (均线斜率) ==========
        # MA20 slope: rate of change of MA20
        df['ma_slope_20'] = df['ma_20'].pct_change(5) / 5  # 5-day rate of change per day

        # ========== Deviation features (乖离率) ==========
        # Absolute deviation from moving averages
        df['deviation_ma5_abs'] = np.abs((df['close'] - df['ma_5']) / df['ma_5'] * 100)
        df['deviation_ma10_abs'] = np.abs((df['close'] - df['ma_10']) / df['ma_10'] * 100)
        df['deviation_ma20_abs'] = np.abs((df['close'] - df['ma_20']) / df['ma_20'] * 100)

        # ========== MACD Cross features (MACD交叉) ==========
        macd_diff = df['macd'] - df['macd_signal']
        macd_diff_prev = macd_diff.shift(1)
        # MACD金叉: MACD从负转正或上穿信号线
        df['macd_cross_up'] = ((macd_diff_prev <= 0) & (macd_diff > 0)).astype(int)
        # MACD在零轴上方金叉
        df['macd_cross_above_zero'] = ((df['macd'] > 0) & (macd_diff_prev <= 0) & (macd_diff > 0)).astype(int)
        # MACD在零轴上方
        df['macd_position'] = (df['macd'] > 0).astype(int)

        # ========== MACD Divergence features (MACD背驰) ==========
        # Top divergence: price makes new high but MACD histogram decreases
        price_high = df['close'].rolling(window=20).max()
        macd_hist_20 = df['macd_hist'].rolling(window=20).max()
        macd_hist_20_min = df['macd_hist'].rolling(window=20).min()

        # Price创新高但MACD红柱缩小
        price_new_high = (df['close'] == price_high) & (df['close'] > df['close'].shift(20))
        macd_hist_decreasing = (df['macd_hist'] < df['macd_hist'].shift(1)) & (df['macd_hist'] > 0)
        df['top_divergence'] = (price_new_high & macd_hist_decreasing).astype(int)

        # Bottom divergence: price makes new low but MACD histogram increases
        price_new_low = (df['close'] == df['close'].rolling(window=20).min()) & (df['close'] < df['close'].shift(20))
        macd_hist_increasing = (df['macd_hist'] > df['macd_hist'].shift(1)) & (df['macd_hist'] < 0)
        df['bottom_divergence'] = (price_new_low & macd_hist_increasing).astype(int)

        # Divergence strength: how much MACD histogram diverges from price
        price_change = df['close'].pct_change(20)
        macd_hist_change = df['macd_hist'].pct_change(20)
        df['divergence_strength'] = np.abs(macd_hist_change - price_change) / (np.abs(price_change) + 1e-8)

        # ========== Volume Pattern features (量能形态) ==========
        # Volume breakout flag: volume > 2x average
        df['volume_breakout_flag'] = (df['volume_ratio'] > 2.0).astype(int)

        # Bottom volume flag: volume > 3x average AND price at low position
        df['bottom_volume_flag'] = ((df['volume_ratio'] > 3.0) & (df['price_position'] < 0.2)).astype(int)

        # Shrink pullback flag: volume < 0.7x average AND price near MA
        df['shrink_pullback_flag'] = ((df['volume_ratio'] < 0.7) & (df['ma_20_ratio'].between(0.95, 1.05))).astype(int)

        # Volume increasing: 3 consecutive days of volume increase
        vol_increase = df['volume_change'] > 0
        df['volume_increasing'] = (vol_increase & vol_increase.shift(1) & vol_increase.shift(2)).astype(int)

        # ========== Candlestick Pattern features (K线形态) ==========
        # Candle body and shadows
        candle_range = df['high'] - df['low']
        df['body_ratio'] = np.abs(df['close'] - df['open']) / (candle_range + 1e-8)
        df['upper_shadow_ratio'] = (df['high'] - np.maximum(df['open'], df['close'])) / (candle_range + 1e-8)
        df['lower_shadow_ratio'] = (np.minimum(df['open'], df['close']) - df['low']) / (candle_range + 1e-8)

        # Is bullish candle
        df['is_bullish'] = (df['close'] > df['open']).astype(int)

        # Close position within day's range
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)

        # Long lower shadow: lower shadow > 60% of range
        df['long_lower_shadow'] = (df['lower_shadow_ratio'] > 0.6).astype(int)

        # ========== One Yang Three Yin pattern (一阳三阴形态) ==========
        # Pattern: first day big bullish, next 3 days shrinking volume, last day breakout
        if len(df) >= 5:
            # First day big bullish (>2% body)
            d1_bullish = df['is_bullish'].shift(4) == 1
            d1_body = df['body_ratio'].shift(4) > 0.02
            # Next 3 days volume shrinking (< 0.8x previous day)
            d2_vol = df['volume'].shift(3) / (df['volume'].shift(4) + 1e-8)
            d3_vol = df['volume'].shift(2) / (df['volume'].shift(3) + 1e-8)
            d4_vol = df['volume'].shift(1) / (df['volume'].shift(2) + 1e-8)
            d2_shrink = d2_vol < 0.8
            d3_shrink = d3_vol < 0.8
            d4_shrink = d4_vol < 0.8
            # Last day bullish breakout
            d5_bullish = df['is_bullish'] == 1
            d5_breakout = df['close'].shift(4) < df['close']  # above first day close
            # Support hold: days 2-4 lowest not below day 1 open
            d2_low = df['low'].shift(3) > df['open'].shift(4)
            d3_low = df['low'].shift(2) > df['open'].shift(4)
            d4_low = df['low'].shift(1) > df['open'].shift(4)

            df['oyty_bullish_body'] = (d1_bullish & d1_body).astype(int)
            df['oyty_shrink_volume'] = (d2_shrink & d3_shrink & d4_shrink).astype(int)
            df['oyty_support_hold'] = (d2_low & d3_low & d4_low).astype(int)
            df['oyty_breakout'] = (d5_bullish & d5_breakout).astype(int)
            df['oyty_pattern'] = (df['oyty_bullish_body'] & df['oyty_shrink_volume'] &
                                   df['oyty_support_hold'] & df['oyty_breakout']).astype(int)
        else:
            df['oyty_bullish_body'] = 0
            df['oyty_shrink_volume'] = 0
            df['oyty_support_hold'] = 0
            df['oyty_breakout'] = 0
            df['oyty_pattern'] = 0

        # ========== Bottom volume surge (跌幅后放量) ==========
        # Price drop > 15% then volume surge > 3x
        price_drop = df['close'].pct_change(10) < -0.15
        vol_surge = df['volume_ratio'] > 3.0
        df['bottom_volume_surge'] = (price_drop & vol_surge).astype(int)

        # Price stabilize: bullish close holding recent low
        df['price_stabilize'] = (df['is_bullish'] == 1) & (df['close'] > df['low_20d']).astype(int)

        # ========== Box/Range features (箱体特征) ==========
        # Rolling box: 20-day high/low as proxy for box top/bottom
        df['box_top'] = df['close'].rolling(window=20).max()
        df['box_bottom'] = df['close'].rolling(window=20).min()
        df['box_width_pct'] = (df['box_top'] - df['box_bottom']) / df['box_bottom'] * 100

        # Box touch counts
        touch_top = (df['high'] >= df['box_top'] * 0.99)  # near box top
        touch_bottom = (df['low'] <= df['box_bottom'] * 1.01)  # near box bottom
        df['box_touch_top_count'] = touch_top.rolling(window=20).sum()
        df['box_touch_bottom_count'] = touch_bottom.rolling(window=20).sum()

        # ========== Support/Resistance features (支撑阻力) ==========
        # Distance to support/resistance
        df['distance_to_support'] = (df['close'] - df['box_bottom']) / df['box_bottom'] * 100
        df['distance_to_resistance'] = (df['box_top'] - df['close']) / df['box_top'] * 100

        # Near box boundaries
        df['near_box_bottom'] = (df['distance_to_support'] <= 5).astype(int)
        df['near_box_top'] = (df['distance_to_resistance'] <= 5).astype(int)

        # In box middle (middle 1/3)
        box_middle = (df['close'] > df['box_bottom'] + df['box_width_pct'] / 3) & \
                     (df['close'] < df['box_top'] - df['box_width_pct'] / 3)
        df['in_box_middle'] = box_middle.astype(int)

        # ========== Breakout features (突破信号) ==========
        df['breakout_up'] = (df['close'] > df['box_top']).astype(int)
        df['breakout_down'] = (df['close'] < df['box_bottom']).astype(int)
        df['breakout_volume_confirm'] = ((df['breakout_up'] | df['breakout_down']) & (df['volume_ratio'] > 2.0)).astype(int)

        # ========== MA Convergence features (均线收敛) ==========
        # Calculate MA convergence as variance of MA ratios
        ma_ratios = df[['ma_5_ratio', 'ma_10_ratio', 'ma_20_ratio']].std(axis=1)
        df['ma_convergence'] = 1 - ma_ratios  # Lower std = more convergence

        # ATR shrinking: ATR in lower quartile
        atr_pct = df['atr'] / df['close'] * 100
        df['atr_shrinking'] = (atr_pct < atr_pct.rolling(60).quantile(0.2)).astype(int)

        # Low volatility flag: volatility in lower 20% of history
        vol_level = df['volatility_20d'].rank(pct=True)
        df['low_volatility_flag'] = (vol_level < 0.2).astype(int)

        # ========== Composite Signals (综合信号) ==========
        # Trend score: based on MA arrangement and RSI
        ma_trend = df['ma_bullish_arrange'] * 2 - df['ma_bearish_arrange']
        rsi_trend = (df['rsi'] - 50) / 50  # normalize to -1 to 1
        df['trend_score'] = (ma_trend + rsi_trend).clip(-1, 1)

        # Signal features
        df['signal_buy'] = ((df['golden_cross_5_10'] == 1) | (df['bottom_divergence'] == 1) |
                            (df['breakout_up'] == 1)).astype(int)
        df['signal_sell'] = ((df['death_cross_5_10'] == 1) | (df['top_divergence'] == 1) |
                             (df['breakout_down'] == 1)).astype(int)
        df['signal_hold'] = (df['in_box_middle'] == 1).astype(int)

        # Risk level based on volatility
        vol_pct = df['volatility_20d'].rank(pct=True)
        df['risk_level'] = pd.cut(vol_pct, bins=[0, 0.33, 0.66, 1.0],
                                   labels=[0, 1, 2]).astype(float)  # 0=low, 1=medium, 2=high

        # ========== VWAP (Volume Weighted Average Price) ==========
        # VWAP = cumulative(price * volume) / cumulative(volume)
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        df['price_to_vwap'] = df['close'] / df['vwap']

        # ========== Aroon Indicator (阿隆指标) ==========
        # Aroon Up = (period - periods since highest) / period * 100
        # Aroon Down = (period - periods since lowest) / period * 100
        aroon_period = 25
        df['aroon_high'] = df['high'].rolling(window=aroon_period + 1).apply(lambda x: float(np.argmax(x)), raw=True)
        df['aroon_low'] = df['low'].rolling(window=aroon_period + 1).apply(lambda x: float(np.argmin(x)), raw=True)
        df['aroon_up'] = (aroon_period - df['aroon_high']) / aroon_period * 100
        df['aroon_down'] = (aroon_period - df['aroon_low']) / aroon_period * 100
        df['aroon_oscillator'] = df['aroon_up'] - df['aroon_down']
        df['aroon_trend'] = (df['aroon_up'] > df['aroon_down']).astype(int)

        # ========== Accumulation/Distribution (A/D Line) ==========
        # A/D = previous A/D + money flow multiplier * volume
        mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
        mf_volume = mf_multiplier * df['volume']
        df['ad_line'] = mf_volume.cumsum()
        df['ad_oscillator'] = df['ad_line'] - df['ad_line'].rolling(5).mean()

        # ========== Rate of Change (ROC) ==========
        # ROC = (close - close n periods ago) / close n periods ago * 100
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period) * 100

        # ========== DMI (Directional Movement Index) ==========
        # Plus DI and Minus DI
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

        # Smoothed DM
        smooth_plus_dm = plus_dm.rolling(window=14).mean()
        smooth_minus_dm = minus_dm.rolling(window=14).mean()

        # DI
        atr_14 = df['atr']
        plus_di = 100 * smooth_plus_dm / (atr_14 + 1e-8)
        minus_di = 100 * smooth_minus_dm / (atr_14 + 1e-8)

        df['dmi_plus_di'] = plus_di
        df['dmi_minus_di'] = minus_di
        df['dmi_di_diff'] = plus_di - minus_di
        df['dmi_adx'] = df['adx']  # ADX already calculated

        # ========== Lag Features (滞后特征) ==========
        # Lag returns
        for lag in [1, 2, 3, 5]:
            df[f'return_lag_{lag}'] = df['returns'].shift(lag)
            df[f'volume_change_lag_{lag}'] = df['volume_change'].shift(lag)

        # Lag RSI
        for lag in [1, 2, 3]:
            df[f'rsi_lag_{lag}'] = df['rsi'].shift(lag)

        # Lag MACD
        for lag in [1, 2]:
            df[f'macd_lag_{lag}'] = df['macd'].shift(lag)
            df[f'macd_hist_lag_{lag}'] = df['macd_hist'].shift(lag)

        # ========== Rolling Features (滚动特征) ==========
        # Rolling returns std (volatility persistence)
        df['returns_std_5'] = df['returns'].rolling(5).std()
        df['returns_std_10'] = df['returns'].rolling(10).std()

        # Rolling skewness and kurtosis
        df['returns_skew_10'] = df['returns'].rolling(10).skew()
        df['returns_skew_20'] = df['returns'].rolling(20).skew()

        # Rolling max drawdown
        rolling_max = df['close'].expanding().max()
        drawdown = (df['close'] - rolling_max) / rolling_max
        df['expanding_drawdown'] = drawdown

        # ========== Cross-asset Features (跨资产特征) ==========
        # Price relative to VWAP
        df['price_vs_vwap'] = (df['close'] > df['vwap']).astype(int)

        # Aroon + DMI combination signal
        df['aroon_dmi_bullish'] = ((df['aroon_up'] > df['aroon_down']) & (df['dmi_plus_di'] > df['dmi_minus_di'])).astype(int)
        df['aroon_dmi_bearish'] = ((df['aroon_down'] > df['aroon_up']) & (df['dmi_minus_di'] > df['dmi_plus_di'])).astype(int)

        # ========== Fill NaN values ==========
        # Fill NaN values with forward fill then backward fill for remaining
        df = df.ffill().bfill()

        # Defragment DataFrame to avoid performance warnings
        df = df.copy()

        return df
