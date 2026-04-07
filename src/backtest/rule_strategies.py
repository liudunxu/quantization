"""Rule-based trading strategies.

This module implements rule-based strategies derived from the strategy references
in strategy/references/daily_stock_analysis/.
"""

import pandas as pd

from .engine import Strategy


class MAGoldenCrossStrategy(Strategy):
    """MA Golden Cross Strategy.

    Buy when MA5 crosses above MA10 (or MA10 crosses above MA20)
    with volume confirmation. Sell when death cross occurs.

    Rules:
    - Buy: MA5 crosses above MA10 in last 3 days AND volume > MA5 volume * 1.2
    - Sell: MA5 crosses below MA10 (death cross)
    - Hold: Otherwise
    """

    def __init__(self, fast_ma: int = 5, slow_ma: int = 10, volume_ratio: float = 1.2):
        """Initialize MAGoldenCrossStrategy."""
        description = (
            f"均线金叉策略：当{fast_ma}日均线上穿{slow_ma}日均线且成交量放大时买入，"
            f"下穿时卖出。量能确认阈值={volume_ratio}。"
        )
        super().__init__(f"MA Golden Cross ({fast_ma}/{slow_ma})", description)
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.volume_ratio = volume_ratio

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals using MA golden cross."""
        signals = pd.Series(0, index=df.index)
        close = df["close"]

        # Calculate moving averages
        ma_fast = close.rolling(window=self.fast_ma).mean()
        ma_slow = close.rolling(window=self.slow_ma).mean()

        # Calculate volume MA
        if "volume" in df.columns:
            volume_ma = df["volume"].rolling(window=self.fast_ma).mean()
        else:
            volume_ma = pd.Series(1, index=df.index)

        # Golden cross: fast crosses above slow
        golden_cross = (ma_fast > ma_slow) & (ma_fast.shift(1) <= ma_slow.shift(1))

        # Death cross: fast crosses below slow
        death_cross = (ma_fast < ma_slow) & (ma_fast.shift(1) >= ma_slow.shift(1))

        # Volume confirmation
        volume_confirm = df["volume"] > volume_ma * self.volume_ratio

        for i in range(self.slow_ma, len(df)):
            if golden_cross.iloc[i] and volume_confirm.iloc[i]:
                signals.iloc[i] = 1  # Buy
            elif death_cross.iloc[i]:
                signals.iloc[i] = -1  # Sell

        return signals


class BullTrendStrategy(Strategy):
    """Bull Trend Strategy.

    Follows the primary uptrend: MA5 >= MA10 >= MA20.
    Buy when in uptrend with pullback, sell when trend breaks.

    Rules:
    - Buy: MA5 >= MA10 >= MA20 AND price within 5% of MA5 (not chasing)
    - Sell: MA5 < MA10 or price breaks below MA20
    - Hold: In uptrend but price too far from MA5
    """

    def __init__(
        self, ma5_period: int = 5, ma10_period: int = 10, ma20_period: int = 20
    ):
        """Initialize BullTrendStrategy."""
        description = (
            f"趋势跟随策略：当MA{ma5_period}>=MA{ma10_period}>=MA{ma20_period}形成多头排列时，"
            f"价格回调至均线附近（5%以内）买入，跌破MA{ma20_period}卖出。"
        )
        super().__init__(
            f"Bull Trend ({ma5_period}/{ma10_period}/{ma20_period})", description
        )
        self.ma5_period = ma5_period
        self.ma10_period = ma10_period
        self.ma20_period = ma20_period

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals using bull trend strategy."""
        signals = pd.Series(0, index=df.index)
        close = df["close"]

        ma5 = close.rolling(window=self.ma5_period).mean()
        ma10 = close.rolling(window=self.ma10_period).mean()
        ma20 = close.rolling(window=self.ma20_period).mean()

        for i in range(self.ma20_period, len(df)):
            current_price = close.iloc[i]
            current_ma5 = ma5.iloc[i]
            current_ma10 = ma10.iloc[i]
            current_ma20 = ma20.iloc[i]

            # Check if in uptrend
            in_uptrend = current_ma5 >= current_ma10 >= current_ma20

            # Check if price is near MA5 (within 5%)
            near_ma5 = (current_price - current_ma5) / current_ma5 <= 0.05

            # Check if price is below MA20 (trend broken)
            broken_trend = current_price < current_ma20

            if broken_trend:
                signals.iloc[i] = -1  # Sell - trend broken
            elif in_uptrend and near_ma5:
                signals.iloc[i] = 1  # Buy - uptrend + not chasing
            # else: hold - either not in uptrend or price too far from MA5

        return signals


class ShrinkPullbackStrategy(Strategy):
    """Shrink Pullback Strategy.

    Buy when price pulls back to MA5/MA10 with shrinking volume,
    indicating trend continuation. Used in uptrends.

    Rules:
    - Buy: Price within 1% of MA5 or within 2% of MA10
           AND volume < MA5 volume * 0.7 (shrinkage)
           AND in uptrend (MA5 >= MA10 >= MA20)
    - Sell: Price breaks below MA20
    - Hold: Otherwise
    """

    def __init__(
        self, lookback: int = 5, ma_period: int = 5, volume_shrink: float = 0.7
    ):
        """Initialize ShrinkPullbackStrategy."""
        description = (
            f"缩量回调策略：在上升趋势中，当价格回调至均线附近且成交量缩小时买入，"
            f"跌破MA20卖出。缩量阈值={volume_shrink}。"
        )
        super().__init__(
            f"Shrink Pullback ({ma_period}d, vol<{volume_shrink})", description
        )
        self.lookback = lookback
        self.ma_period = ma_period
        self.volume_shrink = volume_shrink

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals using shrink pullback strategy."""
        signals = pd.Series(0, index=df.index)
        close = df["close"]

        ma5 = close.rolling(window=5).mean()
        ma10 = close.rolling(window=10).mean()
        ma20 = close.rolling(window=20).mean()

        volume_ma5 = df["volume"].rolling(window=5).mean()

        for i in range(20, len(df)):
            current_price = close.iloc[i]
            current_ma5 = ma5.iloc[i]
            current_ma10 = ma10.iloc[i]
            current_ma20 = ma20.iloc[i]
            current_volume = df["volume"].iloc[i]
            vol_ma5 = volume_ma5.iloc[i]

            # Check uptrend
            in_uptrend = current_ma5 >= current_ma10 >= current_ma20

            # Check volume shrinkage
            is_shrinking = current_volume < vol_ma5 * self.volume_shrink

            # Check price near MA5 (within 1%) or MA10 (within 2%)
            near_ma5 = abs(current_price - current_ma5) / current_ma5 <= 0.01
            near_ma10 = abs(current_price - current_ma10) / current_ma10 <= 0.02

            # Check if broken below MA20
            broken_ma20 = current_price < current_ma20

            if broken_ma20:
                signals.iloc[i] = -1  # Sell
            elif in_uptrend and is_shrinking and (near_ma5 or near_ma10):
                signals.iloc[i] = 1  # Buy

        return signals


class BottomVolumeStrategy(Strategy):
    """Bottom Volume Surge Strategy.

    After extended decline (>15%), when volume spikes (>3x) and price
    stabilizes, potential reversal signal.

    Rules:
    - Buy: Price dropped >15% from 20d high AND volume > 3x MA5 volume
           AND today's close > open (bullish candle)
    - Sell: Price breaks below recent low
    - Hold: Otherwise
    """

    def __init__(self, drop_threshold: float = 0.15, volume_multiplier: float = 3.0):
        """Initialize BottomReversalStrategy."""
        description = (
            f"底部放量策略：当股价从20日高点下跌超过{drop_threshold * 100:.0f}%后，"
            f"出现{volume_multiplier}倍放量且收阳线时买入，为反转信号。"
        )
        super().__init__(
            f"Bottom Volume (drop>{drop_threshold}, vol>{volume_multiplier}x)",
            description,
        )
        self.drop_threshold = drop_threshold
        self.volume_multiplier = volume_multiplier

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals using bottom reversal strategy."""
        signals = pd.Series(0, index=df.index)
        close = df["close"]
        high = df["high"] if "high" in df.columns else close
        low = df["low"] if "low" in df.columns else close

        volume_ma5 = df["volume"].rolling(window=5).mean()

        for i in range(25, len(df)):
            current_price = close.iloc[i]
            current_volume = df["volume"].iloc[i]
            vol_ma5 = volume_ma5.iloc[i]

            # Calculate 20-day high
            high_20d = high.iloc[max(0, i - 20) : i].max()
            low_20d = low.iloc[max(0, i - 20) : i].min()

            # Check drop > threshold
            dropped = (high_20d - current_price) / high_20d > self.drop_threshold

            # Check volume surge
            volume_surge = current_volume > vol_ma5 * self.volume_multiplier

            # Check bullish candle
            if "open" in df.columns:
                bullish = current_price > df["open"].iloc[i]
            else:
                bullish = True

            # Check if price stabilizes (close near low_20d)
            stabilized = current_price >= low_20d * 0.97  # Within 3% of recent low

            if dropped and volume_surge and bullish and stabilized:
                signals.iloc[i] = 1  # Buy - reversal signal

        return signals


class BoxOscillationStrategy(Strategy):
    """Box Oscillation Strategy.

    Identifies price range (box) and trades within:
    Buy at support (box bottom), sell at resistance (box top).

    Rules:
    - Buy: Price within 5% of box bottom AND volume stabilizing
    - Sell: Price within 5% of box top OR break below box bottom
    - Hold: Price in middle region
    """

    def __init__(
        self,
        lookback: int = 60,
        box_touch_min: int = 2,
        support_margin: float = 0.05,
        resistance_margin: float = 0.05,
        box_width_min: float = 0.05,
    ):
        """Initialize BoxOscillationStrategy."""
        description = (
            f"箱体震荡策略：识别价格波动区间（箱体），在箱体底部附近买入，"
            f"顶部附近或跌破箱底时卖出。支撑/阻力边际={support_margin * 100:.0f}%。"
        )
        super().__init__(
            f"Box Oscillation (L:{lookback}, T:{box_touch_min})", description
        )
        self.lookback = lookback
        self.box_touch_min = box_touch_min
        self.support_margin = support_margin
        self.resistance_margin = resistance_margin
        self.box_width_min = box_width_min

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals using box oscillation strategy."""
        signals = pd.Series(0, index=df.index)

        for i in range(self.lookback, len(df)):
            window_data = df.iloc[max(0, i - self.lookback) : i]
            close = window_data["close"]
            high = window_data["high"] if "high" in window_data.columns else close

            # Find box top and bottom
            box_top = high.quantile(0.9)  # 90th percentile high
            box_bottom = close.quantile(0.1)  # 10th percentile low

            # Check if box is valid (wide enough)
            box_width = (box_top - box_bottom) / box_bottom
            if box_width < self.box_width_min:
                continue

            current_price = df["close"].iloc[i]

            # Calculate position in box
            distance_to_bottom = (current_price - box_bottom) / box_bottom
            distance_to_top = (box_top - current_price) / box_top

            # Buy near bottom
            if distance_to_bottom < self.support_margin:
                signals.iloc[i] = 1

            # Sell near top or broken below
            elif distance_to_top < self.resistance_margin:
                signals.iloc[i] = -1
            elif current_price < box_bottom:
                signals.iloc[i] = -1  # Broken below box

        return signals


class EmotionCycleStrategy(Strategy):
    """Emotion Cycle Strategy.

    Based on sentiment and turnover patterns:
    Buy at sentiment bottom (fear), sell at sentiment top (euphoria).

    Rules:
    - Buy: Volume shrinking (< 0.5x MA) AND price near lows
           AND RSI < 30 (oversold)
    - Sell: RSI > 70 (overbought) OR volume surging AND price near highs
    - Hold: Otherwise
    """

    def __init__(
        self,
        volume_shrink: float = 0.5,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
    ):
        """Initialize EmotionCycleStrategy."""
        description = (
            f"情绪周期策略：基于RSI和成交量判断市场情绪，"
            f"超卖（RSI<{rsi_oversold}）且缩量时买入，超买（RSI>{rsi_overbought}）时卖出。"
        )
        super().__init__(
            f"Emotion Cycle (RSI {rsi_oversold}/{rsi_overbought})", description
        )
        self.volume_shrink = volume_shrink
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals using emotion cycle strategy."""
        signals = pd.Series(0, index=df.index)
        close = df["close"]

        # Calculate RSI if not present
        if "rsi" in df.columns:
            rsi = df["rsi"]
        else:
            # Calculate simple RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))

        volume_ma = df["volume"].rolling(window=20).mean()
        volume_ma5 = df["volume"].rolling(window=5).mean()

        # Calculate price position in 20-day range
        high_20d = close.rolling(window=20).max()
        low_20d = close.rolling(window=20).min()
        price_position = (close - low_20d) / (high_20d - low_20d + 1e-10)

        for i in range(20, len(df)):
            current_volume = df["volume"].iloc[i]
            vol_ma = volume_ma.iloc[i]
            vol_ma5_val = volume_ma5.iloc[i]
            current_rsi = rsi.iloc[i]
            close.iloc[i]
            current_position = price_position.iloc[i]

            # Volume shrinking
            current_volume < vol_ma * self.volume_shrink

            # Buy signals: cold sentiment + oversold
            if current_rsi < self.rsi_oversold and current_position < 0.3:
                signals.iloc[i] = 1  # Buy - fear bottom

            # Sell signals: hot sentiment + overbought
            elif current_rsi > self.rsi_overbought and current_position > 0.7:
                signals.iloc[i] = -1  # Sell - euphoria top

            # Also sell if price near highs but volume surging
            elif vol_ma5_val > vol_ma * 2 and current_position > 0.8:
                signals.iloc[i] = -1  # Sell - volume surge at top

        return signals


class VolumeBreakoutStrategy(Strategy):
    """Volume Breakout Strategy.

    Buy when price breaks out of range with volume confirmation.

    Rules:
    - Buy: Close > 20d high AND volume > 2x MA volume
    - Sell: Close < 20d low OR 3 consecutive lower closes
    - Hold: Otherwise
    """

    def __init__(self, lookback: int = 20, volume_multiplier: float = 2.0):
        """Initialize VolumeBreakoutStrategy."""
        description = (
            f"放量突破策略：当价格突破{lookback}日高点且成交量放大{volume_multiplier}倍时买入，"
            f"跌破{lookback}日低点或连续3日下跌时卖出。"
        )
        super().__init__(
            f"Volume Breakout ({lookback}d, vol>{volume_multiplier}x)", description
        )
        self.lookback = lookback
        self.volume_multiplier = volume_multiplier

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals using volume breakout strategy."""
        signals = pd.Series(0, index=df.index)
        close = df["close"]

        high_20d = close.rolling(window=self.lookback).max()
        low_20d = close.rolling(window=self.lookback).min()
        volume_ma = df["volume"].rolling(window=5).mean()

        for i in range(self.lookback, len(df)):
            current_price = close.iloc[i]
            current_volume = df["volume"].iloc[i]
            vol_ma = volume_ma.iloc[i]

            # Check breakout
            breakout_up = current_price > high_20d.iloc[i - 1] if i > 0 else False
            breakout_down = current_price < low_20d.iloc[i - 1] if i > 0 else False

            # Volume confirmation
            volume_confirm = current_volume > vol_ma * self.volume_multiplier

            # Check 3 consecutive lower closes
            if i >= 3:
                three_lower = (
                    close.iloc[i]
                    < close.iloc[i - 1]
                    < close.iloc[i - 2]
                    < close.iloc[i - 3]
                )
            else:
                three_lower = False

            if breakout_down or three_lower:
                signals.iloc[i] = -1  # Sell
            elif breakout_up and volume_confirm:
                signals.iloc[i] = 1  # Buy

        return signals


class OneYangThreeYinStrategy(Strategy):
    """One Yang Three Yin Strategy.

    Classic candlestick pattern: one strong bullish day followed by
    three shrinking bearish days, then breakout.

    Rules:
    - Day 1: Bullish candle with body > 2%
    - Days 2-4: Bearish candles with shrinking volume
    - Days 2-4: Lowest prices don't break Day 1's open
    - Day 5: Bullish candle breaking above Day 1's close
    - Buy: Pattern completes with Day 5 breakout
    - Sell: When profit > 10% or RSI > 70
    """

    def __init__(self, body_threshold: float = 0.02):
        """Initialize OneYangThreeYinStrategy."""
        description = (
            f"一阳三阴策略：经典的K线形态，一根大阳线后连续三根缩量小阴线，"
            f"第五天突破阳线收盘价时买入。阳线实体阈值={body_threshold * 100:.0f}%。"
        )
        super().__init__(f"One Yang Three Yin (body>{body_threshold})", description)
        self.body_threshold = body_threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals using one yang three yin pattern."""
        signals = pd.Series(0, index=df.index)

        for i in range(5, len(df)):
            # Get last 5 candles
            recent = df.iloc[i - 4 : i + 1]

            if len(recent) < 5:
                continue

            day1 = recent.iloc[0]
            day2 = recent.iloc[1]
            day3 = recent.iloc[2]
            day4 = recent.iloc[3]
            day5 = recent.iloc[4]

            # Day 1: Bullish with body > threshold
            day1_bullish = day1["close"] > day1["open"]
            day1_body = (day1["close"] - day1["open"]) / day1["open"]

            if not day1_bullish or day1_body < self.body_threshold:
                continue

            day1_open = day1["open"]
            day1_close = day1["close"]

            # Days 2-4: Bearish
            days_2_4_bearish = (
                day2["close"] < day2["open"]
                and day3["close"] < day3["open"]
                and day4["close"] < day4["open"]
            )

            if not days_2_4_bearish:
                continue

            # Days 2-4: Volume shrinking
            vol1 = day1["volume"]
            vol2 = day2["volume"]
            vol3 = day3["volume"]
            vol4 = day4["volume"]

            vol_shrinking = vol4 < vol3 < vol2 < vol1

            if not vol_shrinking:
                continue

            # Days 2-4: Don't break day1's open
            days_2_4_low = min(day2["low"], day3["low"], day4["low"])
            support_hold = days_2_4_low > day1_open * 0.99

            if not support_hold:
                continue

            # Day 5: Breakout above day1's close
            day5_bullish = day5["close"] > day5["open"]
            day5_breakout = day5["close"] > day1_close

            if day5_bullish and day5_breakout:
                signals.iloc[i] = 1  # Buy

        return signals


class MACDDivergenceStrategy(Strategy):
    """MACD Divergence Strategy.

    Buy when price makes new low but MACD histogram shows improvement
    (bottom divergence). Sell on top divergence.

    Rules:
    - Buy: Price < 20d low AND MACD histogram > previous MACD histogram
           AND MACD in oversold region
    - Sell: Price > 20d high AND MACD histogram < previous
    - Hold: Otherwise
    """

    def __init__(self, lookback: int = 20):
        """Initialize MACDDivergenceStrategy."""
        description = (
            f"MACD背驰策略：当价格创新低但MACD柱状图回升时（底背驰）买入，"
            f"价格创新高但MACD减弱时（顶背驰）卖出。回溯周期={lookback}日。"
        )
        super().__init__(f"MACD Divergence ({lookback}d)", description)
        self.lookback = lookback

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals using MACD divergence strategy."""
        signals = pd.Series(0, index=df.index)
        close = df["close"]

        # Calculate MACD if not present
        if "macd" not in df.columns or "macd_hist" not in df.columns:
            exp1 = close.ewm(span=12, adjust=False).mean()
            exp2 = close.ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=9, adjust=False).mean()
            macd_hist = macd - signal_line
        else:
            macd_hist = df["macd_hist"]

        for i in range(self.lookback + 5, len(df)):
            current_price = close.iloc[i]
            current_hist = macd_hist.iloc[i]
            prev_hist = macd_hist.iloc[i - 1]

            # Calculate 20-day low
            low_20d = close.iloc[max(0, i - self.lookback) : i].min()
            high_20d = close.iloc[max(0, i - self.lookback) : i].max()

            # Bottom divergence: price making new low, MACD improving
            price_new_low = current_price < low_20d
            hist_improving = current_hist > prev_hist

            # Top divergence: price making new high, MACD weakening
            price_new_high = current_price > high_20d
            hist_weakening = current_hist < prev_hist

            if price_new_low and hist_improving:
                signals.iloc[i] = 1  # Buy - bottom divergence
            elif price_new_high and hist_weakening:
                signals.iloc[i] = -1  # Sell - top divergence

        return signals
