"""Technical indicator features."""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from ..data_providers import fetch_stock_data
from ..utils.cache import FeatureCache
from ..utils.config import get_config
from .base import BaseFeatureExtractor

logger = logging.getLogger(__name__)


class TechnicalFeatures(BaseFeatureExtractor):
    """Extract technical indicator features."""

    def __init__(self, cache: Optional[FeatureCache] = None):
        """Initialize TechnicalFeatures with configuration."""
        super().__init__(cache)
        self.config = get_config().get_section("features").get("technical", {})
        self.ma_periods = self.config.get("ma_periods", [5, 10, 20, 60])
        self.rsi_period = self.config.get("rsi_period", 14)
        self.macd_fast = self.config.get("macd_fast", 12)
        self.macd_slow = self.config.get("macd_slow", 26)
        self.macd_signal = self.config.get("macd_signal", 9)
        self.bollinger_period = self.config.get("bollinger_period", 20)
        self.bollinger_std = self.config.get("bollinger_std", 2)

    @property
    def feature_type(self) -> str:
        """Return feature type name."""
        return "technical"

    def _fetch_with_incremental(self, stock_code: str, days: int) -> pd.DataFrame:
        """Fetch stock data with incremental update support.

        Only fetches missing data since last cache date to reduce API calls.
        """
        if not self.cache:
            return fetch_stock_data(stock_code, days=days)

        # Check if we have cached raw data
        cached_raw = self.cache.get(stock_code, "raw_price_data")
        if cached_raw is None or cached_raw.empty:
            # No cache, fetch full data
            data = fetch_stock_data(stock_code, days=days)
            if not data.empty and self.cache:
                self.cache.set(stock_code, "raw_price_data", data)
            return data

        # Get latest date in cache
        if "date" not in cached_raw.columns:
            return fetch_stock_data(stock_code, days=days)

        cached_raw["date"] = pd.to_datetime(cached_raw["date"])
        cached_raw = cached_raw.sort_values("date").reset_index(drop=True)
        latest_date = cached_raw["date"].max()
        today = pd.Timestamp.today().normalize()

        # Check if we need new data (cache is from a previous trading day)
        needs_new_data = latest_date.date() < today.date()

        # Check if we need more historical data
        cached_count = len(cached_raw)
        needs_more_history = cached_count < days

        if not needs_new_data and not needs_more_history:
            # Cache is up to date and has enough data
            return cached_raw.tail(days).reset_index(drop=True)

        if needs_new_data:
            # Fetch only the missing days since last cache date
            # Add extra days to account for weekends/holidays
            missing_days = (today - latest_date).days + 5
            fetch_days = max(missing_days, 10)  # At least 10 days

            logger.info(
                f"Incremental fetch: latest={latest_date.date()}, fetching {fetch_days} days"
            )

            new_data = fetch_stock_data(stock_code, days=fetch_days)
            if new_data.empty:
                # Failed to fetch new data, return cached data
                return cached_raw.tail(days).reset_index(drop=True)

            # Merge new data with cached data
            new_data["date"] = pd.to_datetime(new_data["date"])
            combined = pd.concat([cached_raw, new_data], ignore_index=True)
            combined = combined.drop_duplicates(subset=["date"], keep="last")
            combined = combined.sort_values("date").reset_index(drop=True)

            # Update cache
            self.cache.set(stock_code, "raw_price_data", combined)

            return combined.tail(days).reset_index(drop=True)

        elif needs_more_history:
            # Need more historical data, fetch additional days
            extra_days = days - cached_count + 10  # Add buffer
            logger.info(
                f"Fetching more history: cached={cached_count}, need={days}, fetching {extra_days} days"
            )

            # Fetch from an earlier start date
            total_fetch_days = days + 20  # Add buffer
            full_data = fetch_stock_data(stock_code, days=total_fetch_days)

            if full_data.empty:
                return cached_raw.tail(days).reset_index(drop=True)

            # Merge with cached data
            full_data["date"] = pd.to_datetime(full_data["date"])
            combined = pd.concat([cached_raw, full_data], ignore_index=True)
            combined = combined.drop_duplicates(subset=["date"], keep="last")
            combined = combined.sort_values("date").reset_index(drop=True)

            # Update cache
            self.cache.set(stock_code, "raw_price_data", combined)

            return combined.tail(days).reset_index(drop=True)

        return cached_raw.tail(days).reset_index(drop=True)

    def extract(self, stock_code: str, **kwargs) -> pd.DataFrame:
        """Extract technical features for a stock."""
        days = kwargs.get("days", 120)
        preloaded_data = kwargs.get("_preloaded_data")

        if preloaded_data is not None and not preloaded_data.empty:
            data = preloaded_data
        else:
            data = self._fetch_with_incremental(stock_code, days)
            if data.empty:
                return pd.DataFrame()

        data.columns = [c.lower() for c in data.columns]
        df = self._prepare_base_df(data, stock_code)

        self._add_returns(df)
        self._add_moving_averages(df)
        self._add_ma_cross_features(df)
        self._add_ma_arrangement_features(df)
        self._add_rsi(df)
        self._add_macd(df)
        self._add_bollinger_bands(df)
        self._add_atr(df)
        self._add_volume_indicators(df)
        self._add_momentum_features(df)
        self._add_volatility_features(df)
        self._add_high_low_features(df)
        self._add_short_term_momentum(df)
        self._add_risk_adjusted_features(df)
        self._add_adx(df)
        self._add_stochastic(df)
        self._add_mfi(df)
        self._add_cci(df)
        self._add_streak_features(df)
        self._add_pattern_features(df)
        self._add_ma120(df)
        self._add_ma_slope(df)
        self._add_deviation_features(df)
        self._add_macd_cross_features(df)
        self._add_macd_divergence(df)
        self._add_volume_patterns(df)
        self._add_candlestick_patterns(df)
        self._add_oyty_pattern(df)
        self._add_bottom_volume_surge(df)
        self._add_box_features(df)
        self._add_support_resistance(df)
        self._add_breakout_features(df)
        self._add_ma_convergence(df)
        self._add_composite_signals(df)
        self._add_vwap(df)
        self._add_aroon(df)
        self._add_accumulation_distribution(df)
        self._add_roc(df)
        self._add_dmi(df)
        self._add_lag_features(df)
        self._add_rolling_features(df)
        self._add_cross_asset_features(df)
        self._add_interaction_features(df)
        df = self._defragment_dataframe(df)
        self._clean_and_fill(df)

        return df

    def _defragment_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Defragment DataFrame by copying all columns at once.

        After many individual column insertions, pandas DataFrames become
        fragmented (block manager has many small blocks). This method
        consolidates all columns into a single contiguous block, improving
        memory layout and downstream performance.
        """
        return df.copy(deep=False)

    def _prepare_base_df(self, data: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """Prepare base DataFrame with price columns."""
        df = pd.DataFrame()
        df["date"] = pd.to_datetime(data["date"])
        df["stock_code"] = stock_code
        for col_name in ["close", "open", "high", "low", "volume"]:
            if col_name in data.columns:
                val = data[col_name]
                df[col_name] = val.iloc[:, 0] if isinstance(val, pd.DataFrame) else val
            else:
                raise KeyError(
                    f"Column '{col_name}' not found. Available: {list(data.columns)}"
                )
        return df

    def _add_returns(self, df: pd.DataFrame) -> None:
        """Add return features."""
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

    def _add_moving_averages(self, df: pd.DataFrame) -> None:
        """Add moving averages and EMA."""
        for period in self.ma_periods:
            df[f"ma_{period}"] = df["close"].rolling(window=period).mean()
            df[f"ma_{period}_ratio"] = df["close"] / df[f"ma_{period}"]
        for period in [12, 26]:
            df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()

    def _add_ma_cross_features(self, df: pd.DataFrame) -> None:
        """Add golden/death cross features."""
        for short, long in [(5, 10), (10, 20), (5, 20)]:
            diff = df[f"ma_{short}"] - df[f"ma_{long}"]
            prev_above = diff.shift(1) > 0
            now_above = diff > 0
            df[f"golden_cross_{short}_{long}"] = ((~prev_above) & now_above).astype(int)
            df[f"death_cross_{short}_{long}"] = (prev_above & (~now_above)).astype(int)

    def _add_ma_arrangement_features(self, df: pd.DataFrame) -> None:
        """Add MA arrangement (bullish/bearish) features."""
        df["ma_bullish_arrange"] = (
            (df["ma_5"] > df["ma_10"]) & (df["ma_10"] > df["ma_20"])
        ).astype(int)
        df["ma_bearish_arrange"] = (
            (df["ma_5"] < df["ma_10"]) & (df["ma_10"] < df["ma_20"])
        ).astype(int)
        df["ma_5_above_10"] = (df["ma_5"] > df["ma_10"]).astype(int)
        df["ma_10_above_20"] = (df["ma_10"] > df["ma_20"]).astype(int)
        df["ma_5_above_20"] = (df["ma_5"] > df["ma_20"]).astype(int)

    def _add_rsi(self, df: pd.DataFrame) -> None:
        """Add RSI indicator."""
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

    def _add_macd(self, df: pd.DataFrame) -> None:
        """Add MACD indicator."""
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=self.macd_signal, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

    def _add_bollinger_bands(self, df: pd.DataFrame) -> None:
        """Add Bollinger Bands."""
        df["bb_middle"] = df["close"].rolling(window=self.bollinger_period).mean()
        bb_std = df["close"].rolling(window=self.bollinger_period).std()
        df["bb_upper"] = df["bb_middle"] + (bb_std * self.bollinger_std)
        df["bb_lower"] = df["bb_middle"] - (bb_std * self.bollinger_std)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (
            df["bb_upper"] - df["bb_lower"]
        )

    def _add_atr(self, df: pd.DataFrame) -> None:
        """Add ATR indicator."""
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = true_range.rolling(window=14).mean()

    def _add_volume_indicators(self, df: pd.DataFrame) -> None:
        """Add volume indicators."""
        df["volume_ma5"] = df["volume"].rolling(window=5).mean()
        df["volume_ma20"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma20"]
        df["volume_change"] = df["volume"].pct_change()

    def _add_momentum_features(self, df: pd.DataFrame) -> None:
        """Add momentum features."""
        for period in [5, 10, 20]:
            df[f"momentum_{period}"] = df["close"] / df["close"].shift(period) - 1

    def _add_volatility_features(self, df: pd.DataFrame) -> None:
        """Add volatility features."""
        for period in [5, 10, 20]:
            df[f"volatility_{period}"] = df["returns"].rolling(window=period).std()

    def _add_high_low_features(self, df: pd.DataFrame) -> None:
        """Add high/low based features."""
        df["high_20d"] = df["close"].rolling(window=20).max()
        df["low_20d"] = df["close"].rolling(window=20).min()
        df["high_low_ratio"] = df["close"] / df["high_20d"]
        df["close_low_ratio"] = df["close"] / df["low_20d"]
        df["price_position"] = (df["close"] - df["low_20d"]) / (
            df["high_20d"] - df["low_20d"] + 1e-10
        )

    def _add_short_term_momentum(self, df: pd.DataFrame) -> None:
        """Add short-term momentum features."""
        df["return_2d"] = df["close"].pct_change(2)
        df["return_3d"] = df["close"].pct_change(3)
        df["momentum_acceleration"] = df["momentum_5"] - df["momentum_10"]

    def _add_risk_adjusted_features(self, df: pd.DataFrame) -> None:
        """Add risk-adjusted features."""
        df["volatility_20d"] = df["returns"].rolling(window=20).std() * np.sqrt(252)
        df["sharpe_like"] = df["momentum_5"] / (df["volatility_20d"] + 1e-8)
        df["cv"] = df["returns"].rolling(window=20).std() / (
            df["returns"].rolling(window=20).mean().abs() + 1e-8
        )
        rolling_max = df["close"].rolling(window=20).max()
        drawdown = (df["close"] - rolling_max) / rolling_max
        df["max_drawdown_20d"] = drawdown.rolling(window=20).min()

    def _add_adx(self, df: pd.DataFrame) -> None:
        """Add ADX indicator (stores intermediate DI for reuse by _add_dmi)."""
        high_diff = df["high"].diff()
        low_diff = -df["low"].diff()
        plus_dm = (
            high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
            .rolling(window=14)
            .mean()
        )
        minus_dm = (
            low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
            .rolling(window=14)
            .mean()
        )
        atr_14 = df["atr"]
        plus_di = 100 * plus_dm / (atr_14 + 1e-8)
        minus_di = 100 * minus_dm / (atr_14 + 1e-8)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        df["adx"] = dx
        df["adx_plus_di"] = plus_di
        df["adx_minus_di"] = minus_di

    def _add_stochastic(self, df: pd.DataFrame) -> None:
        """Add Stochastic Oscillator."""
        lowest_low = df["low"].rolling(window=14).min()
        highest_high = df["high"].rolling(window=14).max()
        df["stoch_k"] = (
            100 * (df["close"] - lowest_low) / (highest_high - lowest_low + 1e-8)
        )
        df["stoch_d"] = df["stoch_k"].rolling(window=3).mean()

    def _add_mfi(self, df: pd.DataFrame) -> None:
        """Add Money Flow Index."""
        typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
        money_flow = typical_price * df["volume"]
        positive_flow = (
            money_flow.where(typical_price > typical_price.shift(1), 0)
            .rolling(window=14)
            .sum()
        )
        negative_flow = (
            money_flow.where(typical_price < typical_price.shift(1), 0)
            .rolling(window=14)
            .sum()
        )
        df["mfi"] = 100 - (100 / (1 + positive_flow / (negative_flow + 1e-8)))

    def _add_cci(self, df: pd.DataFrame) -> None:
        """Add Commodity Channel Index."""
        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        sma_tp = tp.rolling(window=14).mean()
        mad = (tp - sma_tp).abs().rolling(window=14).mean()
        df["cci"] = (tp - sma_tp) / (0.015 * mad + 1e-8)

    def _add_streak_features(self, df: pd.DataFrame) -> None:
        """Add consecutive up/down day features."""
        daily_dir = np.sign(df["close"].diff())
        daily_dir = pd.Series(daily_dir).replace(0, np.nan).ffill().fillna(0)
        for window in [2, 3]:
            df[f"streak_up_{window}"] = (
                (daily_dir.shift(1) >= 1).rolling(window=window).sum() >= window
            ).astype(int)
            df[f"streak_down_{window}"] = (
                (daily_dir.shift(1) <= -1).rolling(window=window).sum() >= window
            ).astype(int)

    def _add_pattern_features(self, df: pd.DataFrame) -> None:
        """Add price-volume pattern features using historical data only."""
        ret_5d = df["close"].pct_change(5)
        ret_10d = df["close"].pct_change(10)

        # Pattern: significant drop then stabilization (potential reversal signal)
        # Uses past 10-day drop and recent 5-day stabilization
        df["pattern_drop_then_stabilize"] = (
            (ret_10d < -0.05) & (ret_5d.abs() < 0.02)
        ).astype(int)

        # Pattern: significant rise then pullback (potential continuation or reversal)
        df["pattern_rise_then_pullback"] = (
            (ret_10d > 0.05) & (ret_5d < -0.02)
        ).astype(int)

        # Volume surge patterns (using historical volume comparison)
        vol_10d_avg = df["volume"].rolling(window=10).mean()
        df["pattern_vol_surge"] = (
            (df["volume"] / vol_10d_avg > 2.0)
        ).astype(int)

        # Volume surge with direction
        df["pattern_vol_surge_rise"] = (
            (df["pattern_vol_surge"] == 1) & (ret_5d > 0)
        ).astype(int)
        df["pattern_vol_surge_drop"] = (
            (df["pattern_vol_surge"] == 1) & (ret_5d < 0)
        ).astype(int)

        # Reversal patterns using historical data
        ret_3d = df["close"].pct_change(3)
        # Prior trend (7 days ago to 3 days ago) vs recent trend (last 3 days)
        prior_ret = (df["close"].shift(3) / df["close"].shift(10) - 1)
        df["pattern_reversal_up"] = ((prior_ret < -0.03) & (ret_3d > 0.03)).astype(int)
        df["pattern_reversal_down"] = ((prior_ret > 0.03) & (ret_3d < -0.03)).astype(int)

    def _add_ma120(self, df: pd.DataFrame) -> None:
        """Add MA120 if not already present."""
        if 120 not in self.ma_periods:
            df["ma_120"] = df["close"].rolling(window=120).mean()
            df["ma_120_ratio"] = df["close"] / df["ma_120"]

    def _add_ma_slope(self, df: pd.DataFrame) -> None:
        """Add MA slope features."""
        df["ma_slope_20"] = df["ma_20"].pct_change(5) / 5

    def _add_deviation_features(self, df: pd.DataFrame) -> None:
        """Add deviation (乖离率) features."""
        for period in [5, 10, 20]:
            df[f"deviation_ma{period}_abs"] = np.abs(
                (df["close"] - df[f"ma_{period}"]) / df[f"ma_{period}"] * 100
            )

    def _add_macd_cross_features(self, df: pd.DataFrame) -> None:
        """Add MACD cross features."""
        macd_diff = df["macd"] - df["macd_signal"]
        macd_diff_prev = macd_diff.shift(1)
        df["macd_cross_up"] = ((macd_diff_prev <= 0) & (macd_diff > 0)).astype(int)
        df["macd_cross_above_zero"] = (
            (df["macd"] > 0) & (macd_diff_prev <= 0) & (macd_diff > 0)
        ).astype(int)
        df["macd_position"] = (df["macd"] > 0).astype(int)

    def _add_macd_divergence(self, df: pd.DataFrame) -> None:
        """Add MACD divergence features."""
        price_high = df["close"].rolling(window=20).max()
        price_new_high = (df["close"] == price_high) & (
            df["close"] > df["close"].shift(20)
        )
        macd_hist_decreasing = (df["macd_hist"] < df["macd_hist"].shift(1)) & (
            df["macd_hist"] > 0
        )
        df["top_divergence"] = (price_new_high & macd_hist_decreasing).astype(int)

        price_new_low = (df["close"] == df["close"].rolling(window=20).min()) & (
            df["close"] < df["close"].shift(20)
        )
        macd_hist_increasing = (df["macd_hist"] > df["macd_hist"].shift(1)) & (
            df["macd_hist"] < 0
        )
        df["bottom_divergence"] = (price_new_low & macd_hist_increasing).astype(int)

        price_change = df["close"].pct_change(20)
        macd_hist_change = df["macd_hist"].pct_change(20)
        df["divergence_strength"] = np.abs(macd_hist_change - price_change) / (
            np.abs(price_change) + 1e-8
        )

    def _add_volume_patterns(self, df: pd.DataFrame) -> None:
        """Add volume pattern features."""
        df["volume_breakout_flag"] = (df["volume_ratio"] > 2.0).astype(int)
        df["bottom_volume_flag"] = (
            (df["volume_ratio"] > 3.0) & (df["price_position"] < 0.2)
        ).astype(int)
        df["shrink_pullback_flag"] = (
            (df["volume_ratio"] < 0.7) & (df["ma_20_ratio"].between(0.95, 1.05))
        ).astype(int)
        vol_increase = df["volume_change"] > 0
        df["volume_increasing"] = (
            vol_increase & vol_increase.shift(1) & vol_increase.shift(2)
        ).astype(int)

    def _add_candlestick_patterns(self, df: pd.DataFrame) -> None:
        """Add candlestick pattern features."""
        candle_range = df["high"] - df["low"]
        df["body_ratio"] = np.abs(df["close"] - df["open"]) / (candle_range + 1e-8)
        df["upper_shadow_ratio"] = (
            df["high"] - np.maximum(df["open"], df["close"])
        ) / (candle_range + 1e-8)
        df["lower_shadow_ratio"] = (np.minimum(df["open"], df["close"]) - df["low"]) / (
            candle_range + 1e-8
        )
        df["is_bullish"] = (df["close"] > df["open"]).astype(int)
        df["close_position"] = (df["close"] - df["low"]) / (
            df["high"] - df["low"] + 1e-8
        )
        df["long_lower_shadow"] = (df["lower_shadow_ratio"] > 0.6).astype(int)

    def _add_oyty_pattern(self, df: pd.DataFrame) -> None:
        """Add One Yang Three Yin pattern features."""
        if len(df) < 5:
            for col in [
                "oyty_bullish_body",
                "oyty_shrink_volume",
                "oyty_support_hold",
                "oyty_breakout",
                "oyty_pattern",
            ]:
                df[col] = 0
            return

        d1_bullish = df["is_bullish"].shift(4) == 1
        d1_body = df["body_ratio"].shift(4) > 0.02
        d2_vol = df["volume"].shift(3) / (df["volume"].shift(4) + 1e-8)
        d3_vol = df["volume"].shift(2) / (df["volume"].shift(3) + 1e-8)
        d4_vol = df["volume"].shift(1) / (df["volume"].shift(2) + 1e-8)
        d5_bullish = df["is_bullish"] == 1
        d5_breakout = df["close"].shift(4) < df["close"]
        d2_low = df["low"].shift(3) > df["open"].shift(4)
        d3_low = df["low"].shift(2) > df["open"].shift(4)
        d4_low = df["low"].shift(1) > df["open"].shift(4)

        df["oyty_bullish_body"] = (d1_bullish & d1_body).astype(int)
        df["oyty_shrink_volume"] = (
            (d2_vol < 0.8) & (d3_vol < 0.8) & (d4_vol < 0.8)
        ).astype(int)
        df["oyty_support_hold"] = (d2_low & d3_low & d4_low).astype(int)
        df["oyty_breakout"] = (d5_bullish & d5_breakout).astype(int)
        df["oyty_pattern"] = (
            df["oyty_bullish_body"]
            & df["oyty_shrink_volume"]
            & df["oyty_support_hold"]
            & df["oyty_breakout"]
        ).astype(int)

    def _add_bottom_volume_surge(self, df: pd.DataFrame) -> None:
        """Add bottom volume surge features."""
        price_drop = df["close"].pct_change(10) < -0.15
        vol_surge = df["volume_ratio"] > 3.0
        df["bottom_volume_surge"] = (price_drop & vol_surge).astype(int)
        df["price_stabilize"] = (
            (df["is_bullish"] == 1) & (df["close"] > df["low_20d"])
        ).astype(int)

    def _add_box_features(self, df: pd.DataFrame) -> None:
        """Add box/range features."""
        df["box_top"] = df["close"].rolling(window=20).max()
        df["box_bottom"] = df["close"].rolling(window=20).min()
        df["box_width_pct"] = (
            (df["box_top"] - df["box_bottom"]) / df["box_bottom"] * 100
        )
        touch_top = df["high"] >= df["box_top"] * 0.99
        touch_bottom = df["low"] <= df["box_bottom"] * 1.01
        df["box_touch_top_count"] = touch_top.rolling(window=20).sum()
        df["box_touch_bottom_count"] = touch_bottom.rolling(window=20).sum()

    def _add_support_resistance(self, df: pd.DataFrame) -> None:
        """Add support/resistance features."""
        df["distance_to_support"] = (
            (df["close"] - df["box_bottom"]) / df["box_bottom"] * 100
        )
        df["distance_to_resistance"] = (
            (df["box_top"] - df["close"]) / df["box_top"] * 100
        )
        df["near_box_bottom"] = (df["distance_to_support"] <= 5).astype(int)
        df["near_box_top"] = (df["distance_to_resistance"] <= 5).astype(int)
        box_middle = (df["close"] > df["box_bottom"] + df["box_width_pct"] / 3) & (
            df["close"] < df["box_top"] - df["box_width_pct"] / 3
        )
        df["in_box_middle"] = box_middle.astype(int)

    def _add_breakout_features(self, df: pd.DataFrame) -> None:
        """Add breakout features."""
        df["breakout_up"] = (df["close"] > df["box_top"]).astype(int)
        df["breakout_down"] = (df["close"] < df["box_bottom"]).astype(int)
        df["breakout_volume_confirm"] = (
            (df["breakout_up"] | df["breakout_down"]) & (df["volume_ratio"] > 2.0)
        ).astype(int)

    def _add_ma_convergence(self, df: pd.DataFrame) -> None:
        """Add MA convergence features."""
        ma_ratios = df[["ma_5_ratio", "ma_10_ratio", "ma_20_ratio"]].std(axis=1)
        df["ma_convergence"] = 1 - ma_ratios
        atr_pct = df["atr"] / df["close"] * 100
        df["atr_shrinking"] = (atr_pct < atr_pct.rolling(60).quantile(0.2)).astype(int)
        vol_level = df["volatility_20d"].rank(pct=True)
        df["low_volatility_flag"] = (vol_level < 0.2).astype(int)

    def _add_composite_signals(self, df: pd.DataFrame) -> None:
        """Add composite signal features."""
        ma_trend = df["ma_bullish_arrange"] * 2 - df["ma_bearish_arrange"]
        rsi_trend = (df["rsi"] - 50) / 50
        df["trend_score"] = (ma_trend + rsi_trend).clip(-1, 1)
        df["signal_buy"] = (
            (df["golden_cross_5_10"] == 1)
            | (df["bottom_divergence"] == 1)
            | (df["breakout_up"] == 1)
        ).astype(int)
        df["signal_sell"] = (
            (df["death_cross_5_10"] == 1)
            | (df["top_divergence"] == 1)
            | (df["breakout_down"] == 1)
        ).astype(int)
        df["signal_hold"] = (df["in_box_middle"] == 1).astype(int)
        vol_pct = df["volatility_20d"].rank(pct=True)
        df["risk_level"] = pd.cut(
            vol_pct, bins=[0, 0.33, 0.66, 1.0], labels=[0, 1, 2]
        ).astype(float)

    def _add_vwap(self, df: pd.DataFrame) -> None:
        """Add VWAP features."""
        df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
        df["price_to_vwap"] = df["close"] / df["vwap"]

    def _add_aroon(self, df: pd.DataFrame) -> None:
        """Add Aroon indicator (vectorized)."""
        aroon_period = 25
        rolling_high = df["high"].rolling(window=aroon_period + 1)
        rolling_low = df["low"].rolling(window=aroon_period + 1)
        aroon_up = rolling_high.apply(lambda x: float(np.argmax(x)), raw=True)
        aroon_down = rolling_low.apply(lambda x: float(np.argmin(x)), raw=True)
        days_since_high = aroon_period - aroon_up
        days_since_low = aroon_period - aroon_down
        df["aroon_up"] = (aroon_period - days_since_high) / aroon_period * 100
        df["aroon_down"] = (aroon_period - days_since_low) / aroon_period * 100
        df["aroon_oscillator"] = df["aroon_up"] - df["aroon_down"]
        df["aroon_trend"] = (df["aroon_up"] > df["aroon_down"]).astype(int)

    def _add_accumulation_distribution(self, df: pd.DataFrame) -> None:
        """Add A/D line features."""
        mf_multiplier = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (
            df["high"] - df["low"] + 1e-10
        )
        mf_volume = mf_multiplier * df["volume"]
        df["ad_line"] = mf_volume.cumsum()
        df["ad_oscillator"] = df["ad_line"] - df["ad_line"].rolling(5).mean()

    def _add_roc(self, df: pd.DataFrame) -> None:
        """Add Rate of Change features."""
        for period in [5, 10, 20]:
            df[f"roc_{period}"] = (
                (df["close"] - df["close"].shift(period))
                / df["close"].shift(period)
                * 100
            )

    def _add_dmi(self, df: pd.DataFrame) -> None:
        """Add DMI features (consolidated with ADX)."""
        # Reuse ADX calculation already present; only add DMI-specific columns
        plus_di = df["adx_plus_di"] if "adx_plus_di" in df.columns else None
        minus_di = df["adx_minus_di"] if "adx_minus_di" in df.columns else None

        if plus_di is None or minus_di is None:
            high_diff = df["high"].diff()
            low_diff = -df["low"].diff()
            plus_dm_val = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
            minus_dm_val = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
            smooth_plus_dm = plus_dm_val.rolling(window=14).mean()
            smooth_minus_dm = minus_dm_val.rolling(window=14).mean()
            atr_14 = df["atr"]
            plus_di = 100 * smooth_plus_dm / (atr_14 + 1e-8)
            minus_di = 100 * smooth_minus_dm / (atr_14 + 1e-8)

        df["dmi_plus_di"] = plus_di
        df["dmi_minus_di"] = minus_di
        df["dmi_di_diff"] = plus_di - minus_di
        df["dmi_adx"] = df["adx"]

    def _add_lag_features(self, df: pd.DataFrame) -> None:
        """Add lag features."""
        for lag in [1, 2, 3, 5]:
            df[f"return_lag_{lag}"] = df["returns"].shift(lag)
            df[f"volume_change_lag_{lag}"] = df["volume_change"].shift(lag)
        for lag in [1, 2, 3]:
            df[f"rsi_lag_{lag}"] = df["rsi"].shift(lag)
        for lag in [1, 2]:
            df[f"macd_lag_{lag}"] = df["macd"].shift(lag)
            df[f"macd_hist_lag_{lag}"] = df["macd_hist"].shift(lag)

    def _add_rolling_features(self, df: pd.DataFrame) -> None:
        """Add rolling statistical features."""
        df["returns_std_5"] = df["returns"].rolling(5).std()
        df["returns_std_10"] = df["returns"].rolling(10).std()
        df["returns_skew_10"] = df["returns"].rolling(10).skew()
        df["returns_skew_20"] = df["returns"].rolling(20).skew()
        rolling_max = df["close"].expanding().max()
        drawdown = (df["close"] - rolling_max) / rolling_max
        df["expanding_drawdown"] = drawdown

    def _add_cross_asset_features(self, df: pd.DataFrame) -> None:
        """Add cross-asset combination features."""
        df["price_vs_vwap"] = (df["close"] > df["vwap"]).astype(int)
        df["aroon_dmi_bullish"] = (
            (df["aroon_up"] > df["aroon_down"])
            & (df["dmi_plus_di"] > df["dmi_minus_di"])
        ).astype(int)
        df["aroon_dmi_bearish"] = (
            (df["aroon_down"] > df["aroon_up"])
            & (df["dmi_minus_di"] > df["dmi_plus_di"])
        ).astype(int)

    def _add_interaction_features(self, df: pd.DataFrame) -> None:
        """Add interaction features that capture nonlinear relationships."""
        new_cols = {}

        if "rsi" in df.columns and "volume_ratio" in df.columns:
            new_cols["rsi_volume_interaction"] = df["rsi"] * df["volume_ratio"]

        if "momentum_10" in df.columns and "volatility_20d" in df.columns:
            new_cols["momentum_volatility_interaction"] = df["momentum_10"] * df["volatility_20d"]

        if "macd_hist" in df.columns and "volume_change" in df.columns:
            new_cols["macd_volume_interaction"] = df["macd_hist"] * df["volume_change"]

        if "rsi" in df.columns and "returns" in df.columns:
            new_cols["rsi_return_divergence"] = df["rsi"] * df["returns"]

        if "bb_width" in df.columns and "rsi" in df.columns:
            new_cols["bb_rsi_interaction"] = df["bb_width"] * df["rsi"]

        if "adx" in df.columns and "volume_ratio" in df.columns:
            new_cols["adx_volume_interaction"] = df["adx"] * df["volume_ratio"]

        if "bb_pct" in df.columns and "momentum_5" in df.columns:
            new_cols["bb_momentum_interaction"] = df["bb_pct"] * df["momentum_5"]

        if "stoch_k" in df.columns and "volume_ratio" in df.columns:
            new_cols["stoch_volume_interaction"] = df["stoch_k"] * df["volume_ratio"]

        if "mfi" in df.columns and "atr_ratio" in df.columns:
            new_cols["mfi_atr_interaction"] = df["mfi"] * df["atr_ratio"]

        if "cci" in df.columns and "volume_change" in df.columns:
            new_cols["cci_volume_interaction"] = df["cci"] * df["volume_change"]

        if new_cols:
            for name, values in new_cols.items():
                df[name] = values

    def _clean_and_fill(self, df: pd.DataFrame) -> None:
        """Fill NaN values and handle outliers with improved strategy."""
        df.ffill(inplace=True)
        df.bfill(inplace=True)

        skip_cols = {
            "date",
            "stock_code",
            "close",
            "open",
            "high",
            "low",
            "volume",
            "golden_cross_5_10",
            "death_cross_5_10",
            "golden_cross_10_20",
            "death_cross_10_20",
            "golden_cross_5_20",
            "death_cross_5_20",
            "ma_bullish_arrange",
            "ma_bearish_arrange",
            "signal_buy",
            "signal_sell",
            "signal_hold",
            "is_bullish",
            "is_bearish",
        }

        numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in skip_cols
        ]

        if numeric_cols:
            clipped = df[numeric_cols].clip(
                lower=df[numeric_cols].quantile(0.01),
                upper=df[numeric_cols].quantile(0.99),
                axis=1,
            )
            df[numeric_cols] = clipped

        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        df.ffill(inplace=True)
        df.bfill(inplace=True)

        remaining_na = df[numeric_cols].isna()
        if remaining_na.any().any():
            medians = df[numeric_cols].median()
            df[numeric_cols] = df[numeric_cols].fillna(medians)
            still_na = df[numeric_cols].isna()
            if still_na.any().any():
                df[numeric_cols] = df[numeric_cols].fillna(0)
