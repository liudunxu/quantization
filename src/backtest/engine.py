"""Backtesting engine for stock trading strategies."""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from ..utils.config import get_config


@dataclass
class Trade:
    """Represents a single trade."""

    date: pd.Timestamp
    action: str  # 'buy' or 'sell'
    price: float
    quantity: int
    commission: float = 0.0


@dataclass
class BacktestResult:
    """Results from a backtest."""

    strategy_name: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    buy_hold_return: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    daily_returns: pd.Series = field(default_factory=pd.Series)


class Strategy:
    """Base strategy class."""

    def __init__(self, name: str, description: str = ""):
        """Initialize Strategy."""
        self.name = name
        self.description = description

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals.

        Returns:
            Series with values 1 (buy), 0 (hold), -1 (sell)
        """
        raise NotImplementedError


class BuyAndHoldStrategy(Strategy):
    """Simple buy and hold strategy (benchmark, no trading costs)."""

    def __init__(self):
        """Initialize BuyAndHoldStrategy."""
        super().__init__(
            "Buy & Hold",
            "买入并持有策略：在第一天买入后长期持有，不做任何操作。作为基准策略用于比较。",
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate buy signal on first day, hold thereafter."""
        signals = pd.Series(0, index=df.index)
        signals.iloc[0] = 1  # Buy on first day
        return signals

    def calculate_return(self, df: pd.DataFrame) -> float:
        """Calculate pure price return without trading costs (as benchmark)."""
        if len(df) < 2:
            return 0.0
        start_price = df["close"].iloc[0]
        end_price = df["close"].iloc[-1]
        return (end_price - start_price) / start_price


class HighSellLowBuyStrategy(Strategy):
    """Contrarian strategy: sell when price is high, buy when low."""

    def __init__(self, lookback: int = 20, threshold: float = 0.1):
        """Initialize HighSellLowBuyStrategy."""
        super().__init__(
            f"High Sell Low Buy (L:{lookback}, T:{threshold})",
            f"高抛低吸策略：基于{lookback}日价格区间，当价格处于区间底部{threshold * 100:.0f}%时买入，处于顶部{threshold * 100:.0f}%时卖出。",
        )
        self.lookback = lookback
        self.threshold = threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate contrarian signals based on price range."""
        signals = pd.Series(0, index=df.index)
        close = df["close"]

        # Calculate rolling high/low
        rolling_high = close.rolling(window=self.lookback).max()
        rolling_low = close.rolling(window=self.lookback).min()

        # Position in range
        position = (close - rolling_low) / (rolling_high - rolling_low + 1e-10)

        # Buy when at bottom of range, sell when at top
        signals[position < (0.5 - self.threshold)] = 1
        signals[position > (0.5 + self.threshold)] = -1

        return signals


class MLStrategy(Strategy):
    """Machine learning based strategy with market regime filter."""

    def __init__(
        self,
        model,
        name: str = "ML Strategy",
        min_samples: int = 20,
        confidence_threshold: float = 0.0,
        bear_market_threshold: float = -0.01,
        require_bull_market_for_buy: bool = True,
    ):
        """Initialize ML strategy.

        Args:
            model: Trained model with predict method
            name: Strategy name
            min_samples: Minimum samples before generating signals
            confidence_threshold: Minimum confidence to trade (0-1)
            bear_market_threshold: Index return threshold below which market is considered bearish
            require_bull_market_for_buy: If True, only buy when market is bullish (or neutral)
        """
        description = (
            f"机器学习策略：使用CatBoost模型预测买卖信号，"
            f"置信度阈值={confidence_threshold:.0%}，"
            f"熊市阈值={bear_market_threshold:.1%}，"
            f"{'熊市禁止买入' if require_bull_market_for_buy else '允许熊市买入'}。"
        )
        super().__init__(name, description)
        self.model = model
        self.min_samples = min_samples
        self.confidence_threshold = confidence_threshold
        self.bear_market_threshold = bear_market_threshold
        self.require_bull_market_for_buy = require_bull_market_for_buy

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals using ML model."""
        signals = pd.Series(0, index=df.index)

        if len(df) < self.min_samples:
            return signals

        # Determine market regime based on index returns if available
        market_returns = None
        if "index_returns" in df.columns:
            market_returns = df["index_returns"].values

        for i in range(self.min_samples, len(df)):
            try:
                pred, confidence = self.model.predict(df.iloc[: i + 1])

                # Check market regime
                is_bear_market = False
                if market_returns is not None and i >= 1:
                    # Use 3-day average market return to determine regime (smoother)
                    lookback = min(3, i)
                    recent_returns = market_returns[i - lookback : i]
                    recent_market_return = (
                        np.mean(recent_returns) if len(recent_returns) > 0 else 0
                    )
                    if recent_market_return < self.bear_market_threshold:
                        is_bear_market = True

                # Only signal if confidence exceeds threshold
                if confidence >= self.confidence_threshold:
                    # In bear market, allow sell signals and high-confidence buys
                    if is_bear_market and self.require_bull_market_for_buy:
                        if pred == -1:  # Sell in bear market
                            signals.iloc[i] = pred
                        elif (
                            pred == 1 and confidence >= 0.7
                        ):  # High confidence buy allowed
                            signals.iloc[i] = pred
                        else:
                            signals.iloc[i] = 0  # Hold in bear market
                    else:
                        signals.iloc[i] = pred
                else:
                    signals.iloc[i] = 0
            except Exception:
                signals.iloc[i] = 0

        return signals


class HybridStrategy(Strategy):
    """Hybrid strategy combining ML and HighSellLowBuy.

    Only trades when ML and simple strategy agree, otherwise defaults to simple.
    This provides more robust signals by requiring confirmation.
    """

    def __init__(
        self,
        model,
        lookback: int = 20,
        threshold: float = 0.1,
        min_samples: int = 20,
        ml_confidence_threshold: float = 0.50,
        bear_market_threshold: float = -0.005,
        require_bull_market_for_buy: bool = True,
    ):
        """Initialize hybrid strategy.

        Args:
            model: Trained ML model
            lookback: Lookback period for HighSellLowBuy
            threshold: Threshold for HighSellLowBuy
            min_samples: Min samples before ML generates signals
            ml_confidence_threshold: Min confidence for ML signal
            bear_market_threshold: Market return threshold for bull/bear
            require_bull_market_for_buy: Only buy in bull market
        """
        description = (
            f"混合策略：结合机器学习和高抛低吸，"
            f"ML置信度阈值={ml_confidence_threshold:.0%}，"
            f"高抛低吸回溯={lookback}日，阈值={threshold:.0%}。"
            f"当两者信号一致时才交易。"
        )
        super().__init__("Hybrid Strategy (ML+HSSLB)", description)
        self.model = model
        self.min_samples = min_samples
        self.ml_confidence_threshold = ml_confidence_threshold
        self.bear_market_threshold = bear_market_threshold
        self.require_bull_market_for_buy = require_bull_market_for_buy
        # Simple strategy for fallback
        self.simple_strategy = HighSellLowBuyStrategy(
            lookback=lookback, threshold=threshold
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals using hybrid approach."""
        signals = pd.Series(0, index=df.index)

        if len(df) < self.min_samples:
            return signals

        # Get simple strategy signals
        simple_signals = self.simple_strategy.generate_signals(df)

        # Get market returns for regime detection
        market_returns = None
        if "index_returns" in df.columns:
            market_returns = df["index_returns"].values

        for i in range(self.min_samples, len(df)):
            try:
                # Get ML prediction
                pred, confidence = self.model.predict(df.iloc[: i + 1])
                ml_signal = pred

                # Check market regime
                is_bear_market = False
                if market_returns is not None and i >= 1:
                    # Use 3-day average market return to determine regime (smoother)
                    lookback = min(3, i)
                    recent_returns = market_returns[i - lookback : i]
                    recent_market_return = (
                        np.mean(recent_returns) if len(recent_returns) > 0 else 0
                    )
                    if recent_market_return < self.bear_market_threshold:
                        is_bear_market = True

                simple_signal = simple_signals.iloc[i]

                # Determine final signal
                if confidence >= self.ml_confidence_threshold:
                    # ML is confident
                    if ml_signal == simple_signal:
                        # They agree - use the signal
                        final_signal = ml_signal
                    else:
                        # They disagree - use simple strategy (more conservative)
                        final_signal = simple_signal

                    # Apply market filter for buys (allow high-confidence buys in bear market)
                    if is_bear_market and self.require_bull_market_for_buy:
                        if final_signal == 1 and confidence < 0.7:
                            final_signal = (
                                0  # No buy in bear market unless high confidence
                            )

                    signals.iloc[i] = final_signal
                else:
                    # ML not confident - use simple strategy
                    # But apply market filter
                    if is_bear_market and self.require_bull_market_for_buy:
                        if simple_signal == 1:
                            signals.iloc[i] = 0
                        else:
                            signals.iloc[i] = simple_signal
                    else:
                        signals.iloc[i] = simple_signal

            except Exception:
                # Fall back to simple on error
                signals.iloc[i] = simple_signals.iloc[i]

        return signals


class RollingMLStrategy(Strategy):
    """ML strategy with periodic retraining to adapt to market changes.

    Retrains the model every N days using a rolling window of training data.
    This helps the strategy adapt to changing market regimes.
    """

    def __init__(
        self,
        model_class,  # Model class (not instance)
        train_window: int = 180,  # Days of data to train on
        retrain_interval: int = 30,  # Days between retraining
        min_samples: int = 20,
        confidence_threshold: float = 0.50,
        bear_market_threshold: float = -0.005,
        require_bull_market_for_buy: bool = True,
        model_params: dict = None,  # Parameters for the model
    ):
        """Initialize rolling ML strategy.

        Args:
            model_class: CatBoost model class
            train_window: Number of days to use for training
            retrain_interval: Days between retraining
            min_samples: Min samples before generating signals
            confidence_threshold: Min confidence to trade
            bear_market_threshold: Market return threshold
            require_bull_market_for_buy: Only buy in bull market
            model_params: Dict of model parameters
        """
        description = (
            f"滚动机器学习策略：每{retrain_interval}天重新训练模型，"
            f"使用{train_window}天的滚动窗口数据，"
            f"置信度阈值={confidence_threshold:.0%}，"
            f"适应市场风格变化。"
        )
        super().__init__(f"Rolling ML ({train_window}/{retrain_interval})", description)
        self.model_class = model_class
        self.train_window = train_window
        self.retrain_interval = retrain_interval
        self.min_samples = min_samples
        self.confidence_threshold = confidence_threshold
        self.bear_market_threshold = bear_market_threshold
        self.require_bull_market_for_buy = require_bull_market_for_buy
        self.model_params = model_params or {}
        self.model = None
        self.last_train_idx = 0

    def _retrain_model(self, df: pd.DataFrame, end_idx: int) -> bool:
        """Retrain the model using data up to end_idx."""
        try:
            start_idx = max(0, end_idx - self.train_window)
            train_data = df.iloc[start_idx:end_idx].copy()

            if len(train_data) < self.min_samples + 10:
                return False

            # Create new model
            self.model = self.model_class()

            # Set parameters
            for key, value in self.model_params.items():
                setattr(self.model, key, value)

            # Train
            self.model.train(train_data, forward_days=5, threshold=0.01)
            self.last_train_idx = end_idx
            return True
        except Exception:
            return False

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals using rolling ML model."""
        signals = pd.Series(0, index=df.index)

        if len(df) < self.min_samples:
            return signals

        # Get market returns for regime detection
        market_returns = None
        if "index_returns" in df.columns:
            market_returns = df["index_returns"].values

        # Initial training
        self._retrain_model(df, self.min_samples)

        for i in range(self.min_samples, len(df)):
            # Check if we need to retrain
            if i - self.last_train_idx >= self.retrain_interval:
                self._retrain_model(df, i)

            try:
                if self.model is None:
                    signals.iloc[i] = 0
                    continue

                # Get prediction
                pred, confidence = self.model.predict(df.iloc[: i + 1])

                # Check market regime
                is_bear_market = False
                if market_returns is not None and i >= 1:
                    # Use 3-day average market return to determine regime (smoother)
                    lookback = min(3, i)
                    recent_returns = market_returns[i - lookback : i]
                    recent_market_return = (
                        np.mean(recent_returns) if len(recent_returns) > 0 else 0
                    )
                    if recent_market_return < self.bear_market_threshold:
                        is_bear_market = True

                if confidence >= self.confidence_threshold:
                    # Apply market filter for buys (allow high-confidence buys in bear market)
                    if is_bear_market and self.require_bull_market_for_buy:
                        if pred == 1 and confidence < 0.7:
                            signals.iloc[i] = 0
                        else:
                            signals.iloc[i] = pred
                    else:
                        signals.iloc[i] = pred
                else:
                    signals.iloc[i] = 0

            except Exception:
                signals.iloc[i] = 0

        return signals


class RollingHybridStrategy(Strategy):
    """Rolling ML strategy with hybrid confirmation.

    Combines rolling training with the hybrid approach:
    - Retrains model periodically to adapt to market changes
    - Uses HighSellLowBuy as fallback/confirmation
    - Only trades when ML and simple strategy agree
    - More trades than pure RollingML due to fallback behavior
    """

    def __init__(
        self,
        model_class,  # Model class (not instance)
        train_window: int = 180,  # Days of data to train on
        retrain_interval: int = 20,  # Days between retraining
        lookback: int = 10,  # Lookback for HighSellLowBuy
        threshold: float = 0.10,  # Threshold for HighSellLowBuy
        min_samples: int = 20,
        ml_confidence_threshold: float = 0.45,  # Lower than pure ML for more trades
        bear_market_threshold: float = -0.005,
        require_bull_market_for_buy: bool = True,
        model_params: dict = None,  # Parameters for the model
    ):
        """Initialize rolling hybrid strategy.

        Args:
            model_class: CatBoost model class
            train_window: Number of days to use for training
            retrain_interval: Days between retraining
            lookback: Lookback period for HighSellLowBuy
            threshold: Threshold for HighSellLowBuy
            min_samples: Min samples before generating signals
            ml_confidence_threshold: Min confidence for ML signal
            bear_market_threshold: Market return threshold
            require_bull_market_for_buy: Only buy in bull market
            model_params: Dict of model parameters
        """
        description = (
            f"滚动混合策略：每{retrain_interval}天重新训练模型，"
            f"结合高抛低吸策略，"
            f"ML置信度阈值={ml_confidence_threshold:.0%}，"
            f"高抛低吸回溯={lookback}日。"
        )
        super().__init__(
            f"Rolling Hybrid ({train_window}/{retrain_interval})", description
        )
        self.model_class = model_class
        self.train_window = train_window
        self.retrain_interval = retrain_interval
        self.lookback = lookback
        self.threshold = threshold
        self.min_samples = min_samples
        self.ml_confidence_threshold = ml_confidence_threshold
        self.bear_market_threshold = bear_market_threshold
        self.require_bull_market_for_buy = require_bull_market_for_buy
        self.model_params = model_params or {}
        self.model = None
        self.last_train_idx = 0
        # Simple strategy for fallback
        self.simple_strategy = HighSellLowBuyStrategy(
            lookback=lookback, threshold=threshold
        )

    def _retrain_model(self, df: pd.DataFrame, end_idx: int) -> bool:
        """Retrain the model using data up to end_idx."""
        try:
            start_idx = max(0, end_idx - self.train_window)
            train_data = df.iloc[start_idx:end_idx].copy()

            if len(train_data) < self.min_samples + 10:
                return False

            # Create new model
            self.model = self.model_class()

            # Set parameters
            for key, value in self.model_params.items():
                setattr(self.model, key, value)

            # Train
            self.model.train(train_data, forward_days=5, threshold=0.01)
            self.last_train_idx = end_idx
            return True
        except Exception:
            return False

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals using rolling hybrid approach."""
        signals = pd.Series(0, index=df.index)

        if len(df) < self.min_samples:
            return signals

        # Get simple strategy signals
        simple_signals = self.simple_strategy.generate_signals(df)

        # Get market returns for regime detection
        market_returns = None
        if "index_returns" in df.columns:
            market_returns = df["index_returns"].values

        # Initial training
        self._retrain_model(df, self.min_samples)

        for i in range(self.min_samples, len(df)):
            # Check if we need to retrain
            if i - self.last_train_idx >= self.retrain_interval:
                self._retrain_model(df, i)

            try:
                if self.model is None:
                    signals.iloc[i] = simple_signals.iloc[i]
                    continue

                # Get ML prediction
                ml_pred, ml_confidence = self.model.predict(df.iloc[: i + 1])
                simple_signal = simple_signals.iloc[i]

                # Check market regime
                is_bear_market = False
                if market_returns is not None and i >= 1:
                    # Use 3-day average market return to determine regime (smoother)
                    lookback = min(3, i)
                    recent_returns = market_returns[i - lookback : i]
                    recent_market_return = (
                        np.mean(recent_returns) if len(recent_returns) > 0 else 0
                    )
                    if recent_market_return < self.bear_market_threshold:
                        is_bear_market = True

                # Determine final signal using hybrid logic
                if ml_confidence >= self.ml_confidence_threshold:
                    # ML is confident
                    if ml_pred == simple_signal:
                        # They agree - use the signal
                        final_signal = ml_pred
                    else:
                        # They disagree - use simple strategy (more conservative)
                        final_signal = simple_signal

                    # Apply market filter for buys (allow high-confidence buys in bear market)
                    if is_bear_market and self.require_bull_market_for_buy:
                        if final_signal == 1 and ml_confidence < 0.7:
                            final_signal = (
                                0  # No buy in bear market unless high confidence
                            )

                    signals.iloc[i] = final_signal
                else:
                    # ML not confident - use simple strategy
                    # But apply market filter
                    if is_bear_market and self.require_bull_market_for_buy:
                        if simple_signal == 1:
                            signals.iloc[i] = 0
                        else:
                            signals.iloc[i] = simple_signal
                    else:
                        signals.iloc[i] = simple_signal

            except Exception:
                # Fall back to simple on error
                signals.iloc[i] = simple_signals.iloc[i]

        return signals


class BacktestEngine:
    """Backtesting engine with dynamic position sizing."""

    def __init__(
        self,
        initial_cash: float = 100000,
        commission: float = 0.001,
        slippage: float = 0.001,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
        max_lots_per_trade: int = 3,  # Max 3 lots (300 shares) per trade
        lot_size: int = 100,  # 1 lot = 100 shares
        max_drawdown_threshold: float = 0.20,  # Max 20% drawdown before stopping
    ):
        """Initialize BacktestEngine."""
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_lots_per_trade = max_lots_per_trade
        self.lot_size = lot_size
        self.max_shares_per_trade = max_lots_per_trade * lot_size
        self.max_drawdown_threshold = max_drawdown_threshold  # Risk control

    def _calculate_position_size(
        self,
        signal: int,
        cash: float,
        current_price: float,
        prev_price: float,
        position: int,
        entry_price: float,
        avg_cost: float,
        atr: float = None,
        shares: int = 0,
    ) -> int:
        """Calculate position size using professional volatility-based approach.

        This implements the industry-standard ATR-based position sizing used by
        top quant funds (Bridgewater, Renaissance, etc.):

        Position Size = Risk Amount / (ATR * Multiplier)

        Where Risk Amount = Account * Risk Percentage (typically 1-2%)

        Benefits:
        - Automatically adjusts for volatility (high vol = smaller position)
        - Ensures equal risk contribution per trade
        - Prevents outsized losses in volatile periods

        Args:
            signal: Trading signal (1=buy, 0=hold, -1=sell)
            cash: Available cash
            current_price: Current price
            prev_price: Previous day's close
            position: Current position (0 or 1)
            entry_price: Entry price for current position
            avg_cost: Average cost basis
            atr: Average True Range (volatility measure)

        Returns:
            Number of shares to trade
        """
        if signal == 0:
            return 0

        max_shares = self.max_shares_per_trade

        # Default ATR if not provided (assume 2% of price)
        if atr is None or atr <= 0:
            atr = current_price * 0.02

        # Risk parameters (optimized for smaller positions and more trades)
        risk_per_trade_pct = 0.01  # Risk 1% of capital per trade
        atr_multiplier = 1.2  # Stop loss at 1.2 * ATR

        # Calculate dollar risk amount
        account_value = cash + (position * avg_cost * 100)  # Rough estimate
        risk_amount = max(account_value * risk_per_trade_pct, 1000)  # Min $1000 risk

        # Maximum position based on ATR stop distance
        atr_stop_distance = atr * atr_multiplier
        if atr_stop_distance > 0:
            max_position_by_risk = int(risk_amount / atr_stop_distance)
        else:
            max_position_by_risk = max_shares

        # Maximum position based on cash
        max_position_by_cash = int(cash / (current_price * (1 + self.commission)))

        # Calculate price momentum
        if prev_price > 0:
            daily_change = (current_price - prev_price) / prev_price
        else:
            daily_change = 0

        if signal == 1:
            # Buy signal - support both new position and adding to existing position

            # Base position from risk model
            base_position = min(max_position_by_risk, max_position_by_cash)

            # Momentum adjustment (mean reversion factor)
            if daily_change < -0.02:  # Price dropped significantly
                # Strong buy signal - increase position (contrarian)
                momentum_multiplier = 1.3
            elif daily_change < 0:
                # Slight drop - normal buy
                momentum_multiplier = 1.0
            elif daily_change > 0.02:  # Price rose significantly
                # Weak signal - reduce position (avoid chasing)
                momentum_multiplier = 0.7
            else:
                # Normal - slight increase
                momentum_multiplier = 0.9

            target_shares = int(base_position * momentum_multiplier)

            # Ensure within limits
            target_shares = min(target_shares, max_shares, max_position_by_cash)

            # Minimum 1 lot (100 shares) if we have a signal
            # For US stocks, allow smaller minimum (50 shares)
            min_lot = (
                50 if current_price > 100 else 100
            )  # Smaller min for expensive stocks
            if target_shares >= min_lot:
                return target_shares
            elif max_position_by_cash >= min_lot:
                # Force minimum lot if we have enough cash
                return min_lot
            else:
                return 0

        elif signal == -1 and position == 1:
            # Sell signal - use P&L and momentum

            # Use actual shares passed from caller
            current_shares = (
                shares if shares > 0 else int(account_value * 0.3 / current_price)
            )

            # Calculate unrealized P&L
            if avg_cost > 0:
                unrealized_pnl = (current_price - avg_cost) / avg_cost
            else:
                unrealized_pnl = 0

            # Base sell from risk model
            base_sell = min(max_position_by_risk, current_shares)

            # P&L and momentum adjustment
            if unrealized_pnl > 0.08:  # Good profit (>8%)
                if daily_change > 0:
                    # Rising with profit - sell more (take profit)
                    multiplier = 1.5
                else:
                    multiplier = 1.2
            elif unrealized_pnl > 0.03:  # Small profit
                if daily_change > 0.01:
                    multiplier = 1.2
                else:
                    multiplier = 0.9
            elif unrealized_pnl < -0.05:  # Large loss
                if daily_change < -0.01:
                    # Falling further - hold more (average down)
                    multiplier = 0.3
                else:
                    multiplier = 0.6
            elif unrealized_pnl < -0.02:  # Small loss
                multiplier = 0.7
            else:
                multiplier = 1.0

            target_shares = int(base_sell * multiplier)

            # Ensure within limits
            target_shares = min(target_shares, current_shares, max_shares)

            # Minimum 1 lot
            if target_shares >= 100:
                return target_shares
            else:
                return 0

        return 0

    def run(
        self,
        df: pd.DataFrame,
        strategy: Strategy,
        benchmark: Optional[Strategy] = None,
        precomputed_signals: Optional[pd.Series] = None,
    ) -> BacktestResult:
        """Run backtest for a strategy.

        Args:
            df: DataFrame with price data
            strategy: Strategy to use
            benchmark: Optional benchmark strategy
            precomputed_signals: Optional pre-generated signals (for ML strategy with history)
        """
        if df.empty:
            raise ValueError("Empty DataFrame")

        # Use precomputed signals if provided, otherwise generate from strategy
        if precomputed_signals is not None:
            signals = precomputed_signals
        else:
            signals = strategy.generate_signals(df)
        prices = df["close"].values
        dates = df["date"].values if "date" in df.columns else df.index

        # Calculate ATR for volatility-based position sizing
        # ATR = Average of True Range over 14 periods (standard)
        if "atr" in df.columns:
            atr_values = df["atr"].values
        else:
            # Calculate True Range if ATR not available
            high = df["high"].values if "high" in df.columns else prices
            low = df["low"].values if "low" in df.columns else prices
            tr = np.maximum(
                high - low,
                np.maximum(
                    np.abs(high - np.roll(prices, 1)), np.abs(low - np.roll(prices, 1))
                ),
            )
            tr[0] = high[0] - low[0]
            atr_values = np.convolve(tr, np.ones(14) / 14, mode="valid")
            # Pad to match prices length
            if len(atr_values) < len(prices):
                atr_values = np.pad(
                    atr_values, (len(prices) - len(atr_values), 0), mode="edge"
                )

        # Initialize
        cash = self.initial_cash
        position = 0  # 0 = no stock, 1 = long
        shares = 0
        avg_cost = 0.0  # Average cost basis for position
        entry_price = 0.0  # Track entry price for stop loss
        trades = []
        equity = [self.initial_cash]
        current_drawdown = 0
        max_drawdown = 0
        peak_equity = self.initial_cash

        buy_hold_shares = self.initial_cash / prices[0]
        buy_hold_equity = [buy_hold_shares * p for p in prices]

        # Handle initial position at index 0
        # Calculate position size for consistent initial allocation
        first_price = prices[0] * (1 - self.slippage)
        first_atr = atr_values[0] if len(atr_values) > 0 else first_price * 0.02

        # Always use the same logic: calculate position based on risk model
        # This ensures consistent initial allocation across all strategies
        # Use only 50% of cash for initial position to leave room for adding
        target_shares = self._calculate_position_size(
            signal=1,
            cash=self.initial_cash * 0.5,  # Use 50% of cash for initial position
            current_price=first_price,
            prev_price=first_price,
            position=0,
            entry_price=first_price,
            avg_cost=first_price,
            atr=first_atr,
        )
        # Limit to affordable shares
        shares = min(target_shares, int(self.initial_cash * 0.5 / first_price))

        # For expensive stocks (>100), allow smaller minimum (50 shares)
        min_lot = 50 if first_price > 100 else 100
        if shares < min_lot and int(self.initial_cash / first_price) >= min_lot:
            shares = min_lot

        if position == 0 and shares >= min_lot:
            cost = shares * first_price
            comm = cost * self.commission
            cash = self.initial_cash - cost - comm
            position = 1
            entry_price = first_price
            avg_cost = first_price
            trades.append(
                Trade(
                    date=dates[0]
                    if hasattr(dates[0], "date")
                    else pd.Timestamp(dates[0]),
                    action="buy",
                    price=first_price,
                    quantity=shares,
                    commission=comm,
                )
            )

        for i in range(1, len(prices)):
            current_price = prices[i] * (
                1 + self.slippage if position == 1 else 1 - self.slippage
            )
            prices[i]

            # Calculate equity
            if position == 1:
                equity_value = cash + shares * current_price
            else:
                equity_value = cash

            equity.append(equity_value)

            # Track drawdown
            peak_equity = max(peak_equity, equity_value)
            current_drawdown = (peak_equity - equity_value) / peak_equity
            max_drawdown = max(max_drawdown, current_drawdown)

            # Graduated risk control: reduce position based on drawdown severity
            risk_control_triggered = False
            if self.max_drawdown_threshold > 0:
                # Graduated risk reduction levels
                if current_drawdown >= self.max_drawdown_threshold:
                    # Level 3: Close all positions (severe drawdown)
                    risk_control_triggered = True
                    if position == 1:
                        # Force sell all shares
                        proceeds = shares * current_price
                        comm = proceeds * self.commission
                        cash = cash + proceeds - comm
                        trades.append(
                            Trade(
                                date=dates[i]
                                if hasattr(dates[i], "date")
                                else pd.Timestamp(dates[i]),
                                action="sell",
                                price=current_price,
                                quantity=shares,
                                commission=comm,
                            )
                        )
                        shares = 0
                        position = 0
                elif current_drawdown >= self.max_drawdown_threshold * 0.75:
                    # Level 2: Reduce position by 50% (moderate drawdown)
                    if position == 1 and shares > 100:
                        shares_to_sell = shares // 2
                        if shares_to_sell >= 50:  # Minimum lot size
                            proceeds = shares_to_sell * current_price
                            comm = proceeds * self.commission
                            cash = cash + proceeds - comm
                            trades.append(
                                Trade(
                                    date=dates[i]
                                    if hasattr(dates[i], "date")
                                    else pd.Timestamp(dates[i]),
                                    action="sell",
                                    price=current_price,
                                    quantity=shares_to_sell,
                                    commission=comm,
                                )
                            )
                            shares -= shares_to_sell
                elif current_drawdown >= self.max_drawdown_threshold * 0.5:
                    # Level 1: Reduce position by 25% (mild drawdown)
                    if position == 1 and shares > 200:
                        shares_to_sell = shares // 4
                        if shares_to_sell >= 50:  # Minimum lot size
                            proceeds = shares_to_sell * current_price
                            comm = proceeds * self.commission
                            cash = cash + proceeds - comm
                            trades.append(
                                Trade(
                                    date=dates[i]
                                    if hasattr(dates[i], "date")
                                    else pd.Timestamp(dates[i]),
                                    action="sell",
                                    price=current_price,
                                    quantity=shares_to_sell,
                                    commission=comm,
                                )
                            )
                            shares -= shares_to_sell

            # Check stop loss and take profit first (these override signals)
            stop_loss_triggered = False
            take_profit_triggered = False

            if position == 1 and entry_price > 0:
                price_change = (current_price - entry_price) / entry_price

                if self.stop_loss > 0 and price_change <= -self.stop_loss:
                    stop_loss_triggered = True
                elif self.take_profit > 0 and price_change >= self.take_profit:
                    take_profit_triggered = True

            # Execute signals (only if not stopped out, took profit, or risk controlled)
            if (
                not stop_loss_triggered
                and not take_profit_triggered
                and not risk_control_triggered
            ):
                # Buy signal - support both new position and adding to existing position
                if signals.iloc[i] == 1:
                    prev_day_price = prices[i - 1]
                    current_atr = (
                        atr_values[i] if i < len(atr_values) else current_price * 0.02
                    )
                    target_shares = self._calculate_position_size(
                        signal=1,
                        cash=cash,
                        current_price=current_price,
                        prev_price=prev_day_price,
                        position=position,
                        entry_price=current_price if position == 0 else entry_price,
                        avg_cost=current_price if position == 0 else avg_cost,
                        atr=current_atr,
                    )
                    # Can't buy more than cash allows
                    max_affordable = int(cash / (current_price * (1 + self.commission)))
                    shares_to_buy = min(
                        target_shares, max_affordable, self.max_shares_per_trade
                    )

                    # Check if we can add to position (max position limit)
                    if position == 1:
                        # Adding to existing position - limit to max_lots_per_trade
                        max_additional = self.max_shares_per_trade
                        shares_to_buy = min(shares_to_buy, max_additional)

                    # Minimum lot size: 50 for expensive stocks, 100 otherwise
                    min_lot = 50 if current_price > 100 else 100
                    if shares_to_buy >= min_lot:  # At least min lot
                        cost = shares_to_buy * current_price
                        comm = cost * self.commission
                        cash = cash - cost - comm

                        # Update position
                        if position == 0:
                            # New position
                            shares = shares_to_buy
                            entry_price = current_price
                            avg_cost = current_price
                            position = 1
                        else:
                            # Adding to existing position
                            total_shares = shares + shares_to_buy
                            avg_cost = (
                                shares * avg_cost + shares_to_buy * current_price
                            ) / total_shares
                            shares = total_shares

                        trades.append(
                            Trade(
                                date=dates[i]
                                if hasattr(dates[i], "date")
                                else pd.Timestamp(dates[i]),
                                action="buy",
                                price=current_price,
                                quantity=shares_to_buy,
                                commission=comm,
                            )
                        )

                elif signals.iloc[i] == -1 and position == 1:
                    # Sell with dynamic sizing
                    prev_day_price = prices[i - 1]
                    current_atr = (
                        atr_values[i] if i < len(atr_values) else current_price * 0.02
                    )
                    target_shares = self._calculate_position_size(
                        signal=-1,
                        cash=cash,
                        current_price=current_price,
                        prev_price=prev_day_price,
                        position=position,
                        entry_price=entry_price,
                        avg_cost=avg_cost,
                        atr=current_atr,
                        shares=shares,
                    )
                    shares_to_sell = min(
                        target_shares, shares, self.max_shares_per_trade
                    )

                    # Force minimum 1 lot sell if we have a sell signal and enough shares
                    if shares_to_sell < 50 and shares >= 50:
                        shares_to_sell = min(50, shares)
                    elif shares_to_sell == 0 and shares > 0:
                        # If we have shares but calculated 0, sell all shares
                        shares_to_sell = shares

                    if shares_to_sell >= 50:  # At least 50 shares (reduced from 100)
                        proceeds = shares_to_sell * current_price
                        comm = proceeds * self.commission
                        cash = cash + proceeds - comm
                        shares -= shares_to_sell
                        trades.append(
                            Trade(
                                date=dates[i]
                                if hasattr(dates[i], "date")
                                else pd.Timestamp(dates[i]),
                                action="sell",
                                price=current_price,
                                quantity=shares_to_sell,
                                commission=comm,
                            )
                        )

                        if shares == 0:
                            position = 0
                            entry_price = 0.0
                            avg_cost = 0.0

            # Execute stop loss if triggered
            if stop_loss_triggered and position == 1:
                proceeds = shares * current_price
                comm = proceeds * self.commission
                cash = cash + proceeds - comm
                trades.append(
                    Trade(
                        date=dates[i]
                        if hasattr(dates[i], "date")
                        else pd.Timestamp(dates[i]),
                        action="sell",
                        price=current_price,
                        quantity=shares,
                        commission=comm,
                    )
                )
                position = 0
                entry_price = 0.0
                avg_cost = 0.0
                shares = 0

            # Execute take profit if triggered
            elif take_profit_triggered and position == 1:
                proceeds = shares * current_price
                comm = proceeds * self.commission
                cash = cash + proceeds - comm
                trades.append(
                    Trade(
                        date=dates[i]
                        if hasattr(dates[i], "date")
                        else pd.Timestamp(dates[i]),
                        action="sell",
                        price=current_price,
                        quantity=shares,
                        commission=comm,
                    )
                )
                position = 0
                entry_price = 0.0
                avg_cost = 0.0
                shares = 0

        # Final equity
        final_price = prices[-1]
        final_equity = cash + shares * final_price if position == 1 else cash

        # Calculate returns
        strategy_returns = pd.Series(equity).pct_change().dropna()
        pd.Series(buy_hold_equity).pct_change().dropna()

        # Sharpe ratio (annualized)
        if len(strategy_returns) > 0 and strategy_returns.std() > 0:
            sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0

        # Win rate (based on sell trades)
        sell_trades = [t for t in trades if t.action == "sell"]
        winning_trades = 0
        for t in sell_trades:
            # Find corresponding buy trade before this sell
            buy_trades = [
                bt for bt in trades if bt.action == "buy" and bt.date < t.date
            ]
            if buy_trades:
                last_buy = buy_trades[-1]
                if t.price > last_buy.price:  # Sold higher than bought
                    winning_trades += 1

        win_rate = winning_trades / len(sell_trades) if sell_trades else 0

        # Total returns
        # For Buy & Hold strategy, use pure price return (no trading costs)
        if isinstance(strategy, BuyAndHoldStrategy):
            total_return = strategy.calculate_return(df)
        else:
            total_return = (final_equity - self.initial_cash) / self.initial_cash
        buy_hold_return = (buy_hold_equity[-1] - self.initial_cash) / self.initial_cash

        return BacktestResult(
            strategy_name=strategy.name,
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=len(trades),
            buy_hold_return=buy_hold_return,
            trades=trades,
            equity_curve=pd.Series(equity, index=dates[: len(equity)]),
            daily_returns=strategy_returns,
        )

    def compare_strategies(
        self,
        df: pd.DataFrame,
        strategies: List[Strategy],
        full_history_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Compare multiple strategies.

        Args:
            df: DataFrame with backtest period data
            strategies: List of strategies to compare
            full_history_df: Full history DataFrame (for ML strategy signal generation)
        """
        results = []

        for strategy in strategies:
            # For ML-based strategies, use full history to generate signals
            if (
                isinstance(
                    strategy,
                    (
                        MLStrategy,
                        HybridStrategy,
                        RollingMLStrategy,
                        RollingHybridStrategy,
                    ),
                )
                and full_history_df is not None
            ):
                signals = strategy.generate_signals(full_history_df)
                # Extract only signals for the backtest period
                signals = signals.iloc[-len(df) :]
                result = self.run(df, strategy, precomputed_signals=signals)
            else:
                result = self.run(df, strategy)
            results.append(
                {
                    "Strategy": result.strategy_name,
                    "Total Return": result.total_return,
                    "Total Return Str": f"{result.total_return:.2%}",
                    "Buy & Hold": f"{result.buy_hold_return:.2%}",
                    "Sharpe Ratio": f"{result.sharpe_ratio:.2f}",
                    "Max Drawdown": f"{result.max_drawdown:.2%}",
                    "Win Rate": f"{result.win_rate:.2%}",
                    "Trades": result.total_trades,
                }
            )

        # Create DataFrame and sort by Total Return (descending)
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values("Total Return", ascending=False)

        # Drop the numeric column for display
        df_results = df_results.drop(columns=["Total Return"])
        df_results = df_results.rename(columns={"Total Return Str": "Total Return"})

        # Reorder columns
        cols = [
            "Strategy",
            "Total Return",
            "Buy & Hold",
            "Sharpe Ratio",
            "Max Drawdown",
            "Win Rate",
            "Trades",
        ]
        df_results = df_results[cols]

        return df_results


def run_backtest(
    df: pd.DataFrame, strategy: Strategy, initial_cash: float = 100000
) -> BacktestResult:
    """Convenience function to run a backtest."""
    config = get_config().get_section("backtest")
    engine = BacktestEngine(
        initial_cash=initial_cash or config.get("initial_cash", 100000),
        commission=config.get("commission", 0.001),
        slippage=config.get("slippage", 0.001),
    )
    return engine.run(df, strategy)
