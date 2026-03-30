"""Backtesting engine for stock trading strategies."""

from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
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

    def __init__(self, name: str):
        self.name = name

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals.

        Returns:
            Series with values 1 (buy), 0 (hold), -1 (sell)
        """
        raise NotImplementedError


class BuyAndHoldStrategy(Strategy):
    """Simple buy and hold strategy."""

    def __init__(self):
        super().__init__("Buy & Hold")

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        signals.iloc[0] = 1  # Buy on first day
        return signals


class HighSellLowBuyStrategy(Strategy):
    """Contrarian strategy: sell when price is high, buy when low."""

    def __init__(self, lookback: int = 20, threshold: float = 0.1):
        super().__init__(f"High Sell Low Buy (L:{lookback}, T:{threshold})")
        self.lookback = lookback
        self.threshold = threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        close = df['close']

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
        require_bull_market_for_buy: bool = True
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
        super().__init__(name)
        self.model = model
        self.min_samples = min_samples
        self.confidence_threshold = confidence_threshold
        self.bear_market_threshold = bear_market_threshold
        self.require_bull_market_for_buy = require_bull_market_for_buy

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)

        if len(df) < self.min_samples:
            return signals

        # Determine market regime based on index returns if available
        market_returns = None
        if 'index_returns' in df.columns:
            market_returns = df['index_returns'].values

        for i in range(self.min_samples, len(df)):
            try:
                pred, confidence = self.model.predict(df.iloc[:i+1])

                # Check market regime
                is_bear_market = False
                if market_returns is not None and i >= 1:
                    # Use recent market return to determine regime
                    recent_market_return = market_returns[i-1] if i > 0 else 0
                    if recent_market_return < self.bear_market_threshold:
                        is_bear_market = True

                # Only signal if confidence exceeds threshold
                if confidence >= self.confidence_threshold:
                    # In bear market, only allow sell signals (unless explicitly bullish)
                    if is_bear_market and self.require_bull_market_for_buy:
                        if pred == -1:  # Only sell in bear market
                            signals.iloc[i] = pred
                        else:
                            signals.iloc[i] = 0  # Hold in bear market unless very confident sell
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
        bear_market_threshold: float = 0.005,
        require_bull_market_for_buy: bool = True
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
        super().__init__(f"Hybrid Strategy (ML+HSSLB)")
        self.model = model
        self.min_samples = min_samples
        self.ml_confidence_threshold = ml_confidence_threshold
        self.bear_market_threshold = bear_market_threshold
        self.require_bull_market_for_buy = require_bull_market_for_buy
        # Simple strategy for fallback
        self.simple_strategy = HighSellLowBuyStrategy(lookback=lookback, threshold=threshold)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)

        if len(df) < self.min_samples:
            return signals

        # Get simple strategy signals
        simple_signals = self.simple_strategy.generate_signals(df)

        # Get market returns for regime detection
        market_returns = None
        if 'index_returns' in df.columns:
            market_returns = df['index_returns'].values

        for i in range(self.min_samples, len(df)):
            try:
                # Get ML prediction
                pred, confidence = self.model.predict(df.iloc[:i+1])
                ml_signal = pred

                # Check market regime
                is_bear_market = False
                if market_returns is not None and i >= 1:
                    recent_market_return = market_returns[i-1] if i > 0 else 0
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

                    # Apply market filter for buys
                    if is_bear_market and self.require_bull_market_for_buy:
                        if final_signal == 1:
                            final_signal = 0  # No buy in bear market

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
    """Backtesting engine."""

    def __init__(self, initial_cash: float = 100000, commission: float = 0.001, slippage: float = 0.001):
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage

    def run(
        self,
        df: pd.DataFrame,
        strategy: Strategy,
        benchmark: Optional[Strategy] = None,
        precomputed_signals: Optional[pd.Series] = None
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
        prices = df['close'].values
        dates = df['date'].values if 'date' in df.columns else df.index

        # Initialize
        cash = self.initial_cash
        position = 0  # 0 = no stock, 1 = long
        shares = 0
        trades = []
        equity = [self.initial_cash]
        current_drawdown = 0
        max_drawdown = 0
        peak_equity = self.initial_cash

        buy_hold_shares = self.initial_cash / prices[0]
        buy_hold_equity = [buy_hold_shares * p for p in prices]

        # Handle initial buy signal at index 0
        if signals.iloc[0] == 1 and position == 0:
            first_price = prices[0] * (1 - self.slippage)
            shares = int(self.initial_cash / first_price)
            cost = shares * first_price
            comm = cost * self.commission
            cash = self.initial_cash - cost - comm
            position = 1
            trades.append(Trade(
                date=dates[0] if hasattr(dates[0], 'date') else pd.Timestamp(dates[0]),
                action='buy',
                price=first_price,
                quantity=shares,
                commission=comm
            ))

        for i in range(1, len(prices)):
            current_price = prices[i] * (1 + self.slippage if position == 1 else 1 - self.slippage)
            prev_price = prices[i-1]

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

            # Execute signals
            if signals.iloc[i] == 1 and position == 0:
                # Buy
                shares = int(cash / current_price)
                cost = shares * current_price
                comm = cost * self.commission
                cash = cash - cost - comm
                position = 1
                trades.append(Trade(
                    date=dates[i] if hasattr(dates[i], 'date') else pd.Timestamp(dates[i]),
                    action='buy',
                    price=current_price,
                    quantity=shares,
                    commission=comm
                ))

            elif signals.iloc[i] == -1 and position == 1:
                # Sell
                proceeds = shares * current_price
                comm = proceeds * self.commission
                cash = cash + proceeds - comm
                position = 0
                trades.append(Trade(
                    date=dates[i] if hasattr(dates[i], 'date') else pd.Timestamp(dates[i]),
                    action='sell',
                    price=current_price,
                    quantity=shares,
                    commission=comm
                ))

        # Final equity
        final_price = prices[-1]
        final_equity = cash + shares * final_price if position == 1 else cash

        # Calculate returns
        strategy_returns = pd.Series(equity).pct_change().dropna()
        buy_hold_returns = pd.Series(buy_hold_equity).pct_change().dropna()

        # Sharpe ratio (annualized)
        if len(strategy_returns) > 0 and strategy_returns.std() > 0:
            sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0

        # Win rate
        winning_trades = sum(1 for t in trades if t.action == 'sell' and len(trades) > 0)
        total_exit_trades = sum(1 for t in trades if t.action == 'sell')
        win_rate = winning_trades / total_exit_trades if total_exit_trades > 0 else 0

        # Total returns
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
            equity_curve=pd.Series(equity, index=dates[:len(equity)]),
            daily_returns=strategy_returns
        )

    def compare_strategies(
        self,
        df: pd.DataFrame,
        strategies: List[Strategy],
        full_history_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Compare multiple strategies.

        Args:
            df: DataFrame with backtest period data
            strategies: List of strategies to compare
            full_history_df: Full history DataFrame (for ML strategy signal generation)
        """
        results = []

        for strategy in strategies:
            # For MLStrategy, use full history to generate signals
            if isinstance(strategy, MLStrategy) and full_history_df is not None:
                signals = strategy.generate_signals(full_history_df)
                # Extract only signals for the backtest period
                signals = signals.iloc[-len(df):]
                result = self.run(df, strategy, precomputed_signals=signals)
            else:
                result = self.run(df, strategy)
            results.append({
                'Strategy': result.strategy_name,
                'Total Return': f"{result.total_return:.2%}",
                'Buy & Hold': f"{result.buy_hold_return:.2%}",
                'Sharpe Ratio': f"{result.sharpe_ratio:.2f}",
                'Max Drawdown': f"{result.max_drawdown:.2%}",
                'Win Rate': f"{result.win_rate:.2%}",
                'Trades': result.total_trades
            })

        return pd.DataFrame(results)


def run_backtest(
    df: pd.DataFrame,
    strategy: Strategy,
    initial_cash: float = 100000
) -> BacktestResult:
    """Convenience function to run a backtest."""
    config = get_config().get_section('backtest')
    engine = BacktestEngine(
        initial_cash=initial_cash or config.get('initial_cash', 100000),
        commission=config.get('commission', 0.001),
        slippage=config.get('slippage', 0.001)
    )
    return engine.run(df, strategy)
