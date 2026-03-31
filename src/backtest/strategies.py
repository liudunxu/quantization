"""Strategy configurations for backtesting.

This module provides a centralized location for all strategy configurations
to ensure consistency between decide.py and backtest.py.
"""

from typing import List, Optional, Literal
from .engine import (
    Strategy,
    BuyAndHoldStrategy,
    HighSellLowBuyStrategy,
    MLStrategy,
    HybridStrategy,
    RollingMLStrategy,
    RollingHybridStrategy,
)
from .rule_strategies import (
    MAGoldenCrossStrategy,
    BullTrendStrategy,
    ShrinkPullbackStrategy,
    BottomVolumeStrategy,
    BoxOscillationStrategy,
    EmotionCycleStrategy,
    VolumeBreakoutStrategy,
    OneYangThreeYinStrategy,
    MACDDivergenceStrategy,
)
from ..utils import get_param_manager


def get_market_strategies(
    model,
    market: Literal["a_share", "hk", "us"],
    stock_code: Optional[str] = None,
    min_samples: int = 20,
    require_bull_market_for_buy: bool = True,
) -> List[Strategy]:
    """Get market-specific strategies with tuned parameters.

    Parameters are loaded with priority: stock_code > market > default.

    Args:
        model: Trained ML model instance
        market: Market type ('a_share', 'hk', 'us')
        stock_code: Stock code (optional, for stock-specific params)
        min_samples: Minimum samples for ML strategies
        require_bull_market_for_buy: Require bull market for buy signals

    Returns:
        List of strategy instances with market-tuned parameters
    """
    param_manager = get_param_manager()

    # Get market parameters with priority: stock_code > market > default
    market_params = param_manager.get_market_params(market, stock_code)

    # Get rule strategy parameters with priority
    highsell_params = param_manager.get_all_strategy_params(
        "highsell_lowbuy", market, stock_code
    )
    ma_cross_params = param_manager.get_all_strategy_params(
        "ma_golden_cross", market, stock_code
    )
    bull_trend_params = param_manager.get_all_strategy_params(
        "bull_trend", market, stock_code
    )
    shrink_params = param_manager.get_all_strategy_params(
        "shrink_pullback", market, stock_code
    )
    bottom_params = param_manager.get_all_strategy_params(
        "bottom_volume", market, stock_code
    )
    box_params = param_manager.get_all_strategy_params(
        "box_oscillation", market, stock_code
    )
    volume_params = param_manager.get_all_strategy_params(
        "volume_breakout", market, stock_code
    )
    macd_params = param_manager.get_all_strategy_params(
        "macd_divergence", market, stock_code
    )

    return [
        # === Non-ML / Rule-based strategies ===
        BuyAndHoldStrategy(),
        HighSellLowBuyStrategy(
            lookback=highsell_params.get("lookback", 20),
            threshold=highsell_params.get("threshold", 0.15),
        ),
        # Rule-based strategies from strategy references
        MAGoldenCrossStrategy(
            fast_ma=ma_cross_params.get("fast_ma", 5),
            slow_ma=ma_cross_params.get("slow_ma", 10),
            volume_ratio=ma_cross_params.get("volume_ratio", 1.0),
        ),
        BullTrendStrategy(
            ma5_period=bull_trend_params.get("ma5_period", 5),
            ma10_period=bull_trend_params.get("ma10_period", 10),
            ma20_period=bull_trend_params.get("ma20_period", 20),
        ),
        ShrinkPullbackStrategy(
            lookback=shrink_params.get("lookback", 5),
            ma_period=shrink_params.get("ma_period", 5),
            volume_shrink=shrink_params.get("volume_shrink", 0.8),
        ),
        BottomVolumeStrategy(
            drop_threshold=bottom_params.get("drop_threshold", 0.10),
            volume_multiplier=bottom_params.get("volume_multiplier", 2.0),
        ),
        BoxOscillationStrategy(
            lookback=box_params.get("lookback", 40),
            support_margin=box_params.get("support_margin", 0.03),
            resistance_margin=box_params.get("resistance_margin", 0.03),
        ),
        VolumeBreakoutStrategy(
            lookback=volume_params.get("lookback", 15),
            volume_multiplier=volume_params.get("volume_multiplier", 1.5),
        ),
        MACDDivergenceStrategy(
            lookback=macd_params.get("lookback", 15),
        ),
        # === ML-based strategies ===
        # Pure ML strategy
        MLStrategy(
            model,
            name=f"ML Strategy ({market.upper()})",
            min_samples=min_samples,
            confidence_threshold=market_params.get("ml_confidence_threshold", 0.35),
            bear_market_threshold=market_params.get("bear_market_threshold", -0.008),
            require_bull_market_for_buy=require_bull_market_for_buy,
        ),
        # Hybrid strategy
        HybridStrategy(
            model,
            lookback=market_params.get("highsell_lookback", 10) - 5,
            threshold=market_params.get("highsell_threshold", 0.08) - 0.02,
            min_samples=min_samples,
            ml_confidence_threshold=market_params.get("ml_confidence_threshold", 0.35),
            bear_market_threshold=market_params.get("bear_market_threshold", -0.008),
            require_bull_market_for_buy=require_bull_market_for_buy,
        ),
        # Rolling ML strategy
        RollingMLStrategy(
            model_class=type(model),
            train_window=market_params.get("rolling_train_window", 90),
            retrain_interval=market_params.get("rolling_retrain_interval", 10),
            min_samples=min_samples,
            confidence_threshold=market_params.get("ml_confidence_threshold", 0.35),
            bear_market_threshold=market_params.get("bear_market_threshold", -0.008),
            require_bull_market_for_buy=require_bull_market_for_buy,
        ),
        # Rolling Hybrid strategy
        RollingHybridStrategy(
            model_class=type(model),
            train_window=market_params.get("rolling_train_window", 90),
            retrain_interval=market_params.get("rolling_retrain_interval", 10),
            lookback=market_params.get("highsell_lookback", 10) - 5,
            threshold=market_params.get("highsell_threshold", 0.08) - 0.02,
            min_samples=min_samples,
            ml_confidence_threshold=market_params.get("ml_confidence_threshold", 0.35)
            - 0.05,
            bear_market_threshold=market_params.get("bear_market_threshold", -0.008),
            require_bull_market_for_buy=require_bull_market_for_buy,
        ),
    ]
