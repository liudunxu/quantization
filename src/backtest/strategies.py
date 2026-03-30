"""Strategy configurations for backtesting.

This module provides a centralized location for all strategy configurations
to ensure consistency between decide.py and backtest.py.
"""

from typing import List, Optional
from .engine import (
    Strategy,
    BuyAndHoldStrategy,
    HighSellLowBuyStrategy,
    MLStrategy,
    HybridStrategy,
    RollingMLStrategy,
    RollingHybridStrategy,
)


def get_default_strategies(
    model,
    min_samples: int = 20,
    ml_confidence_threshold: float = 0.50,
    bear_market_threshold: float = -0.005,
    require_bull_market_for_buy: bool = True,
) -> List[Strategy]:
    """Get default strategies for backtesting.

    Args:
        model: Trained ML model instance
        min_samples: Minimum samples for ML strategies
        ml_confidence_threshold: ML confidence threshold
        bear_market_threshold: Bear market threshold
        require_bull_market_for_buy: Require bull market for buy signals

    Returns:
        List of strategy instances
    """
    return [
        # Non-ML strategies
        BuyAndHoldStrategy(),
        HighSellLowBuyStrategy(lookback=20, threshold=0.15),

        # Pure ML strategy
        MLStrategy(
            model,
            name="ML Strategy (CatBoost)",
            min_samples=min_samples,
            confidence_threshold=0.55,
            bear_market_threshold=bear_market_threshold,
            require_bull_market_for_buy=require_bull_market_for_buy
        ),

        # Hybrid strategy
        HybridStrategy(
            model,
            lookback=10,
            threshold=0.10,
            min_samples=min_samples,
            ml_confidence_threshold=ml_confidence_threshold,
            bear_market_threshold=bear_market_threshold,
            require_bull_market_for_buy=require_bull_market_for_buy
        ),

        # Rolling ML strategy
        RollingMLStrategy(
            model_class=type(model),
            train_window=180,
            retrain_interval=20,
            min_samples=min_samples,
            confidence_threshold=0.50,
            bear_market_threshold=bear_market_threshold,
            require_bull_market_for_buy=require_bull_market_for_buy
        ),

        # Rolling Hybrid strategy
        RollingHybridStrategy(
            model_class=type(model),
            train_window=180,
            retrain_interval=15,
            lookback=10,
            threshold=0.10,
            min_samples=min_samples,
            ml_confidence_threshold=0.45,
            bear_market_threshold=bear_market_threshold,
            require_bull_market_for_buy=require_bull_market_for_buy
        ),
    ]


def get_quick_strategies(model, min_samples: int = 20) -> List[Strategy]:
    """Get quick strategies for decide.py (fewer strategies, faster).

    Args:
        model: Trained ML model instance
        min_samples: Minimum samples for ML strategies

    Returns:
        List of strategy instances
    """
    return [
        BuyAndHoldStrategy(),
        HighSellLowBuyStrategy(lookback=20, threshold=0.15),
        MLStrategy(
            model,
            name="ML Strategy",
            min_samples=min_samples,
            confidence_threshold=0.50,
            bear_market_threshold=-0.005,
            require_bull_market_for_buy=True
        ),
        HybridStrategy(
            model,
            lookback=10,
            threshold=0.10,
            min_samples=min_samples,
            ml_confidence_threshold=0.50,
            bear_market_threshold=-0.005,
            require_bull_market_for_buy=True
        ),
    ]
