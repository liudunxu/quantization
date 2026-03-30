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


# Market-specific strategy parameters
MARKET_PARAMS = {
    'a_share': {
        # A股市场：政策影响大、风格切换快、波动性高
        'highsell_lookback': 15,      # 更短的lookback适应快速变化
        'highsell_threshold': 0.12,   # 较低threshold产生更多信号
        'ml_confidence_threshold': 0.45,  # 较低置信度，适应快速变化
        'rolling_train_window': 120,   # 更短的训练窗口
        'rolling_retrain_interval': 15,  # 更频繁的retrain
        'bear_market_threshold': -0.008,  # 更严格的熊市定义
    },
    'hk': {
        # 港股市场：受A股和美股双重影响
        'highsell_lookback': 20,
        'highsell_threshold': 0.15,
        'ml_confidence_threshold': 0.50,
        'rolling_train_window': 180,
        'rolling_retrain_interval': 20,
        'bear_market_threshold': -0.005,
    },
    'us': {
        # 美股市场：趋势性强、波动相对平稳
        'highsell_lookback': 25,       # 更长的lookback捕捉长期趋势
        'highsell_threshold': 0.18,    # 较高threshold减少交易频率
        'ml_confidence_threshold': 0.55,  # 较高置信度
        'rolling_train_window': 240,   # 更长的训练窗口
        'rolling_retrain_interval': 30,  # 较少的retrain
        'bear_market_threshold': -0.003,  # 较宽松的熊市定义
    },
    'default': {
        # 默认参数
        'highsell_lookback': 20,
        'highsell_threshold': 0.15,
        'ml_confidence_threshold': 0.50,
        'rolling_train_window': 180,
        'rolling_retrain_interval': 20,
        'bear_market_threshold': -0.005,
    }
}


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


def get_market_strategies(
    model,
    market: Literal['a_share', 'hk', 'us'],
    min_samples: int = 20,
    require_bull_market_for_buy: bool = True,
) -> List[Strategy]:
    """Get market-specific strategies with tuned parameters.

    Args:
        model: Trained ML model instance
        market: Market type ('a_share', 'hk', 'us')
        min_samples: Minimum samples for ML strategies
        require_bull_market_for_buy: Require bull market for buy signals

    Returns:
        List of strategy instances with market-tuned parameters
    """
    params = MARKET_PARAMS.get(market, MARKET_PARAMS['default'])

    return [
        # Non-ML strategies
        BuyAndHoldStrategy(),
        HighSellLowBuyStrategy(
            lookback=params['highsell_lookback'],
            threshold=params['highsell_threshold']
        ),

        # Pure ML strategy
        MLStrategy(
            model,
            name=f"ML Strategy ({market.upper()})",
            min_samples=min_samples,
            confidence_threshold=params['ml_confidence_threshold'],
            bear_market_threshold=params['bear_market_threshold'],
            require_bull_market_for_buy=require_bull_market_for_buy
        ),

        # Hybrid strategy
        HybridStrategy(
            model,
            lookback=params['highsell_lookback'] - 5,
            threshold=params['highsell_threshold'] - 0.02,
            min_samples=min_samples,
            ml_confidence_threshold=params['ml_confidence_threshold'],
            bear_market_threshold=params['bear_market_threshold'],
            require_bull_market_for_buy=require_bull_market_for_buy
        ),

        # Rolling ML strategy
        RollingMLStrategy(
            model_class=type(model),
            train_window=params['rolling_train_window'],
            retrain_interval=params['rolling_retrain_interval'],
            min_samples=min_samples,
            confidence_threshold=params['ml_confidence_threshold'],
            bear_market_threshold=params['bear_market_threshold'],
            require_bull_market_for_buy=require_bull_market_for_buy
        ),

        # Rolling Hybrid strategy
        RollingHybridStrategy(
            model_class=type(model),
            train_window=params['rolling_train_window'],
            retrain_interval=params['rolling_retrain_interval'],
            lookback=params['highsell_lookback'] - 5,
            threshold=params['highsell_threshold'] - 0.02,
            min_samples=min_samples,
            ml_confidence_threshold=params['ml_confidence_threshold'] - 0.05,
            bear_market_threshold=params['bear_market_threshold'],
            require_bull_market_for_buy=require_bull_market_for_buy
        ),
    ]
