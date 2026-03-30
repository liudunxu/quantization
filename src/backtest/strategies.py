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


# Market-specific strategy parameters
MARKET_PARAMS = {
    'a_share': {
        # A股市场：政策影响大、风格切换快、波动性高
        'highsell_lookback': 10,      # 更短产生更多信号
        'highsell_threshold': 0.08,    # 更低threshold产生更多信号
        'ml_confidence_threshold': 0.35,  # 较低置信度产生更多信号
        'rolling_train_window': 90,    # 更短的训练窗口
        'rolling_retrain_interval': 10,  # 更频繁的retrain
        'bear_market_threshold': -0.01, # 放宽熊市定义
    },
    'hk': {
        # 港股市场：受A股和美股双重影响
        'highsell_lookback': 15,
        'highsell_threshold': 0.10,
        'ml_confidence_threshold': 0.40,
        'rolling_train_window': 120,
        'rolling_retrain_interval': 15,
        'bear_market_threshold': -0.008,
    },
    'us': {
        # 美股市场：趋势性强、波动相对平稳
        'highsell_lookback': 20,       # 中等lookback
        'highsell_threshold': 0.12,     # 适度threshold
        'ml_confidence_threshold': 0.45,  # 适中置信度
        'rolling_train_window': 180,   # 中等训练窗口
        'rolling_retrain_interval': 20,  # 适度retrain频率
        'bear_market_threshold': -0.005,
    },
    'default': {
        # 默认参数
        'highsell_lookback': 15,
        'highsell_threshold': 0.10,
        'ml_confidence_threshold': 0.40,
        'rolling_train_window': 120,
        'rolling_retrain_interval': 15,
        'bear_market_threshold': -0.008,
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
        # === Non-ML / Rule-based strategies ===
        BuyAndHoldStrategy(),
        HighSellLowBuyStrategy(
            lookback=params['highsell_lookback'],
            threshold=params['highsell_threshold']
        ),

        # Rule-based strategies from strategy references
        MAGoldenCrossStrategy(fast_ma=5, slow_ma=10, volume_ratio=1.0),  # 降低量能确认阈值
        BullTrendStrategy(ma5_period=5, ma10_period=10, ma20_period=20),
        ShrinkPullbackStrategy(lookback=5, ma_period=5, volume_shrink=0.8),  # 放宽缩量要求
        BottomVolumeStrategy(drop_threshold=0.10, volume_multiplier=2.0),  # 降低跌幅和量能要求
        BoxOscillationStrategy(lookback=40, support_margin=0.03, resistance_margin=0.03),  # 更宽的支撑阻力
        VolumeBreakoutStrategy(lookback=15, volume_multiplier=1.5),  # 更短周期、更低量能
        MACDDivergenceStrategy(lookback=15),  # 更短周期

        # === ML-based strategies ===
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
