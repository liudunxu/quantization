"""Backtest package."""

from .engine import (
    BacktestEngine,
    BacktestResult,
    BuyAndHoldStrategy,
    HighSellLowBuyStrategy,
    HybridStrategy,
    MLStrategy,
    RollingHybridStrategy,
    RollingMLStrategy,
    Strategy,
    Trade,
    run_backtest,
)
from .rule_strategies import (
    BottomVolumeStrategy,
    BoxOscillationStrategy,
    BullTrendStrategy,
    EmotionCycleStrategy,
    MACDDivergenceStrategy,
    MAGoldenCrossStrategy,
    OneYangThreeYinStrategy,
    ShrinkPullbackStrategy,
    VolumeBreakoutStrategy,
)
from .strategies import (
    MARKET_PARAMS,
    get_default_strategies,
    get_market_strategies,
    get_quick_strategies,
)

__all__ = [
    'BacktestEngine',
    'BacktestResult',
    'Trade',
    'Strategy',
    'BuyAndHoldStrategy',
    'HighSellLowBuyStrategy',
    'MLStrategy',
    'HybridStrategy',
    'RollingMLStrategy',
    'RollingHybridStrategy',
    'run_backtest',
    'get_default_strategies',
    'get_quick_strategies',
    'get_market_strategies',
    'MARKET_PARAMS',
    # Rule-based strategies
    'MAGoldenCrossStrategy',
    'BullTrendStrategy',
    'ShrinkPullbackStrategy',
    'BottomVolumeStrategy',
    'BoxOscillationStrategy',
    'EmotionCycleStrategy',
    'VolumeBreakoutStrategy',
    'OneYangThreeYinStrategy',
    'MACDDivergenceStrategy',
]
