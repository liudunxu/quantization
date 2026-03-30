"""Backtest package."""

from .engine import (
    BacktestEngine,
    BacktestResult,
    Trade,
    Strategy,
    BuyAndHoldStrategy,
    HighSellLowBuyStrategy,
    MLStrategy,
    HybridStrategy,
    RollingMLStrategy,
    RollingHybridStrategy,
    run_backtest
)
from .strategies import (
    get_default_strategies,
    get_quick_strategies,
    get_market_strategies,
    MARKET_PARAMS,
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
]
