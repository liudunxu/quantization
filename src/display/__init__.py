"""Display package for formatting and printing results."""

from .formatters import (
    print_section,
    print_feature_importance,
    print_stock_core_data,
    print_backtest_results,
    print_strategy_comparison,
    print_final_recommendation,
)
from .prediction_formatter import PredictionFormatter

__all__ = [
    "print_section",
    "print_feature_importance",
    "print_stock_core_data",
    "print_backtest_results",
    "print_strategy_comparison",
    "print_final_recommendation",
    "PredictionFormatter",
]
