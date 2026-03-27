#!/usr/bin/env python3
"""Stock trading decision script.

Usage:
    python scripts/decide.py --stock 000001.SZ
    python scripts/decide.py --stock 0700.HK
    python scripts/decide.py --stock AAPL
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_cache, get_config, StockInfoResolver
from src.features import get_feature_combinator
from src.models import StockTradingModel, get_model
from src.backtest import (
    BacktestEngine,
    BuyAndHoldStrategy,
    HighSellLowBuyStrategy,
    MLStrategy,
    run_backtest
)


def parse_args():
    parser = argparse.ArgumentParser(description='Stock trading decision system')
    parser.add_argument('--stock', required=True, help='Stock code (e.g., 000001.SZ, 0700.HK, AAPL)')
    parser.add_argument('--refresh', action='store_true', help='Force refresh cached data')
    parser.add_argument('--backtest-days', type=int, default=30, help='Number of days for backtest')
    parser.add_argument('--train-days', type=int, default=365, help='Number of days for training')
    return parser.parse_args()


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


def print_decision(action: int, confidence: float, probabilities: dict) -> None:
    """Print trading decision."""
    action_map = {1: 'BUY', 0: 'HOLD', -1: 'SELL'}
    action_str = action_map.get(action, 'UNKNOWN')

    print_section(" TRADING DECISION")
    print(f"\n  Action   : {action_str}")
    print(f"  Confidence: {confidence:.2%}")
    print(f"\n  Probabilities:")
    print(f"    - Buy : {probabilities.get('buy_probability', 0):.2%}")
    print(f"    - Hold: {probabilities.get('hold_probability', 0):.2%}")
    print(f"    - Sell: {probabilities.get('sell_probability', 0):.2%}")


def print_backtest_comparison(results_df: pd.DataFrame) -> None:
    """Print backtest comparison."""
    print_section(" BACKTEST RESULTS (vs BUY & HOLD)")
    print()
    print(results_df.to_string(index=False))


def print_feature_importance(model: StockTradingModel, top_n: int = 10) -> None:
    """Print top feature importances."""
    try:
        importance = model.get_feature_importance()
        print_section(f" TOP {top_n} FEATURE IMPORTANCE")
        print()
        top_features = importance.head(top_n)
        for _, row in top_features.iterrows():
            bar = '█' * int(row['importance'] / 2)
            print(f"  {row['feature']:<30} {row['importance']:6.2f} {bar}")
    except Exception as e:
        print(f"  (Could not display feature importance: {e})")


def main():
    args = parse_args()
    stock_code = args.stock

    print_section(f" STOCK TRADING DECISION SYSTEM")
    print(f"\n  Stock Code: {stock_code}")

    # Resolve stock info
    try:
        stock_info = StockInfoResolver.resolve(stock_code)
        print(f"  Market    : {stock_info.market}")
        print(f"  Exchange  : {stock_info.exchange}")
    except ValueError as e:
        print(f"  Error: {e}")
        sys.exit(1)

    # Get config
    config = get_config()
    cache = get_cache(config.get('data.cache_dir', 'cache'))

    # Initialize feature combinator
    combinator = get_feature_combinator(cache)

    # Extract and combine features
    print_section(" FETCHING & PROCESSING DATA")

    print(f"\n  Training period : {args.train_days} days")
    print(f"  Backtest period : {args.backtest_days} days")
    print(f"  Force refresh   : {args.refresh}")

    print("\n  Extracting features...")
    try:
        features_df = combinator.get_combined_features(
            stock_code,
            days=args.train_days,
            force_refresh=args.refresh
        )
        if features_df.empty:
            print("  Error: Could not fetch data for this stock")
            sys.exit(1)
        print(f"  Total samples   : {len(features_df)}")
    except Exception as e:
        print(f"  Error fetching data: {e}")
        sys.exit(1)

    # Train model
    print_section(" TRAINING MODEL")

    model = StockTradingModel()

    # Split data for training and evaluation
    train_df = features_df.iloc[:-args.backtest_days]
    eval_df = features_df.iloc[-args.backtest_days:]

    print(f"\n  Training samples  : {len(train_df)}")
    print(f"  Evaluation samples: {len(eval_df)}")

    try:
        train_metrics = model.train(train_df, forward_days=5, threshold=0.02, eval_df=eval_df)
        print(f"\n  Training accuracy : {train_metrics['train_accuracy']:.2%}")
        print(f"  Features used     : {train_metrics['feature_count']}")
        print(f"  Label distribution:")
        print(f"    - Buy : {train_metrics['label_distribution']['buy']}")
        print(f"    - Sell: {train_metrics['label_distribution']['sell']}")
    except Exception as e:
        print(f"  Error training model: {e}")
        sys.exit(1)

    # Get prediction
    print_section(" GENERATING PREDICTION")

    try:
        action, confidence = model.predict(features_df)
        probabilities = model.predict_proba(features_df)
        print_decision(action, confidence, probabilities)
    except Exception as e:
        print(f"  Error generating prediction: {e}")
        sys.exit(1)

    # Feature importance
    print_feature_importance(model, top_n=10)

    # Backtest comparison
    print_section(" BACKTEST COMPARISON")

    # Use only backtest period data
    backtest_df = features_df.iloc[-args.backtest_days:].copy()

    if len(backtest_df) < 10:
        print("\n  Not enough data for backtesting")
    else:
        strategies = [
            BuyAndHoldStrategy(),
            HighSellLowBuyStrategy(lookback=20, threshold=0.15),
            MLStrategy(model, name="ML Strategy")
        ]

        engine = BacktestEngine(
            initial_cash=config.get('backtest.initial_cash', 100000),
            commission=config.get('backtest.commission', 0.001),
            slippage=config.get('backtest.slippage', 0.001)
        )

        results_df = engine.compare_strategies(backtest_df, strategies, full_history_df=features_df)
        print_backtest_comparison(results_df)

    # Cache info
    print_section(" CACHE STATUS")
    cache_info = cache.get_cache_info(stock_code)
    print(f"\n  Cached feature types: {', '.join(cache_info.get('cached_types', ['None']))}")
    print(f"  Total cached items   : {cache_info.get('count', 0)}")

    print("\n" + "="*60)
    print(" Decision process completed!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
