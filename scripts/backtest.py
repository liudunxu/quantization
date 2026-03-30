#!/usr/bin/env python3
"""Backtest script for stock trading strategies.

Usage:
    python scripts/backtest.py --stock 000001.SZ --days 30
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
from src.models import StockTradingModel
from src.backtest import (
    BacktestEngine,
    BuyAndHoldStrategy,
    HighSellLowBuyStrategy,
    MLStrategy,
    BacktestResult
)


def parse_args():
    parser = argparse.ArgumentParser(description='Backtest stock trading strategies')
    parser.add_argument('--stock', required=True, help='Stock code')
    parser.add_argument('--backtest-days', type=int, default=30, help='Number of days for backtest')
    parser.add_argument('--train-days', type=int, default=365, help='Training days')
    parser.add_argument('--initial-cash', type=float, default=100000, help='Initial cash')
    parser.add_argument('--output', help='Output file for results (CSV)')
    return parser.parse_args()


def print_result(result: BacktestResult, prefix: str = "  ") -> None:
    """Print a single backtest result."""
    print(f"\n{prefix}Strategy     : {result.strategy_name}")
    print(f"{prefix}Total Return  : {result.total_return:+.2%}")
    print(f"{prefix}Buy & Hold     : {result.buy_hold_return:+.2%}")
    print(f"{prefix}vs Benchmark   : {result.total_return - result.buy_hold_return:+.2%}")
    print(f"{prefix}Sharpe Ratio   : {result.sharpe_ratio:.2f}")
    print(f"{prefix}Max Drawdown   : {result.max_drawdown:.2%}")
    print(f"{prefix}Win Rate       : {result.win_rate:.2%}")
    print(f"{prefix}Total Trades   : {result.total_trades}")

    if result.trades:
        print(f"{prefix}Last Trade     : {result.trades[-1].action} on {result.trades[-1].date.strftime('%Y-%m-%d')}")


def main():
    args = parse_args()
    stock_code = args.stock

    print("="*60)
    print(" STOCK TRADING BACKTEST")
    print("="*60)
    print(f"\n  Stock Code     : {stock_code}")
    print(f"  Backtest Days  : {args.backtest_days}")
    print(f"  Training Days  : {args.train_days}")
    print(f"  Initial Cash   : ${args.initial_cash:,.2f}")

    # Get data
    config = get_config()
    cache = get_cache(config.get('data.cache_dir', 'cache'))
    combinator = get_feature_combinator(cache)

    print("\n  Fetching data...")
    try:
        features_df = combinator.get_combined_features(
            stock_code,
            days=args.train_days,
            force_refresh=False
        )
        if features_df.empty:
            print("  Error: Could not fetch data")
            sys.exit(1)
    except Exception as e:
        print(f"  Error fetching data: {e}")
        sys.exit(1)

    # Split data
    backtest_df = features_df.iloc[-args.backtest_days:].copy()
    train_df = features_df.iloc[:-args.backtest_days]

    print(f"  Backtest samples: {len(backtest_df)}")

    if len(backtest_df) < 10:
        print("  Error: Not enough data for backtest")
        sys.exit(1)

    # Train ML model
    print("\n  Training ML model...")
    model = StockTradingModel()

    # Get aggressive training params from config
    model_config = config.get_section('model')
    training_config = model_config.get('training', {})
    forward_days = training_config.get('forward_days', 5)
    threshold = training_config.get('threshold', 0.01)
    min_samples = model_config.get('strategy', {}).get('min_samples', 20)
    confidence_threshold = model_config.get('strategy', {}).get('confidence_threshold', 0.25)

    # Composite label parameters
    use_composite_labels = training_config.get('use_composite_labels', True)
    trend_weight = training_config.get('trend_weight', 0.30)
    momentum_weight = training_config.get('momentum_weight', 0.30)
    market_weight = training_config.get('market_weight', 0.20)

    try:
        model.train(
            train_df, forward_days=forward_days, threshold=threshold,
            use_composite_labels=use_composite_labels,
            trend_weight=trend_weight,
            momentum_weight=momentum_weight,
            market_weight=market_weight
        )
        print(f"  Model trained (forward_days={forward_days}, threshold={threshold:.3f})")
    except Exception as e:
        print(f"  Error training model: {e}")
        sys.exit(1)

    # Run backtests
    print("\n" + "="*60)
    print(" BACKTEST RESULTS")
    print("="*60)

    engine = BacktestEngine(
        initial_cash=args.initial_cash,
        commission=config.get('backtest.commission', 0.001),
        slippage=config.get('backtest.slippage', 0.001)
    )

    strategies = [
        BuyAndHoldStrategy(),
        HighSellLowBuyStrategy(lookback=20, threshold=0.15),
        HighSellLowBuyStrategy(lookback=10, threshold=0.10),
        MLStrategy(
            model,
            name="ML Strategy (CatBoost)",
            min_samples=min_samples,
            confidence_threshold=0.55,
            bear_market_threshold=0.005,  # Market return >0.5% is bullish
            require_bull_market_for_buy=True
        )
    ]

    results = []
    for strategy in strategies:
        # For MLStrategy, use full history to generate signals
        if isinstance(strategy, MLStrategy):
            signals = strategy.generate_signals(features_df)
            signals = signals.iloc[-len(backtest_df):]
            result = engine.run(backtest_df, strategy, precomputed_signals=signals)
        else:
            result = engine.run(backtest_df, strategy)
        results.append(result)
        print_result(result)

    # Summary
    print("\n" + "="*60)
    print(" STRATEGY COMPARISON SUMMARY")
    print("="*60)

    print("\n  vs Buy & Hold (higher is better):")
    best_strategy = None
    best_vs_benchmark = -999

    for result in results:
        vs_benchmark = result.total_return - result.buy_hold_return
        marker = ""
        if vs_benchmark > best_vs_benchmark:
            best_vs_benchmark = vs_benchmark
            best_strategy = result.strategy_name
            marker = " <-- BEST"

        print(f"    {result.strategy_name:<35} {vs_benchmark:+.2%}{marker}")

    print(f"\n  Best strategy: {best_strategy}")

    # Save results if requested
    if args.output:
        output_data = []
        for result in results:
            output_data.append({
                'strategy': result.strategy_name,
                'total_return': result.total_return,
                'buy_hold_return': result.buy_hold_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate,
                'total_trades': result.total_trades
            })

        df = pd.DataFrame(output_data)
        df.to_csv(args.output, index=False)
        print(f"\n  Results saved to: {args.output}")

    print("\n" + "="*60 + "\n")


if __name__ == '__main__':
    main()
