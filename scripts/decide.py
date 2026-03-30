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
    run_backtest,
    get_market_strategies,
    MLStrategy,
    HybridStrategy,
    RollingMLStrategy,
    RollingHybridStrategy,
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


def get_strategy_decision(signals: pd.Series) -> tuple:
    """Get the final decision from signals series.

    Returns:
        tuple: (action_str, confidence)
            action_str: 'BUY', 'HOLD', or 'SELL'
            confidence: 1.0 if clear signal, 0.5 if ambiguous
    """
    if signals.empty:
        return 'HOLD', 0.0

    last_signal = signals.iloc[-1]
    if last_signal == 1:
        return 'BUY', 1.0
    elif last_signal == -1:
        return 'SELL', 1.0
    else:
        # Check recent signals for conviction
        recent = signals.iloc[-5:]
        if recent.sum() > 0:
            return 'HOLD', 0.5
        elif recent.sum() < 0:
            return 'HOLD', 0.5
        return 'HOLD', 0.0


def print_all_strategy_decisions(
    strategies: list,
    results_df: pd.DataFrame,
    backtest_df: pd.DataFrame,
    full_history_df: pd.DataFrame,
    model: StockTradingModel,
    min_samples: int
) -> tuple:
    """Print decisions for all strategies and return the best one.

    Returns:
        tuple: (best_strategy_name, best_action, best_confidence)
    """
    print_section(" ALL STRATEGY DECISIONS")

    decisions = {}

    for strategy in strategies:
        try:
            # Generate signals using full history for ML-based strategies
            if isinstance(strategy, (MLStrategy, HybridStrategy, RollingMLStrategy, RollingHybridStrategy)):
                signals = strategy.generate_signals(full_history_df)
                # Extract signals for backtest period
                signals = signals.iloc[-len(backtest_df):]
            else:
                signals = strategy.generate_signals(backtest_df)

            action, confidence = get_strategy_decision(signals)

            # Find this strategy's backtest result
            result_row = results_df[results_df['Strategy'] == strategy.name]
            if not result_row.empty:
                total_return = result_row['Total Return'].values[0]
                sharpe = result_row['Sharpe Ratio'].values[0]
            else:
                total_return = 'N/A'
                sharpe = 'N/A'

            decisions[strategy.name] = {
                'action': action,
                'confidence': confidence,
                'return': total_return,
                'sharpe': sharpe
            }

            print(f"\n  {strategy.name}:")
            print(f"    Decision  : {action}")
            print(f"    Confidence: {confidence:.2f}")
            print(f"    Return     : {total_return}")
            print(f"    Sharpe     : {sharpe}")

        except Exception as e:
            decisions[strategy.name] = {
                'action': 'ERROR',
                'confidence': 0.0,
                'return': 'N/A',
                'sharpe': 'N/A'
            }
            print(f"\n  {strategy.name}: ERROR - {e}")

    # Find best strategy based on return (parse percentage string)
    # If returns are equal, prefer ML-based strategies
    best_name = None
    best_return = -float('inf')
    best_is_ml = False

    for name, data in decisions.items():
        if data['return'] == 'N/A' or data['return'] == 'ERROR':
            continue
        try:
            # Parse percentage string like "12.34%"
            ret_str = data['return'].replace('%', '')
            ret_val = float(ret_str) / 100 if '%' in data['return'] else float(ret_str)
            # Check if this is an ML-based strategy
            is_ml = 'ML' in name or 'Hybrid' in name

            # Choose based on: higher return first, then ML preference
            if ret_val > best_return:
                best_return = ret_val
                best_name = name
                best_is_ml = is_ml
            elif ret_val == best_return and is_ml and not best_is_ml:
                # Equal return, prefer ML strategy
                best_return = ret_val
                best_name = name
                best_is_ml = is_ml
        except:
            continue

    return decisions, best_name


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

    # Split data for training and evaluation
    train_df = features_df.iloc[:-args.backtest_days]
    eval_df = features_df.iloc[-args.backtest_days:]

    print(f"\n  Training samples  : {len(train_df)}")
    print(f"  Evaluation samples: {len(eval_df)}")
    print(f"  Forward days      : {forward_days}")
    print(f"  Threshold         : {threshold:.3f} ({threshold*100:.2f}%)")
    print(f"  Composite labels  : {use_composite_labels}")

    try:
        train_metrics = model.train(
            train_df, forward_days=forward_days, threshold=threshold, eval_df=eval_df,
            use_composite_labels=use_composite_labels,
            trend_weight=trend_weight,
            momentum_weight=momentum_weight,
            market_weight=market_weight
        )
        print(f"\n  Training accuracy : {train_metrics['train_accuracy']:.2%}")
        print(f"  Features used     : {train_metrics['feature_count']}")
        print(f"  Label distribution:")
        print(f"    - Buy : {train_metrics['label_distribution']['buy']}")
        print(f"    - Sell: {train_metrics['label_distribution']['sell']}")
    except Exception as e:
        print(f"  Error training model: {e}")
        sys.exit(1)

    # Get prediction
    prediction_action = None
    prediction_confidence = None
    prediction_proba = None

    try:
        prediction_action, prediction_confidence = model.predict(features_df)
        prediction_proba = model.predict_proba(features_df)
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
        # Map stock_info.market to market key
        market_map = {
            'a_share': 'a_share',
            'hk': 'hk',
            'us': 'us',
        }
        market_key = market_map.get(stock_info.market, 'default')

        print(f"\n  Market config : {market_key}")

        strategies = get_market_strategies(
            model=model,
            market=market_key,
            min_samples=min_samples,
            require_bull_market_for_buy=True
        )

        engine = BacktestEngine(
            initial_cash=config.get('backtest.initial_cash', 100000),
            commission=config.get('backtest.commission', 0.001),
            slippage=config.get('backtest.slippage', 0.001)
        )

        results_df = engine.compare_strategies(backtest_df, strategies, full_history_df=features_df)
        print_backtest_comparison(results_df)

        # Print all strategy decisions and find the best one
        decisions, best_strategy_name = print_all_strategy_decisions(
            strategies, results_df, backtest_df, features_df, model, min_samples
        )

        # Trading decision (moved to end)
        # Use best performing strategy's decision instead of raw model prediction
        if best_strategy_name and best_strategy_name in decisions:
            best_decision = decisions[best_strategy_name]
            prediction_action_map = {'BUY': 1, 'HOLD': 0, 'SELL': -1}
            prediction_action = prediction_action_map.get(best_decision['action'], 0)
            prediction_confidence = best_decision['confidence']
            print_section(" FINAL RECOMMENDATION (BEST PERFORMING STRATEGY)")
            print(f"\n  Best Strategy : {best_strategy_name}")
            print(f"  Action        : {best_decision['action']}")
            print(f"  Confidence     : {best_decision['confidence']:.2f}")
            print(f"  Backtest Ret  : {best_decision['return']}")
        else:
            print_decision(prediction_action, prediction_confidence, prediction_proba)

    print("\n" + "="*60)
    print(" Decision process completed!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
