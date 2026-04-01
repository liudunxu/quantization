#!/usr/bin/env python
"""Strategy parameter exploration script.

Search for optimal rule-based strategy parameters for a specific stock.
Results are saved to SQLite for use in decide.py, backtest.py, and predict.py.

Usage:
    # 决策场景（默认）- 优化 decide.py 相关策略参数
    python scripts/explore_params.py --stock 000001.SZ
    python scripts/explore_params.py --stock 000001.SZ --strategies ma_golden_cross box_oscillation

    # 预测场景 - 优化 predict.py 相关参数
    python scripts/explore_params.py --stock 000001.SZ --scenario prediction
    python scripts/explore_params.py --stock 000001.SZ --scenario prediction --metric accuracy

    # 通用参数
    python scripts/explore_params.py --stock 000001.SZ --train-days 200 --backtest-days 60
    python scripts/explore_params.py --stock 000001.SZ --search-method grid --param-samples 5
"""

import sys
import argparse
import logging
import itertools
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import get_cache, get_config, get_param_manager, StockInfoResolver
from src.features import get_feature_combinator
from src.backtest import BacktestEngine
from src.backtest.rule_strategies import (
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/explore_params.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


# Parameter search spaces for each strategy
PARAM_SPACES = {
    "ma_golden_cross": {
        "fast_ma": {"type": "int", "min": 3, "max": 10, "step": 1},
        "slow_ma": {"type": "int", "min": 10, "max": 30, "step": 5},
        "volume_ratio": {"type": "float", "min": 0.8, "max": 2.0, "step": 0.2},
    },
    "bull_trend": {
        "ma5_period": {"type": "int", "min": 3, "max": 10, "step": 2},
        "ma10_period": {"type": "int", "min": 8, "max": 15, "step": 2},
        "ma20_period": {"type": "int", "min": 15, "max": 30, "step": 5},
    },
    "shrink_pullback": {
        "lookback": {"type": "int", "min": 3, "max": 10, "step": 2},
        "volume_shrink": {"type": "float", "min": 0.5, "max": 0.9, "step": 0.1},
    },
    "bottom_volume": {
        "drop_threshold": {"type": "float", "min": 0.05, "max": 0.20, "step": 0.05},
        "volume_multiplier": {"type": "float", "min": 1.5, "max": 3.0, "step": 0.5},
    },
    "box_oscillation": {
        "lookback": {"type": "int", "min": 20, "max": 60, "step": 10},
        "support_margin": {"type": "float", "min": 0.02, "max": 0.08, "step": 0.02},
        "resistance_margin": {"type": "float", "min": 0.02, "max": 0.08, "step": 0.02},
    },
    "volume_breakout": {
        "lookback": {"type": "int", "min": 10, "max": 30, "step": 5},
        "volume_multiplier": {"type": "float", "min": 1.2, "max": 2.5, "step": 0.3},
    },
    "macd_divergence": {
        "lookback": {"type": "int", "min": 10, "max": 30, "step": 5},
    },
    "emotion_cycle": {
        "volume_shrink": {"type": "float", "min": 0.3, "max": 0.7, "step": 0.1},
        "rsi_oversold": {"type": "int", "min": 20, "max": 40, "step": 5},
        "rsi_overbought": {"type": "int", "min": 60, "max": 80, "step": 5},
    },
}


# Strategy class mapping
STRATEGY_CLASSES = {
    "ma_golden_cross": MAGoldenCrossStrategy,
    "bull_trend": BullTrendStrategy,
    "shrink_pullback": ShrinkPullbackStrategy,
    "bottom_volume": BottomVolumeStrategy,
    "box_oscillation": BoxOscillationStrategy,
    "volume_breakout": VolumeBreakoutStrategy,
    "macd_divergence": MACDDivergenceStrategy,
    "emotion_cycle": EmotionCycleStrategy,
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Strategy parameter exploration for optimal rule-based strategy parameters"
    )
    parser.add_argument(
        "--stock",
        type=str,
        required=True,
        help="Stock code (e.g., 000001.SZ, 0700.HK, AAPL)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["decision", "prediction"],
        default="decision",
        help="Scenario type: 'decision' for decide.py, 'prediction' for predict.py (default: decision)",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        default=None,
        help="Strategies to explore (default: all rule-based strategies)",
    )
    parser.add_argument(
        "--train-days",
        type=int,
        default=200,
        help="Training data days (default: 200)",
    )
    parser.add_argument(
        "--backtest-days",
        type=int,
        default=60,
        help="Backtest period days (default: 60)",
    )
    parser.add_argument(
        "--search-method",
        type=str,
        choices=["grid", "random"],
        default="grid",
        help="Search method: grid search or random search (default: grid)",
    )
    parser.add_argument(
        "--param-samples",
        type=int,
        default=5,
        help="Number of parameter samples per dimension for grid search (default: 5)",
    )
    parser.add_argument(
        "--random-samples",
        type=int,
        default=50,
        help="Number of random samples for random search (default: 50)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["total_return", "sharpe_ratio", "win_rate", "composite"],
        default="composite",
        help="Optimization metric (default: composite)",
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=100000,
        help="Initial cash for backtesting (default: 100000)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode: don't save results to database",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    return parser.parse_args()


def generate_param_grid(
    param_space: Dict[str, Dict], max_samples: int = 5
) -> List[Dict[str, Any]]:
    """Generate parameter grid from parameter space.

    Args:
        param_space: Parameter space definition
        max_samples: Maximum samples per dimension

    Returns:
        List of parameter combinations
    """
    param_values = {}

    for param_name, param_def in param_space.items():
        if param_def["type"] == "int":
            values = list(
                range(param_def["min"], param_def["max"] + 1, param_def["step"])
            )
            # Limit to max_samples
            if len(values) > max_samples:
                indices = np.linspace(0, len(values) - 1, max_samples, dtype=int)
                values = [values[i] for i in indices]
            param_values[param_name] = values

        elif param_def["type"] == "float":
            values = np.arange(
                param_def["min"],
                param_def["max"] + param_def["step"] / 2,
                param_def["step"],
            )
            # Round to 3 decimal places
            values = np.round(values, 3).tolist()
            # Limit to max_samples
            if len(values) > max_samples:
                indices = np.linspace(0, len(values) - 1, max_samples, dtype=int)
                values = [values[i] for i in indices]
            param_values[param_name] = values

    # Generate all combinations
    keys = list(param_values.keys())
    values = list(param_values.values())
    combinations = list(itertools.product(*values))

    return [dict(zip(keys, combo)) for combo in combinations]


def generate_random_params(
    param_space: Dict[str, Dict], n_samples: int = 50
) -> List[Dict[str, Any]]:
    """Generate random parameter samples from parameter space.

    Args:
        param_space: Parameter space definition
        n_samples: Number of samples to generate

    Returns:
        List of parameter combinations
    """
    samples = []

    for _ in range(n_samples):
        sample = {}
        for param_name, param_def in param_space.items():
            if param_def["type"] == "int":
                sample[param_name] = np.random.randint(
                    param_def["min"], param_def["max"] + 1
                )
            elif param_def["type"] == "float":
                sample[param_name] = np.round(
                    np.random.uniform(param_def["min"], param_def["max"]), 3
                )
        samples.append(sample)

    return samples


def calculate_composite_score(result) -> float:
    """Calculate composite score from backtest result.

    Args:
        result: BacktestResult object

    Returns:
        Composite score (higher is better)
    """
    # Normalize each metric to [0, 1] range
    # Return: assume -50% to +50% range
    return_score = (result.total_return + 0.5) / 1.0

    # Sharpe: assume -3 to +3 range
    sharpe_score = (result.sharpe_ratio + 3) / 6.0

    # Win rate: already [0, 1]
    win_rate_score = result.win_rate

    # Max drawdown: lower is better, assume 0 to -30% range
    drawdown_score = 1.0 + result.max_drawdown / 0.3

    # Composite: weighted average
    composite = (
        0.35 * max(0, min(1, return_score))
        + 0.30 * max(0, min(1, sharpe_score))
        + 0.20 * max(0, min(1, win_rate_score))
        + 0.15 * max(0, min(1, drawdown_score))
    )

    return composite


def evaluate_strategy(
    strategy_class,
    params: Dict[str, Any],
    df: pd.DataFrame,
    metric: str = "composite",
) -> Tuple[float, Dict[str, Any]]:
    """Evaluate a strategy with given parameters.

    Args:
        strategy_class: Strategy class
        params: Strategy parameters
        df: Backtest data
        metric: Optimization metric

    Returns:
        Tuple of (score, result_metrics)
    """
    try:
        strategy = strategy_class(**params)
        engine = BacktestEngine()
        result = engine.run(df, strategy)

        # Calculate score based on metric
        if metric == "total_return":
            score = result.total_return
        elif metric == "sharpe_ratio":
            score = result.sharpe_ratio
        elif metric == "win_rate":
            score = result.win_rate
        else:  # composite
            score = calculate_composite_score(result)

        metrics = {
            "total_return": result.total_return,
            "sharpe_ratio": result.sharpe_ratio,
            "win_rate": result.win_rate,
            "max_drawdown": result.max_drawdown,
            "total_trades": result.total_trades,
            "composite_score": calculate_composite_score(result),
        }

        return score, metrics

    except Exception as e:
        logger.warning(f"Error evaluating {strategy_class.__name__} with {params}: {e}")
        return -999.0, {}


def explore_strategy(
    strategy_name: str,
    stock_code: str,
    df: pd.DataFrame,
    search_method: str = "grid",
    param_samples: int = 5,
    random_samples: int = 50,
    metric: str = "composite",
    verbose: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    """Explore parameters for a single strategy.

    Args:
        strategy_name: Strategy name
        stock_code: Stock code
        df: Backtest data
        search_method: Search method ('grid' or 'random')
        param_samples: Number of samples per dimension for grid search
        random_samples: Number of random samples
        metric: Optimization metric
        verbose: Verbose output

    Returns:
        Tuple of (best_params, best_metrics, all_results)
    """
    if strategy_name not in PARAM_SPACES:
        logger.warning(f"No parameter space defined for {strategy_name}")
        return {}, {}, []

    if strategy_name not in STRATEGY_CLASSES:
        logger.warning(f"No strategy class defined for {strategy_name}")
        return {}, {}, []

    param_space = PARAM_SPACES[strategy_name]
    strategy_class = STRATEGY_CLASSES[strategy_name]

    # Generate parameter combinations
    if search_method == "grid":
        param_combinations = generate_param_grid(param_space, param_samples)
    else:
        param_combinations = generate_random_params(param_space, random_samples)

    logger.info(
        f"Exploring {strategy_name} for {stock_code}: {len(param_combinations)} combinations"
    )

    # Evaluate each parameter combination
    best_score = -float("inf")
    best_params = {}
    best_metrics = {}
    all_results = []

    for i, params in enumerate(param_combinations):
        score, metrics = evaluate_strategy(strategy_class, params, df, metric)

        result_record = {
            "params": params,
            "score": score,
            "metrics": metrics,
        }
        all_results.append(result_record)

        if score > best_score:
            best_score = score
            best_params = params
            best_metrics = metrics

        if verbose and (i + 1) % 10 == 0:
            logger.info(f"  Evaluated {i + 1}/{len(param_combinations)} combinations")

    logger.info(
        f"  Best {strategy_name} params: {best_params} "
        f"(score={best_score:.4f}, return={best_metrics.get('total_return', 0):.2%})"
    )

    return best_params, best_metrics, all_results


def main():
    """Main function."""
    args = parse_args()

    # Create logs directory if not exists
    Path("logs").mkdir(exist_ok=True)

    print("=" * 60)
    print(" STRATEGY PARAMETER EXPLORATION")
    print("=" * 60)

    stock_code = args.stock
    stock_info = StockInfoResolver.resolve(stock_code)
    # Convert market format: hk_share -> hk, a_share -> a_share, us_share -> us
    market = stock_info.market.replace("_share", "")
    exchange = stock_info.exchange

    print(f"\n  Stock Code  : {stock_code}")
    print(f"  Market      : {market}")
    print(f"  Exchange    : {exchange}")
    print(f"  Train Days  : {args.train_days}")
    print(f"  Backtest Days: {args.backtest_days}")
    print(f"  Search Method: {args.search_method}")
    print(f"  Metric      : {args.metric}")
    print(f"  Dry Run     : {args.dry_run}")

    # Get configuration
    config = get_config()
    cache = get_cache(config.get("data.cache_dir", "cache"))

    # Fetch and prepare data
    print(f"\n{'=' * 60}")
    print(" FETCHING DATA")
    print("=" * 60)

    total_days = args.train_days + args.backtest_days
    combinator = get_feature_combinator(cache)
    features_df = combinator.get_combined_features(
        stock_code=stock_code,
        days=total_days,
        force_refresh=False,
    )

    if features_df.empty or len(features_df) < 30:
        print(f"  Error: Insufficient data for {stock_code}")
        return

    print(f"  Total samples: {len(features_df)}")

    # Split into train and backtest periods
    train_df = (
        features_df.iloc[: -args.backtest_days]
        if len(features_df) > args.backtest_days
        else features_df
    )
    backtest_df = features_df

    print(f"  Backtest period: {len(backtest_df)} days")

    # Determine strategies to explore
    strategies_to_explore = (
        args.strategies if args.strategies else list(PARAM_SPACES.keys())
    )

    print(f"\n{'=' * 60}")
    print(" EXPLORING STRATEGIES")
    print("=" * 60)

    all_best_params = {}
    all_best_metrics = {}

    # Check if prediction scenario
    if args.scenario == "prediction":
        print("\n  === PREDICTION SCENARIO ===")
        print("  Optimizing parameters for predict.py")

        # For prediction scenario, we optimize ML model parameters
        from src.pipelines import ModelPipeline

        model_pipeline = ModelPipeline(config)

        train_df = (
            features_df.iloc[: -args.backtest_days]
            if len(features_df) > args.backtest_days
            else features_df
        )
        eval_df = (
            features_df.iloc[-args.backtest_days :]
            if len(features_df) > args.backtest_days
            else features_df
        )

        # ========== 1. 测试不同 forward_days ==========
        print("\n  [1/4] Testing different forward_days...")
        forward_days_list = [1, 2, 3, 5]
        thresholds = [0.003, 0.005, 0.007, 0.01]

        best_accuracy = 0
        best_f1 = 0
        best_forward_days = 1
        best_threshold = 0.005
        best_metrics = {}

        for fwd_days in forward_days_list:
            for threshold in thresholds:
                try:
                    model = model_pipeline.train(
                        train_df, forward_days=fwd_days, threshold=threshold
                    )
                    metrics = model_pipeline.evaluate_metrics(
                        model, eval_df, threshold=threshold
                    )
                    # 综合评分: accuracy * 0.4 + f1_up * 0.3 + f1_down * 0.3
                    composite = (
                        metrics["accuracy"] * 0.4
                        + metrics["f1_up"] * 0.3
                        + metrics["f1_down"] * 0.3
                    )
                    if composite > best_accuracy + best_f1:
                        best_accuracy = metrics["accuracy"]
                        best_f1 = composite
                        best_forward_days = fwd_days
                        best_threshold = threshold
                        best_metrics = metrics
                except Exception as e:
                    logger.debug(f"Failed: fwd={fwd_days}, thr={threshold}: {e}")

        print(
            f"    Best: forward_days={best_forward_days}, threshold={best_threshold:.3f}"
        )
        print(f"    Accuracy={best_accuracy:.1%}, Composite={best_f1:.1%}")

        # ========== 2. 测试不同模型参数 ==========
        print("\n  [2/4] Testing different model parameters...")
        model_params_list = [
            {"iterations": 200, "depth": 3, "learning_rate": 0.05},
            {"iterations": 300, "depth": 4, "learning_rate": 0.03},
            {"iterations": 500, "depth": 5, "learning_rate": 0.02},
            {"iterations": 300, "depth": 6, "learning_rate": 0.03},
        ]

        best_model_params = {}
        best_model_score = 0

        for mp in model_params_list:
            try:
                from src.models import StockTradingModel

                model = StockTradingModel()
                model.train(
                    train_df,
                    forward_days=best_forward_days,
                    threshold=best_threshold,
                    **mp,
                )
                metrics = model_pipeline.evaluate_metrics(
                    model, eval_df, threshold=best_threshold
                )
                score = (
                    metrics["accuracy"] * 0.4
                    + metrics["f1_up"] * 0.3
                    + metrics["f1_down"] * 0.3
                )
                print(f"    {mp}: score={score:.1%}")
                if score > best_model_score:
                    best_model_score = score
                    best_model_params = mp
                    best_metrics = metrics
            except Exception as e:
                logger.debug(f"Failed model params {mp}: {e}")

        print(f"    Best model params: {best_model_params}")

        # ========== 3. 测试更多权重组合 ==========
        print("\n  [3/4] Testing different signal weights...")
        weight_combinations = [
            {
                "ml_weight": 0.50,
                "technical_weight": 0.30,
                "momentum_weight": 0.10,
                "trend_weight": 0.07,
                "alpha_weight": 0.03,
            },
            {
                "ml_weight": 0.40,
                "technical_weight": 0.30,
                "momentum_weight": 0.15,
                "trend_weight": 0.10,
                "alpha_weight": 0.05,
            },
            {
                "ml_weight": 0.35,
                "technical_weight": 0.35,
                "momentum_weight": 0.15,
                "trend_weight": 0.10,
                "alpha_weight": 0.05,
            },
            {
                "ml_weight": 0.30,
                "technical_weight": 0.40,
                "momentum_weight": 0.15,
                "trend_weight": 0.10,
                "alpha_weight": 0.05,
            },
            {
                "ml_weight": 0.45,
                "technical_weight": 0.25,
                "momentum_weight": 0.15,
                "trend_weight": 0.10,
                "alpha_weight": 0.05,
            },
            {
                "ml_weight": 0.60,
                "technical_weight": 0.20,
                "momentum_weight": 0.10,
                "trend_weight": 0.05,
                "alpha_weight": 0.05,
            },
        ]

        best_weights = weight_combinations[0]
        best_weight_score = 0

        for weights in weight_combinations:
            # 简单评分: ML权重越高，依赖模型准确性
            score = best_accuracy * weights["ml_weight"] + 0.5 * (
                1 - weights["ml_weight"]
            )
            print(f"    {weights}: estimated_score={score:.1%}")
            if score > best_weight_score:
                best_weight_score = score
                best_weights = weights

        # ========== 4. 测试基于规则的策略信号叠加 ==========
        print("\n  [4/4] Testing rule-based strategy stacking...")
        strategy_results = {}

        for strategy_name in [
            "ma_golden_cross",
            "bull_trend",
            "volume_breakout",
            "macd_divergence",
        ]:
            if strategy_name in STRATEGY_CLASSES:
                try:
                    strategy_class = STRATEGY_CLASSES[strategy_name]
                    strategy = strategy_class()
                    from src.backtest.engine import BacktestEngine

                    engine = BacktestEngine()
                    result = engine.run(eval_df, strategy)
                    strategy_results[strategy_name] = {
                        "return": result.total_return,
                        "sharpe": result.sharpe_ratio,
                        "win_rate": result.win_rate,
                    }
                    print(
                        f"    {strategy_name}: return={result.total_return:.2%}, sharpe={result.sharpe_ratio:.2f}"
                    )
                except Exception as e:
                    logger.debug(f"Failed strategy {strategy_name}: {e}")

        # 选择表现最好的策略用于叠加
        best_strategies = []
        if strategy_results:
            best_strategies = sorted(
                strategy_results.items(),
                key=lambda x: x[1].get("return", 0) + x[1].get("sharpe", 0),
                reverse=True,
            )[:3]
            print(f"    Top strategies for stacking: {[s[0] for s in best_strategies]}")

        # Save prediction parameters
        prediction_params = {
            "threshold": best_threshold,
            "forward_days": best_forward_days,
            **best_weights,
            **best_model_params,
            "stacking_strategies": [s[0] for s in best_strategies]
            if strategy_results
            else [],
        }

        all_best_params["prediction"] = prediction_params
        all_best_metrics["prediction"] = {
            "accuracy": best_accuracy,
            "precision_up": best_metrics.get("precision_up", 0),
            "recall_up": best_metrics.get("recall_up", 0),
            "f1_up": best_metrics.get("f1_up", 0),
            "precision_down": best_metrics.get("precision_down", 0),
            "recall_down": best_metrics.get("recall_down", 0),
            "f1_down": best_metrics.get("f1_down", 0),
        }

        print(f"\n  Best prediction parameters:")
        print(f"    forward_days: {best_forward_days}")
        print(f"    threshold: {best_threshold}")
        print(f"    model_params: {best_model_params}")
        print(f"    weights: {best_weights}")
        print(f"    accuracy: {best_accuracy:.1%}")
        print(f"    f1_up: {best_metrics.get('f1_up', 0):.1%}")
        print(f"    f1_down: {best_metrics.get('f1_down', 0):.1%}")

    else:
        # Decision scenario - original logic
        for strategy_name in strategies_to_explore:
            if strategy_name not in PARAM_SPACES:
                print(f"\n  Skipping {strategy_name}: no parameter space defined")
                continue

            print(f"\n  === {strategy_name.upper()} ===")

            best_params, best_metrics, all_results = explore_strategy(
                strategy_name=strategy_name,
                stock_code=stock_code,
                df=backtest_df,
                search_method=args.search_method,
                param_samples=args.param_samples,
                random_samples=args.random_samples,
                metric=args.metric,
                verbose=args.verbose,
            )

            if best_params:
                all_best_params[strategy_name] = best_params
                all_best_metrics[strategy_name] = best_metrics

                print(f"    Best Params:")
                for k, v in best_params.items():
                    print(f"      {k}: {v}")
                print(f"    Performance:")
                print(f"      Total Return : {best_metrics.get('total_return', 0):.2%}")
                print(f"      Sharpe Ratio : {best_metrics.get('sharpe_ratio', 0):.2f}")
                print(f"      Win Rate     : {best_metrics.get('win_rate', 0):.2%}")
                print(f"      Max Drawdown : {best_metrics.get('max_drawdown', 0):.2%}")
                print(
                    f"      Composite    : {best_metrics.get('composite_score', 0):.4f}"
                )

    # Save results to database
    if not args.dry_run and all_best_params:
        print(f"\n{'=' * 60}")
        print(" SAVING TO DATABASE")
        print("=" * 60)

        param_manager = get_param_manager()

        for strategy_name, params in all_best_params.items():
            description = (
                f"Optimized for {stock_code} on {datetime.now().strftime('%Y-%m-%d')}. "
                f"Return={all_best_metrics[strategy_name].get('total_return', 0):.2%}, "
                f"Sharpe={all_best_metrics[strategy_name].get('sharpe_ratio', 0):.2f}"
            )

            param_manager.set_strategy_params(
                strategy_name=strategy_name,
                params=params,
                market=market,
                stock_code=stock_code,
                description=description,
            )

            print(f"  Saved {strategy_name} params for {stock_code}")

        print(f"\n  Done! Parameters saved to cache/strategy_params.db")
    elif args.dry_run:
        print(f"\n{'=' * 60}")
        print(" DRY RUN - Results NOT saved to database")
        print("=" * 60)

    # Summary
    print(f"\n{'=' * 60}")
    print(" SUMMARY")
    print("=" * 60)

    if all_best_metrics:
        if args.scenario == "prediction":
            # Prediction scenario - show accuracy
            print(f"\n  {'Scenario':<25} {'Accuracy':>10} {'Threshold':>10}")
            print(f"  {'-' * 25} {'-' * 10} {'-' * 10}")

            for name, metrics in all_best_metrics.items():
                accuracy = metrics.get("accuracy", 0)
                threshold = all_best_params.get(name, {}).get("threshold", 0)
                print(f"  {name:<25} {accuracy:>9.1%} {threshold:>10.3f}")
        else:
            # Decision scenario - show return, sharpe, win rate
            print(
                f"\n  {'Strategy':<25} {'Return':>10} {'Sharpe':>10} {'Win Rate':>10}"
            )
            print(f"  {'-' * 25} {'-' * 10} {'-' * 10} {'-' * 10}")

            for strategy_name, metrics in sorted(
                all_best_metrics.items(),
                key=lambda x: x[1].get("composite_score", 0),
                reverse=True,
            ):
                print(
                    f"  {strategy_name:<25} "
                    f"{metrics.get('total_return', 0):>9.2%} "
                    f"{metrics.get('sharpe_ratio', 0):>10.2f} "
                    f"{metrics.get('win_rate', 0):>9.2%}"
                )

    print(f"\n{'=' * 60}")
    print(" EXPLORATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
