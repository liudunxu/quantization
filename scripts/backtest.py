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
from src.features import get_feature_combinator, SentimentFeatures
from src.models import StockTradingModel
from src.backtest import (
    BacktestEngine,
    BacktestResult,
    MLStrategy,
    HybridStrategy,
    RollingMLStrategy,
    RollingHybridStrategy,
    get_market_strategies,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Backtest stock trading strategies")
    parser.add_argument("--stock", required=True, help="Stock code")
    parser.add_argument(
        "--backtest-days", type=int, default=30, help="Number of days for backtest"
    )
    parser.add_argument("--train-days", type=int, default=365, help="Training days")
    parser.add_argument(
        "--initial-cash", type=float, default=100000, help="Initial cash"
    )
    parser.add_argument("--output", help="Output file for results (CSV)")
    return parser.parse_args()


def print_result(result: BacktestResult, prefix: str = "  ") -> None:
    """Print a single backtest result."""
    print(f"\n{prefix}Strategy     : {result.strategy_name}")
    print(f"{prefix}Total Return  : {result.total_return:+.2%}")
    print(f"{prefix}Buy & Hold     : {result.buy_hold_return:+.2%}")
    print(
        f"{prefix}vs Benchmark   : {result.total_return - result.buy_hold_return:+.2%}"
    )
    print(f"{prefix}Sharpe Ratio   : {result.sharpe_ratio:.2f}")
    print(f"{prefix}Max Drawdown   : {result.max_drawdown:.2%}")
    print(f"{prefix}Win Rate       : {result.win_rate:.2%}")
    print(f"{prefix}Total Trades   : {result.total_trades}")

    if result.trades:
        print(
            f"{prefix}Last Trade     : {result.trades[-1].action} on {result.trades[-1].date.strftime('%Y-%m-%d')}"
        )


def print_sentiment_summary(df: pd.DataFrame, stock_code: str) -> None:
    """Print sentiment analysis summary for backtest."""
    print("\n" + "=" * 60)
    print(" SENTIMENT ANALYSIS SUMMARY")
    print("=" * 60)

    # Check if sentiment columns exist
    sentiment_cols = [col for col in df.columns if "sentiment" in col.lower()]
    if not sentiment_cols:
        print("\n  情绪数据不可用 (Sentiment data not available)")
        return

    latest = df.iloc[-1]

    # Basic sentiment info
    sentiment_score = latest.get("sentiment_score", 0)
    news_count = latest.get("news_count", 0)
    sentiment_ma3 = latest.get("sentiment_ma3", 0)
    sentiment_ma7 = latest.get("sentiment_ma7", 0)
    sentiment_trend = latest.get("sentiment_trend", 0)

    # Sentiment label
    if sentiment_score > 0.3:
        sentiment_label = "积极 (Positive)"
        sentiment_emoji = "🟢"
    elif sentiment_score < -0.3:
        sentiment_label = "消极 (Negative)"
        sentiment_emoji = "🔴"
    else:
        sentiment_label = "中性 (Neutral)"
        sentiment_emoji = "⚪"

    print(f"\n  股票代码 (Stock Code)  : {stock_code}")
    print(f"  情绪分数 (Sentiment)  : {sentiment_score:.3f} {sentiment_emoji}")
    print(f"  情绪标签 (Label)      : {sentiment_label}")
    print(f"  新闻数量 (News Count) : {news_count:.0f}")
    print(f"  3日均线 (MA3)         : {sentiment_ma3:.3f}")
    print(f"  7日均线 (MA7)         : {sentiment_ma7:.3f}")
    print(f"  情绪趋势 (Trend)      : {sentiment_trend:+.3f}")

    # News activity level
    if news_count > 10:
        news_activity = "高 (High)"
    elif news_count > 3:
        news_activity = "中 (Moderate)"
    else:
        news_activity = "低 (Low)"
    print(f"  新闻活跃度 (Activity) : {news_activity}")

    # Extreme sentiment warning
    if sentiment_score > 0.5:
        print(f"\n  ⚠️  市场情绪极度乐观，注意风险")
        print(f"     (Market sentiment extremely bullish, be cautious)")
    elif sentiment_score < -0.5:
        print(f"\n  ⚠️  市场情绪极度悲观，可能存在机会")
        print(f"     (Market sentiment extremely bearish, potential opportunity)")

    # Sentiment volatility
    sentiment_std = latest.get("sentiment_std7", 0)
    if sentiment_std > 0.3:
        print(f"\n  📊 情绪波动较大 (High sentiment volatility: {sentiment_std:.3f})")
    elif sentiment_std < 0.1:
        print(f"\n  📊 情绪波动较小 (Low sentiment volatility: {sentiment_std:.3f})")

    # Recent sentiment trend
    if len(df) >= 5:
        recent_sentiment = df["sentiment_score"].tail(5)
        if recent_sentiment.mean() > 0.2:
            print(f"  📈 近期情绪偏积极 (Recent sentiment: bullish)")
        elif recent_sentiment.mean() < -0.2:
            print(f"  📉 近期情绪偏消极 (Recent sentiment: bearish)")
        else:
            print(f"  ➡️  近期情绪中性 (Recent sentiment: neutral)")

    # Trading suggestion based on sentiment
    print(f"\n  === 基于情绪的交易建议 (Sentiment-based Suggestion) ===")
    if sentiment_score > 0.4 and sentiment_trend > 0:
        print(f"  情绪积极且上升，可考虑买入 (Positive and rising sentiment)")
    elif sentiment_score < -0.4 and sentiment_trend < 0:
        print(f"  情绪消极且下降，注意风险 (Negative and declining sentiment)")
    elif sentiment_score > 0.2:
        print(f"  情绪略偏积极，谨慎乐观 (Slightly positive, cautious optimism)")
    elif sentiment_score < -0.2:
        print(f"  情绪略偏消极，谨慎观望 (Slightly negative, cautious观望)")
    else:
        print(f"  情绪中性，建议观望 (Neutral sentiment, wait and see)")


def main():
    args = parse_args()
    stock_code = args.stock

    print("=" * 60)
    print(" STOCK TRADING BACKTEST")
    print("=" * 60)
    print(f"\n  Stock Code     : {stock_code}")
    print(f"  Backtest Days  : {args.backtest_days}")
    print(f"  Training Days  : {args.train_days}")
    print(f"  Initial Cash   : ${args.initial_cash:,.2f}")

    # Resolve stock info for market-specific strategies
    try:
        stock_info = StockInfoResolver.resolve(stock_code)
        print(f"  Market         : {stock_info.market}")
    except ValueError as e:
        print(f"  Warning: Could not resolve market: {e}")
        stock_info = None

    # Get data
    config = get_config()
    cache = get_cache(config.get("data.cache_dir", "cache"))
    combinator = get_feature_combinator(cache)

    print("\n  Fetching data...")
    try:
        features_df = combinator.get_combined_features(
            stock_code, days=args.train_days, force_refresh=False
        )
        if features_df.empty:
            print("  Error: Could not fetch data")
            sys.exit(1)
    except Exception as e:
        print(f"  Error fetching data: {e}")
        sys.exit(1)

    # Split data
    backtest_df = features_df.iloc[-args.backtest_days :].copy()
    train_df = features_df.iloc[: -args.backtest_days]

    print(f"  Backtest samples: {len(backtest_df)}")

    if len(backtest_df) < 10:
        print("  Error: Not enough data for backtest")
        sys.exit(1)

    # Train ML model
    print("\n  Training ML model...")
    model = StockTradingModel()

    # Get aggressive training params from config
    model_config = config.get_section("model")
    training_config = model_config.get("training", {})
    forward_days = training_config.get("forward_days", 5)
    threshold = training_config.get("threshold", 0.01)
    min_samples = model_config.get("strategy", {}).get("min_samples", 20)
    confidence_threshold = model_config.get("strategy", {}).get(
        "confidence_threshold", 0.25
    )

    # Composite label parameters
    use_composite_labels = training_config.get("use_composite_labels", True)
    trend_weight = training_config.get("trend_weight", 0.30)
    momentum_weight = training_config.get("momentum_weight", 0.30)
    market_weight = training_config.get("market_weight", 0.20)

    try:
        model.train(
            train_df,
            forward_days=forward_days,
            threshold=threshold,
            use_composite_labels=use_composite_labels,
            trend_weight=trend_weight,
            momentum_weight=momentum_weight,
            market_weight=market_weight,
        )
        print(
            f"  Model trained (forward_days={forward_days}, threshold={threshold:.3f})"
        )
    except Exception as e:
        print(f"  Error training model: {e}")
        sys.exit(1)

    # Run backtests
    print("\n" + "=" * 60)
    print(" BACKTEST RESULTS")
    print("=" * 60)

    engine = BacktestEngine(
        initial_cash=args.initial_cash,
        commission=config.get("backtest.commission", 0.001),
        slippage=config.get("backtest.slippage", 0.001),
        stop_loss=0.05,  # 5% stop loss
        take_profit=0.15,  # 15% take profit
    )

    # Map stock_info.market to market key
    if stock_info:
        market_map = {
            "a_share": "a_share",
            "hk": "hk",
            "hk_share": "hk",
            "us": "us",
            "us_share": "us",
        }
        market_key = market_map.get(stock_info.market, "default")
    else:
        market_key = "default"

    strategies = get_market_strategies(
        model=model,
        market=market_key,
        min_samples=min_samples,
        require_bull_market_for_buy=True,
    )

    results = []
    for strategy in strategies:
        # For ML-based strategies, use full history to generate signals
        if isinstance(
            strategy,
            (MLStrategy, HybridStrategy, RollingMLStrategy, RollingHybridStrategy),
        ):
            signals = strategy.generate_signals(features_df)
            signals = signals.iloc[-len(backtest_df) :]
            result = engine.run(backtest_df, strategy, precomputed_signals=signals)
        else:
            result = engine.run(backtest_df, strategy)
        results.append(result)
        print_result(result)

    # Summary
    print("\n" + "=" * 60)
    print(" STRATEGY COMPARISON SUMMARY")
    print("=" * 60)

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

    # Print sentiment analysis summary
    print_sentiment_summary(backtest_df, stock_code)

    # Save results if requested
    if args.output:
        output_data = []
        for result in results:
            output_data.append(
                {
                    "strategy": result.strategy_name,
                    "total_return": result.total_return,
                    "buy_hold_return": result.buy_hold_return,
                    "sharpe_ratio": result.sharpe_ratio,
                    "max_drawdown": result.max_drawdown,
                    "win_rate": result.win_rate,
                    "total_trades": result.total_trades,
                }
            )

        df = pd.DataFrame(output_data)
        df.to_csv(args.output, index=False)
        print(f"\n  Results saved to: {args.output}")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
