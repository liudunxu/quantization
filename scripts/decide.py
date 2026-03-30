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
    parser = argparse.ArgumentParser(description="Stock trading decision system")
    parser.add_argument(
        "--stock", required=True, help="Stock code (e.g., 000001.SZ, 0700.HK, AAPL)"
    )
    parser.add_argument(
        "--refresh", action="store_true", help="Force refresh cached data"
    )
    parser.add_argument(
        "--backtest-days", type=int, default=30, help="Number of days for backtest"
    )
    parser.add_argument(
        "--train-days", type=int, default=365, help="Number of days for training"
    )
    return parser.parse_args()


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print("=" * 60)


def print_decision(action: int, confidence: float, probabilities: dict) -> None:
    """Print trading decision."""
    action_map = {1: "BUY", 0: "HOLD", -1: "SELL"}
    action_str = action_map.get(action, "UNKNOWN")

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
    """Get the final decision from signals series with conviction analysis.

    Returns:
        tuple: (action_str, confidence)
            action_str: 'BUY', 'HOLD', or 'SELL'
            confidence: 0.0-1.0 based on signal conviction
    """
    if signals.empty:
        return "HOLD", 0.0

    last_signal = signals.iloc[-1]

    # Calculate conviction based on recent signals
    recent = signals.iloc[-5:]
    recent_sum = recent.sum()

    if last_signal == 1:
        # BUY signal - check conviction
        conviction = 0.6 + 0.1 * min(
            abs(recent_sum), 3
        )  # 0.6-0.9 based on recent history
        return "BUY", min(conviction, 1.0)
    elif last_signal == -1:
        # SELL signal - higher conviction for sell
        conviction = 0.7 + 0.1 * min(abs(recent_sum), 3)  # 0.7-1.0
        return "SELL", min(conviction, 1.0)
    else:
        # HOLD - check if recent signals show momentum
        if recent_sum > 2:
            return "HOLD", 0.4  # Slightly bullish momentum
        elif recent_sum < -2:
            return "HOLD", 0.4  # Slightly bearish momentum
        return "HOLD", 0.2


def calculate_suggested_lots(
    action: str,
    price: float,
    atr: float,
    cash: float = 100000,
    risk_pct: float = 0.01,
    atr_multiplier: float = 1.2,
) -> dict:
    """Calculate suggested trading lots based on ATR risk model.

    Args:
        action: 'BUY', 'SELL', or 'HOLD'
        price: Current stock price
        atr: Average True Range
        cash: Available cash (default 100000)
        risk_pct: Risk percentage per trade (default 1%)
        atr_multiplier: ATR multiplier for stop distance (default 1.2)

    Returns:
        dict with lots, shares, estimated_cost, stop_loss, take_profit, position_pct
    """
    if action == "HOLD" or price <= 0:
        return {
            "lots": 0,
            "shares": 0,
            "estimated_cost": 0,
            "stop_loss": 0,
            "take_profit": 0,
            "position_pct": 0,
        }

    # Risk amount
    risk_amount = cash * risk_pct

    # Stop distance based on ATR
    stop_distance = atr * atr_multiplier if atr > 0 else price * 0.04

    # Position size based on risk
    position_value = risk_amount / (stop_distance / price)
    shares = int(position_value / price / 100) * 100  # Round to lots

    # For expensive stocks (>100), allow smaller minimum (50 shares)
    if shares < 50 and price > 100:
        shares = 50
    elif shares < 100:
        shares = 100

    lots = shares / 100

    # Calculate position percentage
    estimated_cost = shares * price
    position_pct = estimated_cost / cash * 100 if cash > 0 else 0

    # Calculate stop loss and take profit prices
    if action == "BUY":
        stop_loss = price - stop_distance
        take_profit = price + stop_distance * 2
    else:  # SELL
        stop_loss = price + stop_distance
        take_profit = price - stop_distance * 2

    return {
        "lots": lots,
        "shares": shares,
        "estimated_cost": estimated_cost,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "position_pct": position_pct,
    }


def _parse_return(return_str: str) -> float:
    """Parse return string like '12.34%' to float 0.1234."""
    if return_str in ("N/A", "ERROR", None):
        return None
    try:
        ret_str = return_str.replace("%", "")
        return float(ret_str) / 100 if "%" in return_str else float(ret_str)
    except:
        return None


def _parse_sharpe(sharpe_str: str) -> float:
    """Parse Sharpe string to float."""
    if sharpe_str in ("N/A", "ERROR", None):
        return 0.0
    try:
        return float(sharpe_str)
    except:
        return 0.0


def _calculate_strategy_score(
    return_val: float, sharpe: float, win_rate: str, max_drawdown: str, is_ml: bool
) -> float:
    """Calculate comprehensive strategy score with risk adjustment.

    Score = return * 0.4 + sharpe * 0.3 + win_rate * 0.2 - drawdown_penalty * 0.1
    """
    if return_val is None:
        return -999.0

    # Parse win rate and max drawdown
    try:
        wr_str = win_rate.replace("%", "") if isinstance(win_rate, str) else "0"
        win_rate_val = float(wr_str) / 100 if "%" in win_rate else float(wr_str)
    except:
        win_rate_val = 0.0

    try:
        dd_str = max_drawdown.replace("%", "") if isinstance(max_drawdown, str) else "0"
        drawdown_val = float(dd_str) / 100 if "%" in max_drawdown else float(dd_str)
    except:
        drawdown_val = 0.0

    # Risk-adjusted return: avoid division by very small drawdown
    if drawdown_val < 0.01:
        drawdown_val = 0.01

    # Calculate score components
    return_component = return_val * 10  # Scale to comparable range
    sharpe_component = sharpe * 5  # Sharpe is usually 0-3
    win_rate_component = win_rate_val * 10  # Win rate 0-1 -> 0-10
    drawdown_penalty = drawdown_val * 5  # Larger drawdown = larger penalty

    score = (
        return_component * 0.4
        + sharpe_component * 0.3
        + win_rate_component * 0.2
        - drawdown_penalty * 0.1
    )

    # Prefer ML strategies slightly when scores are close
    if is_ml:
        score *= 1.05

    return score


def print_all_strategy_decisions(
    strategies: list,
    results_df: pd.DataFrame,
    backtest_df: pd.DataFrame,
    full_history_df: pd.DataFrame,
    model: StockTradingModel,
    min_samples: int,
) -> tuple:
    """Print decisions for all strategies and return the best one.

    Uses multi-indicator scoring and consensus mechanism.
    """
    print_section(" ALL STRATEGY DECISIONS")

    decisions = {}
    action_counts = {"BUY": 0, "HOLD": 0, "SELL": 0}  # For consensus

    for strategy in strategies:
        try:
            # Generate signals using full history for ML-based strategies
            if isinstance(
                strategy,
                (MLStrategy, HybridStrategy, RollingMLStrategy, RollingHybridStrategy),
            ):
                signals = strategy.generate_signals(full_history_df)
                # Extract signals for backtest period
                signals = signals.iloc[-len(backtest_df) :]
            else:
                signals = strategy.generate_signals(backtest_df)

            action, confidence = get_strategy_decision(signals)
            is_ml = isinstance(
                strategy,
                (MLStrategy, HybridStrategy, RollingMLStrategy, RollingHybridStrategy),
            )

            # Find this strategy's backtest result
            result_row = results_df[results_df["Strategy"] == strategy.name]
            if not result_row.empty:
                total_return = result_row["Total Return"].values[0]
                sharpe = _parse_sharpe(result_row["Sharpe Ratio"].values[0])
                win_rate = result_row["Win Rate"].values[0]
                max_drawdown = result_row["Max Drawdown"].values[0]
            else:
                total_return = "N/A"
                sharpe = 0.0
                win_rate = "0%"
                max_drawdown = "0%"

            # Calculate comprehensive score
            return_val = _parse_return(total_return)
            score = _calculate_strategy_score(
                return_val, sharpe, win_rate, max_drawdown, is_ml
            )

            decisions[strategy.name] = {
                "action": action,
                "confidence": confidence,
                "return": total_return,
                "sharpe": f"{sharpe:.2f}",
                "win_rate": win_rate,
                "max_drawdown": max_drawdown,
                "score": score,
                "is_ml": is_ml,
            }

            # Count actions for consensus
            if action != "ERROR":
                action_counts[action] = action_counts.get(action, 0) + 1

            print(f"\n  {strategy.name}:")
            print(f"    Decision  : {action}")
            print(f"    Confidence: {confidence:.2f}")
            print(f"    Return    : {total_return}")
            print(f"    Sharpe    : {sharpe:.2f}")
            print(f"    Score     : {score:.3f}")

        except Exception as e:
            decisions[strategy.name] = {
                "action": "ERROR",
                "confidence": 0.0,
                "return": "N/A",
                "sharpe": "N/A",
                "score": -999.0,
                "is_ml": False,
            }
            print(f"\n  {strategy.name}: ERROR - {e}")

    # Find best strategy based on comprehensive score
    best_name = None
    best_score = -float("inf")
    best_action = None
    best_confidence = 0.0

    # Find strategy with highest return
    best_return_name = None
    best_return_value = -float("inf")
    best_return_action = None

    for name, data in decisions.items():
        if data["score"] == -999.0:  # Skip ERROR
            continue

        score = data["score"]

        # Track best return
        return_val = _parse_return(data["return"])
        if return_val > best_return_value:
            best_return_value = return_val
            best_return_name = name
            best_return_action = data["action"]

        # Consensus bonus: if >= 3 strategies agree, boost their scores
        consensus = action_counts.get(data["action"], 0)
        if consensus >= 3:
            score *= 1.2  # 20% boost for consensus

        if score > best_score:
            best_score = score
            best_name = name
            best_action = data["action"]
            best_confidence = data["confidence"]

    # Apply consensus bonus to confidence
    if best_action and action_counts.get(best_action, 0) >= 3:
        best_confidence = min(best_confidence * 1.3, 1.0)

    return (
        decisions,
        best_name,
        best_action,
        best_confidence,
        action_counts,
        best_return_name,
        best_return_action,
        best_return_value,
    )


def print_feature_importance(model: StockTradingModel, top_n: int = 10) -> None:
    """Print top feature importances."""
    try:
        importance = model.get_feature_importance()
        print_section(f" TOP {top_n} FEATURE IMPORTANCE")
        print()
        top_features = importance.head(top_n)
        for _, row in top_features.iterrows():
            bar = "█" * int(row["importance"] / 2)
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
    cache = get_cache(config.get("data.cache_dir", "cache"))

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
            stock_code, days=args.train_days, force_refresh=args.refresh
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

    # Split data for training and evaluation
    train_df = features_df.iloc[: -args.backtest_days]
    eval_df = features_df.iloc[-args.backtest_days :]

    print(f"\n  Training samples  : {len(train_df)}")
    print(f"  Evaluation samples: {len(eval_df)}")
    print(f"  Forward days      : {forward_days}")
    print(f"  Threshold         : {threshold:.3f} ({threshold * 100:.2f}%)")
    print(f"  Composite labels  : {use_composite_labels}")

    try:
        train_metrics = model.train(
            train_df,
            forward_days=forward_days,
            threshold=threshold,
            eval_df=eval_df,
            use_composite_labels=use_composite_labels,
            trend_weight=trend_weight,
            momentum_weight=momentum_weight,
            market_weight=market_weight,
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
    backtest_df = features_df.iloc[-args.backtest_days :].copy()

    if len(backtest_df) < 10:
        print("\n  Not enough data for backtesting")
    else:
        # Map stock_info.market to market key
        market_map = {
            "a_share": "a_share",
            "hk": "hk",
            "hk_share": "hk",
            "us": "us",
            "us_share": "us",
        }
        market_key = market_map.get(stock_info.market, "default")

        print(f"\n  Market config : {market_key}")

        strategies = get_market_strategies(
            model=model,
            market=market_key,
            min_samples=min_samples,
            require_bull_market_for_buy=True,
        )

        engine = BacktestEngine(
            initial_cash=config.get("backtest.initial_cash", 100000),
            commission=config.get("backtest.commission", 0.001),
            slippage=config.get("backtest.slippage", 0.001),
        )

        results_df = engine.compare_strategies(
            backtest_df, strategies, full_history_df=features_df
        )
        print_backtest_comparison(results_df)

        # Print all strategy decisions and find the best one
        (
            decisions,
            best_strategy_name,
            best_action,
            best_confidence,
            action_counts,
            best_return_name,
            best_return_action,
            best_return_value,
        ) = print_all_strategy_decisions(
            strategies, results_df, backtest_df, features_df, model, min_samples
        )

        # Determine final action by majority vote
        final_action = (
            max(action_counts, key=action_counts.get) if action_counts else "HOLD"
        )
        final_confidence = best_confidence
        final_strategy = best_strategy_name

        # Trading decision (moved to end)
        if final_action != "HOLD":
            current_price = backtest_df["close"].iloc[-1]
            atr_value = (
                backtest_df["atr"].iloc[-1]
                if "atr" in backtest_df.columns
                else current_price * 0.02
            )
            initial_cash = config.get("backtest.initial_cash", 100000)
            suggested = calculate_suggested_lots(
                action=final_action,
                price=current_price,
                atr=atr_value,
                cash=initial_cash,
            )

            print_section(" FINAL RECOMMENDATION (MAJORITY VOTE)")
            print(f"\n  === Strategy Votes ===")
            print(f"  BUY  : {action_counts.get('BUY', 0)} votes")
            print(f"  HOLD : {action_counts.get('HOLD', 0)} votes")
            print(f"  SELL : {action_counts.get('SELL', 0)} votes")
            print(f"\n  Final Action: {final_action} (majority)")
            print(f"\n  Best Strategy by Score ({best_strategy_name}):")
            print(f"    Score     : {decisions[best_strategy_name]['score']:.3f}")
            print(f"    Return    : {decisions[best_strategy_name]['return']}")
            print(f"\n  Best Strategy by Return ({best_return_name}):")
            print(f"    Return    : {best_return_value:.2%}")
            print(f"    Decision  : {best_return_action}")
            print(f"\n  === Suggested Position ===")
            print(f"  Lots          : {suggested['lots']:.1f} 手")
            print(f"  Shares        : {suggested['shares']} 股")
            print(f"  Position      : {suggested['position_pct']:.1f}% of capital")
            print(f"  Est. Cost     : ${suggested['estimated_cost']:,.2f}")

            # Calculate stop loss and take profit percentages
            sl_pct = abs((current_price - suggested["stop_loss"]) / current_price * 100)
            tp_pct = abs(
                (suggested["take_profit"] - current_price) / current_price * 100
            )

            if final_action == "BUY":
                print(
                    f"  Stop Loss    : ${suggested['stop_loss']:.2f} (-{sl_pct:.1f}%)"
                )
                print(
                    f"  Take Profit   : ${suggested['take_profit']:.2f} (+{tp_pct:.1f}%)"
                )
            else:  # SELL
                print(
                    f"  Stop Loss    : ${suggested['stop_loss']:.2f} (+{sl_pct:.1f}%)"
                )
                print(
                    f"  Take Profit   : ${suggested['take_profit']:.2f} (-{tp_pct:.1f}%)"
                )
        else:
            print_section(" FINAL RECOMMENDATION")
            print(f"\n  Action       : HOLD (no clear consensus)")
            print(f"\n  === Strategy Votes ===")
            print(f"  BUY  : {action_counts.get('BUY', 0)} votes")
            print(f"  HOLD : {action_counts.get('HOLD', 0)} votes")
            print(f"  SELL : {action_counts.get('SELL', 0)} votes")
            print(f"\n  Best Strategy by Return ({best_return_name}):")
            print(f"    Return    : {best_return_value:.2%}")
            print(f"    Decision  : {best_return_action}")

            # Show suggested position based on best return strategy's decision
            if best_return_action != "HOLD":
                current_price = backtest_df["close"].iloc[-1]
                atr_value = (
                    backtest_df["atr"].iloc[-1]
                    if "atr" in backtest_df.columns
                    else current_price * 0.02
                )
                initial_cash = config.get("backtest.initial_cash", 100000)
                suggested = calculate_suggested_lots(
                    action=best_return_action,
                    price=current_price,
                    atr=atr_value,
                    cash=initial_cash,
                )
                print(f"\n  === Suggested Position (Based on Best Return) ===")
                print(f"  Lots          : {suggested['lots']:.1f} 手")
                print(f"  Shares        : {suggested['shares']} 股")
                print(f"  Position      : {suggested['position_pct']:.1f}% of capital")
                print(f"  Est. Cost     : ${suggested['estimated_cost']:,.2f}")

                # Calculate stop loss and take profit percentages
                sl_pct = abs(
                    (current_price - suggested["stop_loss"]) / current_price * 100
                )
                tp_pct = abs(
                    (suggested["take_profit"] - current_price) / current_price * 100
                )

                if best_return_action == "BUY":
                    print(
                        f"  Stop Loss    : ${suggested['stop_loss']:.2f} (-{sl_pct:.1f}%)"
                    )
                    print(
                        f"  Take Profit   : ${suggested['take_profit']:.2f} (+{tp_pct:.1f}%)"
                    )
                else:  # SELL
                    print(
                        f"  Stop Loss    : ${suggested['stop_loss']:.2f} (+{sl_pct:.1f}%)"
                    )
                    print(
                        f"  Take Profit   : ${suggested['take_profit']:.2f} (-{tp_pct:.1f}%)"
                    )

    print("\n" + "=" * 60)
    print(" Decision process completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
