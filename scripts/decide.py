#!/usr/bin/env python3
"""Stock trading decision script.

Usage:
    python scripts/decide.py --stock 000001.SZ
    python scripts/decide.py --stock 0700.HK
    python scripts/decide.py --stock AAPL
    python scripts/decide.py --stock 000001.SZ --exclude-dates
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

from src.utils import (
    get_cache,
    get_config,
    StockInfoResolver,
    get_important_dates_manager,
)
from src.features import get_feature_combinator, SentimentFeatures
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
    parser.add_argument(
        "--exclude-dates",
        action="store_true",
        help="Exclude extreme volatility dates to reduce outlier impact",
    )
    parser.add_argument(
        "--exclude-threshold",
        type=float,
        default=2.0,
        help="Extreme volatility detection threshold (std multiplier, default: 2.0)",
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


def _parse_return(return_str: str) -> Optional[float]:
    """Parse return string like '12.34%' to float 0.1234."""
    if return_str in ("N/A", "ERROR", None):
        return None
    try:
        ret_str = return_str.replace("%", "")
        return float(ret_str) / 100 if "%" in return_str else float(ret_str)
    except (ValueError, AttributeError) as e:
        logger.debug(f"Failed to parse return '{return_str}': {e}")
        return None


def _parse_sharpe(sharpe_str: str) -> float:
    """Parse Sharpe string to float."""
    if sharpe_str in ("N/A", "ERROR", None):
        return 0.0
    try:
        return float(sharpe_str)
    except (ValueError, AttributeError) as e:
        logger.debug(f"Failed to parse sharpe '{sharpe_str}': {e}")
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
    except (ValueError, AttributeError) as e:
        logger.debug(f"Failed to parse win rate '{win_rate}': {e}")
        win_rate_val = 0.0

    try:
        dd_str = max_drawdown.replace("%", "") if isinstance(max_drawdown, str) else "0"
        drawdown_val = float(dd_str) / 100 if "%" in max_drawdown else float(dd_str)
    except (ValueError, AttributeError) as e:
        logger.debug(f"Failed to parse drawdown '{max_drawdown}': {e}")
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
    Strategies are printed in the same order as results_df (sorted by Total Return).
    """
    print_section(" ALL STRATEGY DECISIONS")

    decisions = {}
    action_counts = {"BUY": 0, "HOLD": 0, "SELL": 0}  # For consensus

    # Create a mapping from strategy name to strategy object
    strategy_map = {s.name: s for s in strategies}

    # Process strategies in the order of results_df (sorted by Total Return)
    for _, result_row in results_df.iterrows():
        strategy_name = result_row["Strategy"]
        strategy = strategy_map.get(strategy_name)

        if strategy is None:
            continue

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

            # Get backtest result from results_df
            total_return = result_row["Total Return"]
            sharpe = _parse_sharpe(result_row["Sharpe Ratio"])
            win_rate = result_row["Win Rate"]
            max_drawdown = result_row["Max Drawdown"]

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
            if hasattr(strategy, "description") and strategy.description:
                print(f"    Description: {strategy.description}")

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
    """Print top feature importances with descriptions."""
    # Feature descriptions mapping - complete list
    feature_descriptions = {
        # ============ Moving Averages ============
        "ma_5": "5日均线",
        "ma_10": "10日均线",
        "ma_20": "20日均线",
        "ma_60": "60日均线",
        "ma_120": "120日均线",
        "ma_5_ratio": "价格/MA5比率",
        "ma_10_ratio": "价格/MA10比率",
        "ma_20_ratio": "价格/MA20比率",
        "ma_60_ratio": "价格/MA60比率",
        "ma_120_ratio": "价格/MA120比率",
        "ma_5_above_10": "MA5在MA10上方",
        "ma_5_above_20": "MA5在MA20上方",
        "ma_10_above_20": "MA10在MA20上方",
        "ma_bullish_arrange": "均线多头排列",
        "ma_bearish_arrange": "均线空头排列",
        "ma_convergence": "均线收敛度",
        "ma_slope_20": "20日均线斜率",
        "ema_12": "12日指数均线",
        "ema_26": "26日指数均线",
        "deviation_ma5_abs": "偏离MA5幅度",
        "deviation_ma10_abs": "偏离MA10幅度",
        "deviation_ma20_abs": "偏离MA20幅度",
        # ============ Price Features ============
        "close": "收盘价",
        "open": "开盘价",
        "high": "最高价",
        "low": "最低价",
        "volume": "成交量",
        "current_price": "当前价格",
        "high_20d": "20日最高价",
        "low_20d": "20日最低价",
        "price_position": "价格位置(20日区间)",
        "close_position": "收盘价位置",
        "close_low_ratio": "收盘价/最低价比率",
        "high_low_ratio": "最高/最低价比率",
        # ============ Bollinger Bands ============
        "bb_upper": "布林带上轨",
        "bb_lower": "布林带下轨",
        "bb_middle": "布林带中轨",
        "bb_width": "布林带宽度",
        "bb_position": "布林带位置",
        # ============ RSI ============
        "rsi": "RSI相对强弱指标",
        "rsi_lag_1": "RSI滞后1日",
        "rsi_lag_2": "RSI滞后2日",
        "rsi_lag_3": "RSI滞后3日",
        # ============ MACD ============
        "macd": "MACD指标",
        "macd_signal": "MACD信号线",
        "macd_hist": "MACD柱状图",
        "macd_position": "MACD位置",
        "macd_cross_up": "MACD金叉",
        "macd_cross_above_zero": "MACD上穿零轴",
        "macd_lag_1": "MACD滞后1日",
        "macd_lag_2": "MACD滞后2日",
        "macd_lag_3": "MACD滞后3日",
        "macd_hist_lag_1": "MACD柱状图滞后1日",
        "macd_hist_lag_2": "MACD柱状图滞后2日",
        "macd_hist_lag_3": "MACD柱状图滞后3日",
        # ============ Stochastic ============
        "stoch_k": "随机指标K值",
        "stoch_d": "随机指标D值",
        # ============ ATR ============
        "atr": "平均真实波幅",
        "atr_shrinking": "ATR收缩",
        # ============ Volume ============
        "volume_ratio": "成交量比率",
        "volume_ma5": "5日成交量均线",
        "volume_ma20": "20日成交量均线",
        "volume_change": "成交量变化",
        "volume_change_lag_1": "成交量变化滞后1日",
        "volume_change_lag_2": "成交量变化滞后2日",
        "volume_change_lag_3": "成交量变化滞后3日",
        "volume_change_lag_5": "成交量变化滞后5日",
        "volume_increasing": "成交量放大",
        "volume_breakout_flag": "成交量突破标志",
        "volume_vs": "相对成交量",
        # ============ Momentum ============
        "returns": "日收益率",
        "momentum_5": "5日动量",
        "momentum_10": "10日动量",
        "momentum_20": "20日动量",
        "momentum_acceleration": "动量加速度",
        "return_lag_1": "收益率滞后1日",
        "return_lag_2": "收益率滞后2日",
        "return_lag_3": "收益率滞后3日",
        "return_lag_5": "收益率滞后5日",
        "return_2d": "2日收益率",
        "return_3d": "3日收益率",
        "roc_5": "5日变动率",
        "roc_10": "10日变动率",
        "roc_20": "20日变动率",
        # ============ Volatility ============
        "volatility_5": "5日波动率",
        "volatility_10": "10日波动率",
        "volatility_20": "20日波动率",
        "volatility_20d": "20日波动率(标准)",
        "returns_std_5": "5日收益标准差",
        "returns_std_10": "10日收益标准差",
        "returns_skew_10": "10日收益偏度",
        "returns_skew_20": "20日收益偏度",
        "cv": "变异系数",
        "low_volatility_flag": "低波动标志",
        # ============ CCI ============
        "cci": "CCI商品通道指数",
        # ============ MFI ============
        "mfi": "MFI资金流量指标",
        # ============ DMI ============
        "dmi_plus_di": "DMI+DI正向指标",
        "dmi_minus_di": "DMI-DI负向指标",
        "dmi_di_diff": "DMI差值",
        "dmi_adx": "ADX趋势强度",
        "adx": "ADX平均趋向指数",
        # ============ Aroon ============
        "aroon_high": "Aroon高位",
        "aroon_low": "Aroon低位",
        "aroon_up": "Aroon上升",
        "aroon_down": "Aroon下降",
        "aroon_oscillator": "Aroon振荡器",
        "aroon_trend": "Aroon趋势",
        "aroon_dmi_bullish": "Aroon-DMI多头",
        "aroon_dmi_bearish": "Aroon-DMI空头",
        # ============ AD Line ============
        "ad_line": "累积/派发线",
        "ad_oscillator": "AD振荡器",
        # ============ VWAP ============
        "vwap": "成交量加权平均价",
        "price_to_vwap": "价格/VWAP比率",
        "price_vs_vwap": "价格相对VWAP",
        # ============ Candlestick ============
        "body_ratio": "实体比例",
        "upper_shadow_ratio": "上影线比例",
        "lower_shadow_ratio": "下影线比例",
        "long_lower_shadow": "长下影线",
        # ============ Divergence ============
        "bottom_divergence": "底背离",
        "top_divergence": "顶背离",
        "divergence_strength": "背离强度",
        # ============ Support/Resistance ============
        "distance_to_support": "到支撑位距离",
        "distance_to_resistance": "到阻力位距离",
        # ============ Box Oscillation ============
        "box_top": "箱体顶部",
        "box_bottom": "箱体底部",
        "box_width_pct": "箱体宽度%",
        "box_touch_top_count": "触及箱顶次数",
        "box_touch_bottom_count": "触及箱底次数",
        "near_box_top": "接近箱顶",
        "near_box_bottom": "接近箱底",
        "in_box_middle": "在箱体中部",
        "breakout_up": "向上突破",
        "breakout_down": "向下突破",
        "breakout_volume_confirm": "放量突破确认",
        # ============ Trend ============
        "is_bullish": "是否看多",
        "trend_score": "趋势评分",
        "sharpe_like": "类夏普比率",
        "max_drawdown_20d": "20日最大回撤",
        "expanding_drawdown": "回撤扩大",
        "risk_level": "风险等级",
        # ============ Signals ============
        "signal_buy": "买入信号",
        "signal_sell": "卖出信号",
        "signal_hold": "持有信号",
        # ============ Golden/Death Cross ============
        "golden_cross_5_10": "5/10日金叉",
        "golden_cross_5_20": "5/20日金叉",
        "golden_cross_10_20": "10/20日金叉",
        "death_cross_5_10": "5/10日死叉",
        "death_cross_5_20": "5/20日死叉",
        "death_cross_10_20": "10/20日死叉",
        # ============ Streak ============
        "streak_up_2": "连续上涨2日",
        "streak_up_3": "连续上涨3日",
        "streak_down_2": "连续下跌2日",
        "streak_down_3": "连续下跌3日",
        # ============ Patterns ============
        "pattern_reversal_up": "反转上涨形态",
        "pattern_reversal_down": "反转下跌形态",
        "pattern_rise_then_drop": "冲高回落形态",
        "pattern_drop_then_rise": "探底回升形态",
        "pattern_vol_surge_rise": "放量上涨形态",
        "pattern_vol_surge_drop": "放量下跌形态",
        "oyty_pattern": "OYTY形态",
        "oyty_bullish_body": "OYTY阳线实体",
        "oyty_breakout": "OYTY突破",
        "oyty_support_hold": "OYTY支撑确认",
        "oyty_shrink_volume": "OYTY缩量",
        "price_stabilize": "价格企稳",
        "bottom_volume_flag": "底部放量标志",
        "bottom_volume_surge": "底部放量",
        "shrink_pullback_flag": "缩量回调标志",
        # ============ Market Features ============
        "index_returns": "指数收益率",
        "index_rsi": "指数RSI",
        "index_ma_5": "指数MA5",
        "index_ma_10": "指数MA10",
        "index_ma_20": "指数MA20",
        "index_ma_5_ratio": "指数MA5比率",
        "index_ma_10_ratio": "指数MA10比率",
        "index_ma_20_ratio": "指数MA20比率",
        "index_momentum_5": "指数5日动量",
        "index_momentum_10": "指数10日动量",
        "index_momentum_20": "指数20日动量",
        "index_volume_ratio": "指数成交量比率",
        "index_volatility_5": "指数5日波动率",
        "index_volatility_20": "指数20日波动率",
        "index_close": "指数收盘价",
        "index_high": "指数最高价",
        "index_low": "指数最低价",
        "index_open": "指数开盘价",
        "index_volume": "指数成交量",
        "index_volume_ma20": "指数20日成交量均线",
        "index_log_returns": "指数对数收益率",
        "index_high_52w": "指数52周最高",
        "index_low_52w": "指数52周最低",
        "index_position_52w": "指数52周位置",
        # ============ Relative Performance ============
        "alpha": "超额收益(Alpha)",
        "alpha_5d": "5日超额收益",
        "alpha_10d": "10日超额收益",
        "sector_relative": "相对行业表现",
        "sector_relative_5d": "5日相对行业表现",
        "market_corr_5": "5日市场相关性",
        "market_corr_10": "10日市场相关性",
        "beta_5": "5日Beta系数",
        "beta_20": "20日Beta系数",
        # ============ Sector Features ============
        "sector_returns": "行业收益率",
        "sector_rsi": "行业RSI",
        "sector_volume": "行业成交量",
        "sector_ma20": "行业20日均线",
        "sector_ma_ratio": "行业均线比率",
        "sector_close": "行业收盘价",
        # ============ Money Flow ============
        "main_net_flow": "主力净流入",
        "main_net_flow_ratio": "主力净流入比率",
        "main_net_flow_ratio_ma5": "主力净流入比率5日均线",
        "main_net_flow_momentum_3d": "主力净流入3日动量",
        "main_net_flow_volatility": "主力净流入波动率",
        "large_net_flow": "大单净流入",
        "large_net_flow_ratio": "大单净流入比率",
        "large_net_flow_momentum_3d": "大单净流入3日动量",
        "medium_net_flow": "中单净流入",
        "medium_net_flow_ratio": "中单净流入比率",
        "small_net_flow": "小单净流入",
        "small_net_flow_ratio": "小单净流入比率",
        "small_net_flow_momentum_3d": "小单净流入3日动量",
        "super_large_net_flow": "超大单净流入",
        "super_large_net_flow_ratio": "超大单净流入比率",
        # ============ Sentiment Features ============
        "sentiment_score": "情绪分数",
        "sentiment_ma3": "3日情绪均线",
        "sentiment_ma7": "7日情绪均线",
        "sentiment_ma14": "14日情绪均线",
        "sentiment_std7": "7日情绪波动",
        "sentiment_std14": "14日情绪波动",
        "sentiment_momentum": "情绪动量",
        "sentiment_momentum3": "3日情绪动量",
        "sentiment_acceleration": "情绪加速度",
        "sentiment_trend": "情绪趋势",
        "sentiment_regime": "情绪状态",
        "sentiment_extreme_positive": "极度乐观情绪",
        "sentiment_extreme_negative": "极度悲观情绪",
        "sentiment_neutral": "中性情绪",
        "news_count": "新闻数量",
        "news_count_ma3": "3日新闻数量均线",
        "news_count_ma7": "7日新闻数量均线",
        "news_volume_change": "新闻量变化",
        "news_volume_spike": "新闻量激增",
        "weighted_sentiment": "加权情绪",
        "sentiment_divergence": "情绪背离",
        # ============ Fundamental Features ============
        "pe_ratio": "市盈率",
        "pb_ratio": "市净率",
        "ps_ratio": "市销率",
        "peg_ratio": "PEG比率",
        "forward_pe": "远期市盈率",
        "roe": "净资产收益率",
        "roa": "总资产收益率",
        "revenue_growth": "营收增长率",
        "earnings_growth": "盈利增长率",
        "earnings_quarterly_growth": "季度盈利增长",
        "gross_margin": "毛利率",
        "net_margin": "净利率",
        "operating_margin": "营业利润率",
        "current_ratio": "流动比率",
        "quick_ratio": "速动比率",
        "debt_to_equity": "资产负债率",
        "dividend_yield": "股息率",
        "dividend_rate": "分红率",
        "payout_ratio": "派息率",
        "shares_outstanding": "流通股本",
        "market_cap": "市值",
        "enterprise_value": "企业价值",
        "institutional_ratio": "机构持股比例",
        "number_of_analyst_recommendation": "分析师推荐数",
        "price_to_target": "价格/目标价",
        "target_high_price": "目标最高价",
        "target_low_price": "目标最低价",
        "target_mean_price": "目标平均价",
        "recommendation_key": "推荐等级",
        "week_52_high": "52周最高价",
        "week_52_low": "52周最低价",
        "week_52_high_ratio": "价格/52周最高",
        "week_52_low_ratio": "价格/52周最低",
        # ============ Log Returns ============
        "log_returns": "对数收益率",
    }

    try:
        importance = model.get_feature_importance()
        print_section(f" TOP {top_n} FEATURE IMPORTANCE")
        print()
        top_features = importance.head(top_n)
        for _, row in top_features.iterrows():
            feature_name = row["feature"]
            importance_val = row["importance"]
            bar = "█" * int(importance_val / 2)
            description = feature_descriptions.get(feature_name, "")
            if description:
                print(
                    f"  {feature_name:<30} {importance_val:6.2f} {bar}  # {description}"
                )
            else:
                print(f"  {feature_name:<30} {importance_val:6.2f} {bar}")
    except Exception as e:
        print(f"  (Could not display feature importance: {e})")


def print_stock_core_data(
    df: pd.DataFrame, full_history_df: pd.DataFrame = None
) -> None:
    """Print core stock data including support/resistance levels and key indicators.

    Args:
        df: Current period data (backtest_df) for display
        full_history_df: Full historical data for calculating performance metrics
    """
    if df.empty:
        return

    # Use full_history_df for performance calculations if provided
    history_df = (
        full_history_df
        if full_history_df is not None and not full_history_df.empty
        else df
    )

    print_section(" STOCK CORE DATA")

    # Get latest data
    latest = df.iloc[-1]
    current_price = latest.get("close", 0)

    print(f"\n  === Price Info ===")
    print(f"  Current Price : {current_price:.2f}")
    print(f"  Open          : {latest.get('open', 'N/A')}")
    print(f"  High          : {latest.get('high', 'N/A')}")
    print(f"  Low           : {latest.get('low', 'N/A')}")
    print(f"  Volume        : {latest.get('volume', 'N/A'):,.0f}")

    # Moving Averages
    print(f"\n  === Moving Averages ===")
    for period in [5, 10, 20, 60]:
        ma_col = f"ma_{period}"
        if ma_col in df.columns:
            ma_value = latest.get(ma_col)
            if pd.notna(ma_value) and ma_value > 0:
                diff_pct = (current_price - ma_value) / ma_value * 100
                direction = "↑" if diff_pct > 0 else "↓"
                print(
                    f"  MA{period:<2}          : {ma_value:.2f} ({direction}{abs(diff_pct):.1f}%)"
                )

    # Support and Resistance
    print(f"\n  === Support & Resistance ===")

    # Recent high/low
    high_20d = df["high"].tail(20).max() if len(df) >= 20 else df["high"].max()
    low_20d = df["low"].tail(20).min() if len(df) >= 20 else df["low"].min()
    print(
        f"  Resistance (20d High) : {high_20d:.2f} (+{((high_20d - current_price) / current_price * 100):.1f}%)"
    )
    print(
        f"  Support (20d Low)     : {low_20d:.2f} ({((low_20d - current_price) / current_price * 100):.1f}%)"
    )

    # Bollinger Bands
    if "bb_upper" in df.columns and "bb_lower" in df.columns:
        bb_upper = latest.get("bb_upper")
        bb_lower = latest.get("bb_lower")
        if pd.notna(bb_upper) and pd.notna(bb_lower):
            print(f"  BB Upper       : {bb_upper:.2f}")
            print(f"  BB Lower       : {bb_lower:.2f}")

    # Key Indicators
    print(f"\n  === Key Indicators ===")

    # RSI
    if "rsi" in df.columns:
        rsi = latest.get("rsi")
        if pd.notna(rsi):
            rsi_status = "超买" if rsi > 70 else ("超卖" if rsi < 30 else "中性")
            print(f"  RSI (14)       : {rsi:.1f} ({rsi_status})")

    # MACD
    if "macd" in df.columns and "macd_signal" in df.columns:
        macd = latest.get("macd")
        macd_signal = latest.get("macd_signal")
        macd_hist = latest.get("macd_hist")
        if pd.notna(macd) and pd.notna(macd_signal):
            macd_status = "多头" if macd > macd_signal else "空头"
            print(f"  MACD           : {macd:.4f} ({macd_status})")
            if pd.notna(macd_hist):
                print(f"  MACD Hist      : {macd_hist:.4f}")

    # ATR
    if "atr" in df.columns:
        atr = latest.get("atr")
        if pd.notna(atr) and atr > 0:
            atr_pct = atr / current_price * 100
            print(f"  ATR (14)       : {atr:.2f} ({atr_pct:.1f}%)")

    # Volume Analysis
    print(f"\n  === Volume Analysis ===")
    if "volume_ratio" in df.columns:
        vol_ratio = latest.get("volume_ratio")
        if pd.notna(vol_ratio):
            vol_status = (
                "放量" if vol_ratio > 1.5 else ("缩量" if vol_ratio < 0.8 else "正常")
            )
            print(f"  Volume Ratio   : {vol_ratio:.2f} ({vol_status})")

    # Returns
    print(f"\n  === Recent Returns ===")
    for period in [1, 5, 10, 20]:
        ret_col = f"momentum_{period}"
        if ret_col in df.columns:
            ret = latest.get(ret_col)
            if pd.notna(ret):
                print(f"  {period}d Return     : {ret * 100:+.2f}%")

    # Sentiment Analysis
    print(f"\n  === Market Sentiment ===")
    sentiment_cols = [col for col in df.columns if "sentiment" in col.lower()]
    if sentiment_cols:
        sentiment_score = latest.get("sentiment_score", 0)
        news_count = latest.get("news_count", 0)
        sentiment_ma3 = latest.get("sentiment_ma3", 0)
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

        print(f"  Sentiment Score : {sentiment_score:.3f} {sentiment_emoji}")
        print(f"  Sentiment Label : {sentiment_label}")
        print(f"  News Count      : {news_count:.0f}")
        print(f"  3-day MA        : {sentiment_ma3:.3f}")
        print(f"  Trend           : {sentiment_trend:+.3f}")

        # News activity level
        if news_count > 10:
            news_activity = "高 (High)"
        elif news_count > 3:
            news_activity = "中 (Moderate)"
        else:
            news_activity = "低 (Low)"
        print(f"  News Activity   : {news_activity}")

        # Extreme sentiment warning
        if sentiment_score > 0.5:
            print(f"  ⚠️  市场情绪极度乐观，注意风险")
        elif sentiment_score < -0.5:
            print(f"  ⚠️  市场情绪极度悲观，可能存在机会")
    else:
        print(f"  情绪数据不可用 (Sentiment data not available)")

    # ========== Simple Analysis for Beginners ==========
    print(f"\n  === Simple Analysis ===")
    analyses = []

    # Price vs MA analysis
    for period in [5, 10, 20, 60]:
        ma_col = f"ma_{period}"
        if ma_col in df.columns:
            ma_value = latest.get(ma_col)
            if pd.notna(ma_value) and ma_value > 0:
                if current_price > ma_value * 1.05:
                    analyses.append(f"价格高于MA{period}均线5%以上，可能偏高")
                elif current_price < ma_value * 0.95:
                    analyses.append(f"价格低于MA{period}均线5%以下，可能偏低")
                elif current_price > ma_value:
                    analyses.append(f"价格在MA{period}均线上方运行")
                else:
                    analyses.append(f"价格跌破MA{period}均线")

    # RSI analysis
    if "rsi" in df.columns:
        rsi = latest.get("rsi")
        if pd.notna(rsi):
            if rsi > 80:
                analyses.append("RSI严重超买，短期回调风险大")
            elif rsi > 70:
                analyses.append("RSI进入超买区域，注意风险")
            elif rsi < 20:
                analyses.append("RSI严重超卖，可能存在反弹机会")
            elif rsi < 30:
                analyses.append("RSI进入超卖区域，可关注反弹机会")

    # MACD analysis
    if "macd" in df.columns and "macd_signal" in df.columns:
        macd = latest.get("macd")
        macd_signal = latest.get("macd_signal")
        macd_hist = latest.get("macd_hist")
        if pd.notna(macd) and pd.notna(macd_signal) and pd.notna(macd_hist):
            if macd > macd_signal and macd_hist > 0:
                analyses.append("MACD金叉且柱状图向上，多头信号")
            elif macd < macd_signal and macd_hist < 0:
                analyses.append("MACD死叉且柱状图向下，空头信号")
            elif macd_hist > 0 and macd_hist > latest.get("macd_hist_lag_1", 0):
                analyses.append("MACD柱状图放大，动能增强")
            elif macd_hist < 0 and macd_hist < latest.get("macd_hist_lag_1", 0):
                analyses.append("MACD柱状图缩小，动能减弱")

    # Volume analysis
    if "volume_ratio" in df.columns:
        vol_ratio = latest.get("volume_ratio")
        if pd.notna(vol_ratio):
            if vol_ratio > 3:
                analyses.append("成交量异常放大，需关注是否有重大消息")
            elif vol_ratio > 2:
                analyses.append("成交量明显放大，市场关注度提升")
            elif vol_ratio < 0.5:
                analyses.append("成交量极度萎缩，市场观望情绪浓厚")

    # Support/Resistance analysis
    if current_price >= high_20d * 0.99:
        analyses.append("价格接近20日高点，上方有压力")
    elif current_price <= low_20d * 1.01:
        analyses.append("价格接近20日低点，下方有支撑")

    # Bollinger Bands analysis
    if "bb_upper" in df.columns and "bb_lower" in df.columns:
        bb_upper = latest.get("bb_upper")
        bb_lower = latest.get("bb_lower")
        if pd.notna(bb_upper) and pd.notna(bb_lower):
            if current_price >= bb_upper * 0.99:
                analyses.append("价格触及布林带上轨，短期可能回调")
            elif current_price <= bb_lower * 1.01:
                analyses.append("价格触及布林带下轨，短期可能反弹")

    # Trend analysis
    if "ma_5" in df.columns and "ma_20" in df.columns:
        ma5 = latest.get("ma_5")
        ma20 = latest.get("ma_20")
        if pd.notna(ma5) and pd.notna(ma20):
            if ma5 > ma20:
                analyses.append("短期均线在长期均线上方，趋势偏多")
            else:
                analyses.append("短期均线在长期均线下方，趋势偏空")

    # Print analyses
    if analyses:
        for i, analysis in enumerate(analyses[:8], 1):  # Limit to 8 analyses
            print(f"  {i}. {analysis}")
    else:
        print("  暂无明显技术信号")

    # ========== Historical Performance Summary ==========
    print(f"\n  === Historical Performance ===")

    # Calculate performance metrics using full history data
    if len(history_df) >= 20:
        # Get last 20 trading days data
        last_20 = history_df.tail(20).copy()

        # 20-day performance (use actual close prices)
        start_price_20d = last_20["close"].iloc[0]
        end_price = last_20["close"].iloc[-1]
        perf_20d = (end_price - start_price_20d) / start_price_20d * 100

        # Daily volatility (standard deviation of daily returns)
        if "returns" in last_20.columns:
            daily_returns = last_20["returns"].dropna()
            if len(daily_returns) > 0:
                daily_volatility = daily_returns.std() * 100
                annualized_volatility = daily_volatility * (252**0.5)  # Annualize
            else:
                daily_volatility = 0
                annualized_volatility = 0
        else:
            # Calculate returns from close prices
            daily_returns = last_20["close"].pct_change().dropna()
            daily_volatility = (
                daily_returns.std() * 100 if len(daily_returns) > 0 else 0
            )
            annualized_volatility = daily_volatility * (252**0.5)

        # Max drawdown (from peak to trough)
        rolling_max = last_20["close"].cummax()
        drawdown_series = (last_20["close"] - rolling_max) / rolling_max
        max_dd = drawdown_series.min() * 100

        # Verify max drawdown is reasonable
        price_range_pct = (
            (last_20["close"].max() - last_20["close"].min())
            / last_20["close"].max()
            * 100
        )
        if abs(max_dd) > price_range_pct * 1.5:
            # Recalculate if max drawdown seems unreasonable
            max_dd = -price_range_pct

        # Win rate (days with positive returns)
        if "returns" in last_20.columns:
            positive_days = (last_20["returns"] > 0).sum()
        else:
            calculated_returns = last_20["close"].pct_change()
            positive_days = (calculated_returns > 0).sum()
        win_rate = positive_days / 20 * 100

        print(f"  20日涨跌幅     : {perf_20d:+.2f}%")
        print(
            f"  20日波动率     : {daily_volatility:.2f}% (日) / {annualized_volatility:.1f}% (年化)"
        )
        print(f"  20日最大回撤   : {max_dd:.2f}%")
        print(f"  20日胜率       : {win_rate:.0f}% ({positive_days}/20天)")

        # Performance rating
        if perf_20d > 5:
            perf_rating = "近期表现强势"
        elif perf_20d > 0:
            perf_rating = "近期表现一般"
        elif perf_20d > -5:
            perf_rating = "近期表现偏弱"
        else:
            perf_rating = "近期表现较弱"
        print(f"  综合评价       : {perf_rating}")


def main():
    args = parse_args()
    stock_code = args.stock

    print_section(f" STOCK TRADING DECISION SYSTEM")
    print(f"\n  Stock Code: {stock_code}")

    # Resolve stock info
    try:
        stock_info = StockInfoResolver.resolve(stock_code)
        market = stock_info.market.replace("_share", "")
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

    # Process important dates if enabled
    excluded_dates = []
    if args.exclude_dates:
        print("\n  Processing important dates...")
        dates_manager = get_important_dates_manager()

        start_date = (
            features_df["date"].min().strftime("%Y-%m-%d")
            if "date" in features_df.columns
            else None
        )
        end_date = (
            features_df["date"].max().strftime("%Y-%m-%d")
            if "date" in features_df.columns
            else None
        )

        excluded_dates = dates_manager.get_or_detect_dates(
            df=features_df,
            market=market,
            start_date=start_date,
            end_date=end_date,
            auto_detect=True,
        )

        if excluded_dates:
            print(f"  Found {len(excluded_dates)} extreme volatility dates to exclude")
            for d in excluded_dates[:5]:
                print(f"    - {d}")
            if len(excluded_dates) > 5:
                print(f"    ... and {len(excluded_dates) - 5} more")

            features_df_dates = pd.to_datetime(features_df["date"]).dt.strftime(
                "%Y-%m-%d"
            )
            mask = ~features_df_dates.isin(excluded_dates)
            features_df = features_df[mask].reset_index(drop=True)
            print(f"  Remaining samples after filtering: {len(features_df)}")

            if len(features_df) < 30:
                print(
                    "  Warning: Insufficient data after filtering, disabling date exclusion"
                )
                features_df = combinator.get_combined_features(
                    stock_code, days=args.train_days, force_refresh=args.refresh
                )
                excluded_dates = []
        else:
            print("  No extreme volatility dates detected")

    # Try to get real-time price for latest data point
    try:
        from src.data_providers import fetch_realtime_price

        realtime_price = fetch_realtime_price(stock_code)
        if realtime_price is not None:
            # Update the last row's close price with real-time price
            last_idx = features_df.index[-1]
            old_close = features_df.loc[last_idx, "close"]
            features_df.loc[last_idx, "close"] = realtime_price
            # Also update high/low if needed
            if realtime_price > features_df.loc[last_idx, "high"]:
                features_df.loc[last_idx, "high"] = realtime_price
            if realtime_price < features_df.loc[last_idx, "low"]:
                features_df.loc[last_idx, "low"] = realtime_price
            print(f"  Realtime price  : {realtime_price:.2f} (was {old_close:.2f})")
    except Exception as e:
        logger.debug(f"Failed to fetch realtime price, using historical data: {e}")

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

        # Top3 Voting Strategy: select top 3 strategies by return and vote
        top3_strategies = results_df.head(3)
        top3_votes = {"BUY": 0, "HOLD": 0, "SELL": 0}
        top3_details = []

        for _, row in top3_strategies.iterrows():
            strategy_name = row["Strategy"]
            if strategy_name in decisions:
                action = decisions[strategy_name]["action"]
                top3_votes[action] = top3_votes.get(action, 0) + 1
                top3_details.append(
                    {
                        "name": strategy_name,
                        "return": row["Total Return"],
                        "action": action,
                    }
                )

        # Determine Top3 action with priority: BUY > SELL > HOLD
        top3_action = "HOLD"
        if top3_votes["BUY"] > 0:
            top3_action = "BUY"
        elif top3_votes["SELL"] > 0:
            top3_action = "SELL"

        # Check if there's a clear majority
        max_votes = max(top3_votes.values())
        actions_with_max_votes = [a for a, v in top3_votes.items() if v == max_votes]
        if len(actions_with_max_votes) == 1:
            top3_action = actions_with_max_votes[0]

        # Determine final action by majority vote
        final_action = (
            max(action_counts, key=action_counts.get) if action_counts else "HOLD"
        )
        final_confidence = best_confidence
        final_strategy = best_strategy_name

        # Print stock core data before final recommendation
        print_stock_core_data(backtest_df, features_df)

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

            # Get best strategy description
            best_strategy_desc = ""
            for strategy in strategies:
                if (
                    strategy.name == best_strategy_name
                    and hasattr(strategy, "description")
                    and strategy.description
                ):
                    best_strategy_desc = strategy.description
                    break
            if best_strategy_desc:
                print(f"    Description: {best_strategy_desc}")

            print(f"\n  Best Strategy by Return ({best_return_name}):")
            print(f"    Return    : {best_return_value:.2%}")
            print(f"    Decision  : {best_return_action}")

            # Get best return strategy description
            best_return_desc = ""
            for strategy in strategies:
                if (
                    strategy.name == best_return_name
                    and hasattr(strategy, "description")
                    and strategy.description
                ):
                    best_return_desc = strategy.description
                    break
            if best_return_desc:
                print(f"    Description: {best_return_desc}")

            # Print Top3 Voting Strategy
            print(f"\n  === Top3 Voting Strategy ===")
            print(f"  Action: {top3_action}")
            print(
                f"  Votes: BUY={top3_votes['BUY']}, SELL={top3_votes['SELL']}, HOLD={top3_votes['HOLD']}"
            )
            print(f"  Top Strategies:")
            for detail in top3_details:
                print(
                    f"    - {detail['name']}: {detail['return']} -> {detail['action']}"
                )

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

            # Get best return strategy description
            best_return_desc = ""
            for strategy in strategies:
                if (
                    strategy.name == best_return_name
                    and hasattr(strategy, "description")
                    and strategy.description
                ):
                    best_return_desc = strategy.description
                    break
            if best_return_desc:
                print(f"    Description: {best_return_desc}")

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
    print("=" * 60)


if __name__ == "__main__":
    main()
