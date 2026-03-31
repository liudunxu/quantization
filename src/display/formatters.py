"""Output formatters for displaying results."""

import pandas as pd
from typing import Dict, Any, Optional

from ..models import StockTradingModel


# Feature descriptions mapping
FEATURE_DESCRIPTIONS = {
    # Moving Averages
    "ma_5": "5日均线",
    "ma_10": "10日均线",
    "ma_20": "20日均线",
    "ma_60": "60日均线",
    "ma_5_ratio": "价格/MA5比率",
    "ma_10_ratio": "价格/MA10比率",
    "ma_20_ratio": "价格/MA20比率",
    "ma_5_above_10": "MA5在MA10上方",
    "ma_5_above_20": "MA5在MA20上方",
    "ma_10_above_20": "MA10在MA20上方",
    "ma_bullish_arrange": "均线多头排列",
    "ma_bearish_arrange": "均线空头排列",
    "ema_12": "12日指数均线",
    "ema_26": "26日指数均线",
    # Price Features
    "close": "收盘价",
    "open": "开盘价",
    "high": "最高价",
    "low": "最低价",
    "volume": "成交量",
    "high_20d": "20日最高价",
    "low_20d": "20日最低价",
    "price_position": "价格位置(20日区间)",
    # Bollinger Bands
    "bb_upper": "布林带上轨",
    "bb_lower": "布林带下轨",
    "bb_middle": "布林带中轨",
    "bb_width": "布林带宽度",
    "bb_position": "布林带位置",
    # RSI
    "rsi": "RSI相对强弱指标",
    "rsi_lag_1": "RSI滞后1日",
    "rsi_lag_2": "RSI滞后2日",
    "rsi_lag_3": "RSI滞后3日",
    # MACD
    "macd": "MACD指标",
    "macd_signal": "MACD信号线",
    "macd_hist": "MACD柱状图",
    "macd_lag_1": "MACD滞后1日",
    "macd_lag_2": "MACD滞后2日",
    "macd_hist_lag_1": "MACD柱状图滞后1日",
    "macd_hist_lag_2": "MACD柱状图滞后2日",
    # Stochastic
    "stoch_k": "随机指标K值",
    "stoch_d": "随机指标D值",
    # ATR
    "atr": "平均真实波幅",
    # Volume
    "volume_ratio": "成交量比率",
    "volume_ma5": "5日成交量均线",
    "volume_ma20": "20日成交量均线",
    "volume_change": "成交量变化",
    # Momentum
    "returns": "日收益率",
    "momentum_5": "5日动量",
    "momentum_10": "10日动量",
    "momentum_20": "20日动量",
    "return_lag_1": "收益率滞后1日",
    "return_lag_2": "收益率滞后2日",
    "return_lag_3": "收益率滞后3日",
    "roc_5": "5日变动率",
    "roc_10": "10日变动率",
    "roc_20": "20日变动率",
    # Volatility
    "volatility_5": "5日波动率",
    "volatility_10": "10日波动率",
    "volatility_20": "20日波动率",
    "returns_std_5": "5日收益标准差",
    "returns_std_10": "10日收益标准差",
    "returns_skew_10": "10日收益偏度",
    "returns_skew_20": "20日收益偏度",
    # CCI & MFI
    "cci": "CCI商品通道指数",
    "mfi": "MFI资金流量指标",
    # DMI
    "dmi_plus_di": "DMI+DI正向指标",
    "dmi_minus_di": "DMI-DI负向指标",
    "dmi_adx": "ADX趋势强度",
    # Aroon
    "aroon_high": "Aroon高位",
    "aroon_low": "Aroon低位",
    "aroon_oscillator": "Aroon振荡器",
    # AD Line
    "ad_line": "累积/派发线",
    "ad_oscillator": "AD振荡器",
    # VWAP
    "vwap": "成交量加权平均价",
    "price_to_vwap": "价格/VWAP比率",
    # Candlestick
    "body_ratio": "实体比例",
    "upper_shadow_ratio": "上影线比例",
    "lower_shadow_ratio": "下影线比例",
    # Divergence
    "bottom_divergence": "底背离",
    "top_divergence": "顶背离",
    # Support/Resistance
    "distance_to_support": "到支撑位距离",
    "distance_to_resistance": "到阻力位距离",
    # Box Oscillation
    "box_top": "箱体顶部",
    "box_bottom": "箱体底部",
    "box_width_pct": "箱体宽度%",
    "breakout_up": "向上突破",
    "breakout_down": "向下突破",
    # Trend
    "is_bullish": "是否看多",
    "trend_score": "趋势评分",
    "max_drawdown_20d": "20日最大回撤",
    # Signals
    "signal_buy": "买入信号",
    "signal_sell": "卖出信号",
    "signal_hold": "持有信号",
    # Golden/Death Cross
    "golden_cross_5_10": "5/10日金叉",
    "golden_cross_5_20": "5/20日金叉",
    "death_cross_5_10": "5/10日死叉",
    "death_cross_5_20": "5/20日死叉",
    # Market Features
    "index_returns": "指数收益率",
    "index_rsi": "指数RSI",
    "index_momentum_5": "指数5日动量",
    "index_momentum_10": "指数10日动量",
    "index_momentum_20": "指数20日动量",
    "index_ma_5": "指数MA5",
    "index_ma_10": "指数MA10",
    "index_ma_20": "指数MA20",
    # Relative Performance
    "alpha": "超额收益(Alpha)",
    "alpha_5d": "5日超额收益",
    "alpha_10d": "10日超额收益",
    "market_corr_5": "5日市场相关性",
    "market_corr_10": "10日市场相关性",
    "beta_5": "5日Beta系数",
    "beta_20": "20日Beta系数",
    # Sentiment
    "sentiment_score": "情绪分数",
    "sentiment_ma3": "3日情绪均线",
    "sentiment_ma7": "7日情绪均线",
    "sentiment_trend": "情绪趋势",
    "news_count": "新闻数量",
    # Fundamental
    "pe_ratio": "市盈率",
    "pb_ratio": "市净率",
    "roe": "净资产收益率",
}


def print_section(title: str, width: int = 60) -> None:
    """Print a section header."""
    print(f"\n{'=' * width}")
    print(f" {title}")
    print("=" * width)


def print_feature_importance(model: StockTradingModel, top_n: int = 10) -> None:
    """Print top feature importances with descriptions."""
    try:
        importance = model.get_feature_importance()
        print_section(f" TOP {top_n} FEATURE IMPORTANCE")
        print()
        top_features = importance.head(top_n)
        for _, row in top_features.iterrows():
            feature_name = row["feature"]
            importance_val = row["importance"]
            bar = "█" * int(importance_val / 2)
            description = FEATURE_DESCRIPTIONS.get(feature_name, "")
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
    """Print core stock data including support/resistance levels and key indicators."""
    if df.empty:
        return

    history_df = (
        full_history_df
        if full_history_df is not None and not full_history_df.empty
        else df
    )
    latest = df.iloc[-1]

    print_section(" STOCK CORE DATA")

    # Price Info
    print(f"\n  === Price Info ===")
    print(f"  Current Price : {latest.get('close', 'N/A'):.2f}")
    print(f"  Open          : {latest.get('open', 'N/A'):.2f}")
    print(f"  High          : {latest.get('high', 'N/A'):.2f}")
    print(f"  Low           : {latest.get('low', 'N/A'):.2f}")
    print(f"  Volume        : {latest.get('volume', 0):,.0f}")

    # Moving Averages
    print(f"\n  === Moving Averages ===")
    for period in [5, 10, 20, 60]:
        ma_col = f"ma_{period}"
        if ma_col in df.columns:
            ma_val = latest.get(ma_col)
            if pd.notna(ma_val):
                current_price = latest.get("close", 0)
                pct_change = (
                    ((current_price - ma_val) / ma_val * 100) if ma_val != 0 else 0
                )
                direction = "↑" if pct_change > 0 else "↓"
                print(
                    f"  MA{period:<10} : {ma_val:.2f} ({direction}{abs(pct_change):.1f}%)"
                )

    # Key Indicators
    print(f"\n  === Key Indicators ===")
    if "rsi" in df.columns:
        rsi = latest.get("rsi", 0)
        rsi_label = "超买" if rsi > 70 else "超卖" if rsi < 30 else "中性"
        print(f"  RSI (14)       : {rsi:.1f} ({rsi_label})")

    if "macd" in df.columns:
        macd = latest.get("macd", 0)
        macd_label = "多头" if macd > 0 else "空头"
        print(f"  MACD           : {macd:.4f} ({macd_label})")

    if "atr" in df.columns:
        atr = latest.get("atr", 0)
        current_price = latest.get("close", 1)
        atr_pct = (atr / current_price * 100) if current_price else 0
        print(f"  ATR (14)       : {atr:.2f} ({atr_pct:.1f}%)")

    # Recent Returns
    print(f"\n  === Recent Returns ===")
    for period in [5, 10, 20]:
        mom_col = f"momentum_{period}"
        if mom_col in df.columns:
            ret = latest.get(mom_col)
            if pd.notna(ret):
                print(f"  {period}d Return     : {ret * 100:+.2f}%")

    # Sentiment
    print(f"\n  === Market Sentiment ===")
    sentiment_score = latest.get("sentiment_score", 0)
    news_count = latest.get("news_count", 0)
    if pd.notna(sentiment_score):
        sentiment_label = (
            "积极"
            if sentiment_score > 0.3
            else "消极"
            if sentiment_score < -0.3
            else "中性"
        )
        sentiment_emoji = (
            "🟢" if sentiment_score > 0.3 else "🔴" if sentiment_score < -0.3 else "⚪"
        )
        print(f"  Sentiment Score : {sentiment_score:.3f} {sentiment_emoji}")
        print(f"  Sentiment Label : {sentiment_label}")
        print(f"  News Count      : {news_count:.0f}")
    else:
        print(f"  情绪数据不可用")

    # Historical Performance
    print(f"\n  === Historical Performance ===")
    if len(history_df) >= 20:
        last_20 = history_df.tail(20)
        start_price = last_20["close"].iloc[0]
        end_price = last_20["close"].iloc[-1]
        perf_20d = (end_price - start_price) / start_price * 100

        if "returns" in last_20.columns:
            daily_vol = last_20["returns"].std() * 100
            annual_vol = daily_vol * (252**0.5)
        else:
            daily_vol = 0
            annual_vol = 0

        rolling_max = last_20["close"].cummax()
        drawdown = ((last_20["close"] - rolling_max) / rolling_max).min() * 100

        if "returns" in last_20.columns:
            win_rate = (last_20["returns"] > 0).sum() / 20 * 100
        else:
            win_rate = 0

        print(f"  20日涨跌幅     : {perf_20d:+.2f}%")
        print(f"  20日波动率     : {daily_vol:.2f}% (日) / {annual_vol:.1f}% (年化)")
        print(f"  20日最大回撤   : {drawdown:.2f}%")
        print(f"  20日胜率       : {win_rate:.0f}%")
    else:
        print(f"  数据不足20天")


def print_backtest_results(results_df: pd.DataFrame) -> None:
    """Print backtest comparison results."""
    print_section(" BACKTEST RESULTS")
    print()
    print(results_df.to_string(index=False))


def print_strategy_comparison(decisions: Dict[str, Dict[str, Any]]) -> None:
    """Print strategy comparison table."""
    print_section(" ALL STRATEGY DECISIONS")

    for name, data in decisions.items():
        print(f"\n  {name}:")
        print(f"    Decision  : {data.get('action', 'N/A')}")
        print(f"    Confidence: {data.get('confidence', 0):.2f}")
        print(f"    Return    : {data.get('return', 'N/A')}")
        print(f"    Sharpe    : {data.get('sharpe', 'N/A')}")
        print(f"    Score     : {data.get('score', 0):.3f}")


def print_final_recommendation(
    action: str,
    confidence: float,
    reason: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Print final trading recommendation."""
    print_section(" FINAL RECOMMENDATION")

    action_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}.get(action, "❓")
    print(f"\n  {action_emoji} {action} (置信度: {confidence:.2f})")
    print(f"\n  Reason: {reason}")

    if details:
        print(f"\n  Details:")
        for key, value in details.items():
            print(f"    {key}: {value}")
