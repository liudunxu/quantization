"""技术信号生成器 - 基于技术指标生成交易信号"""

import logging
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TechnicalSignalGenerator:
    """技术信号生成器

    基于多种技术指标生成买入/卖出信号，并提供信号解释。
    """

    def __init__(self):
        self.signals = []
        self.explanations = []

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析技术指标并生成信号

        Args:
            df: 包含技术指标的 DataFrame

        Returns:
            包含信号、解释和置信度的字典
        """
        if df.empty:
            return {
                "direction": "NEUTRAL",
                "confidence": 0.0,
                "signals": [],
                "bullish_factors": [],
                "bearish_factors": [],
                "neutral_factors": [],
            }

        latest = df.iloc[-1]
        self.signals = []
        self.explanations = []

        bullish_factors = []
        bearish_factors = []
        neutral_factors = []

        # 1. 均线信号
        ma_signal, ma_explanation = self._check_moving_averages(latest, df)
        if ma_signal == "BULLISH":
            bullish_factors.append(ma_explanation)
        elif ma_signal == "BEARISH":
            bearish_factors.append(ma_explanation)
        else:
            neutral_factors.append(ma_explanation)

        # 2. RSI信号
        rsi_signal, rsi_explanation = self._check_rsi(latest)
        if rsi_signal == "BULLISH":
            bullish_factors.append(rsi_explanation)
        elif rsi_signal == "BEARISH":
            bearish_factors.append(rsi_explanation)
        else:
            neutral_factors.append(rsi_explanation)

        # 3. MACD信号
        macd_signal, macd_explanation = self._check_macd(latest, df)
        if macd_signal == "BULLISH":
            bullish_factors.append(macd_explanation)
        elif macd_signal == "BEARISH":
            bearish_factors.append(macd_explanation)
        else:
            neutral_factors.append(macd_explanation)

        # 4. 布林带信号
        bb_signal, bb_explanation = self._check_bollinger_bands(latest)
        if bb_signal == "BULLISH":
            bullish_factors.append(bb_explanation)
        elif bb_signal == "BEARISH":
            bearish_factors.append(bb_explanation)
        else:
            neutral_factors.append(bb_explanation)

        # 5. 成交量信号
        vol_signal, vol_explanation = self._check_volume(latest, df)
        if vol_signal == "BULLISH":
            bullish_factors.append(vol_explanation)
        elif vol_signal == "BEARISH":
            bearish_factors.append(vol_explanation)
        else:
            neutral_factors.append(vol_explanation)

        # 6. 动量信号
        mom_signal, mom_explanation = self._check_momentum(latest)
        if mom_signal == "BULLISH":
            bullish_factors.append(mom_explanation)
        elif mom_signal == "BEARISH":
            bearish_factors.append(mom_explanation)
        else:
            neutral_factors.append(mom_explanation)

        # 7. KDJ信号
        kdj_signal, kdj_explanation = self._check_kdj(latest)
        if kdj_signal == "BULLISH":
            bullish_factors.append(kdj_explanation)
        elif kdj_signal == "BEARISH":
            bearish_factors.append(kdj_explanation)
        else:
            neutral_factors.append(kdj_explanation)

        # 8. ATR波动率信号
        atr_signal, atr_explanation = self._check_atr(latest, df)
        if atr_signal == "BULLISH":
            bullish_factors.append(atr_explanation)
        elif atr_signal == "BEARISH":
            bearish_factors.append(atr_explanation)
        else:
            neutral_factors.append(atr_explanation)

        # 综合判断
        bullish_count = len(bullish_factors)
        bearish_count = len(bearish_factors)
        total_signals = bullish_count + bearish_count + len(neutral_factors)

        if bullish_count > bearish_count:
            direction = "UP"
            confidence = bullish_count / total_signals if total_signals > 0 else 0.5
        elif bearish_count > bullish_count:
            direction = "DOWN"
            confidence = bearish_count / total_signals if total_signals > 0 else 0.5
        else:
            direction = "NEUTRAL"
            confidence = 0.5

        # 增强置信度计算
        signal_strength = abs(bullish_count - bearish_count)
        if signal_strength >= 3:
            confidence = min(confidence + 0.15, 0.95)
        elif signal_strength >= 2:
            confidence = min(confidence + 0.10, 0.90)

        return {
            "direction": direction,
            "confidence": confidence,
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "neutral_count": len(neutral_factors),
            "bullish_factors": bullish_factors,
            "bearish_factors": bearish_factors,
            "neutral_factors": neutral_factors,
            "signal_strength": signal_strength,
        }

    def _check_moving_averages(
        self, latest: pd.Series, df: pd.DataFrame
    ) -> Tuple[str, str]:
        """检查均线信号"""
        price = latest.get("close", 0)
        ma5 = latest.get("ma_5", None)
        ma10 = latest.get("ma_10", None)
        ma20 = latest.get("ma_20", None)
        ma60 = latest.get("ma_60", None)

        if ma5 is None or ma20 is None:
            return "NEUTRAL", "均线数据不足"

        # 价格在均线上方
        above_ma5 = price > ma5 if ma5 else False
        above_ma10 = price > ma10 if ma10 else False
        above_ma20 = price > ma20 if ma20 else False

        # 均线排列
        ma_bullish = (ma5 > ma10 > ma20) if (ma5 and ma10 and ma20) else False
        ma_bearish = (ma5 < ma10 < ma20) if (ma5 and ma10 and ma20) else False

        if ma_bullish and above_ma5:
            return (
                "BULLISH",
                f"均线多头排列 (MA5>{ma5:.2f}>MA10>{ma10:.2f}>MA20>{ma20:.2f})",
            )
        elif ma_bearish and not above_ma20:
            return (
                "BEARISH",
                f"均线空头排列 (MA5<{ma5:.2f}<MA10<{ma10:.2f}<MA20<{ma20:.2f})",
            )
        elif above_ma20:
            return "BULLISH", f"价格在MA20上方 ({price:.2f} > {ma20:.2f})"
        elif not above_ma20:
            return "BEARISH", f"价格跌破MA20 ({price:.2f} < {ma20:.2f})"
        else:
            return "NEUTRAL", "均线系统中性"

    def _check_rsi(self, latest: pd.Series) -> Tuple[str, str]:
        """检查RSI信号"""
        rsi = latest.get("rsi", None)
        if rsi is None:
            return "NEUTRAL", "RSI数据不足"

        if rsi < 30:
            return "BULLISH", f"RSI超卖 ({rsi:.1f} < 30)"
        elif rsi > 70:
            return "BEARISH", f"RSI超买 ({rsi:.1f} > 70)"
        elif rsi < 40:
            return "BULLISH", f"RSI偏弱有反弹机会 ({rsi:.1f})"
        elif rsi > 60:
            return "BEARISH", f"RSI偏强需谨慎 ({rsi:.1f})"
        else:
            return "NEUTRAL", f"RSI中性区域 ({rsi:.1f})"

    def _check_macd(self, latest: pd.Series, df: pd.DataFrame) -> Tuple[str, str]:
        """检查MACD信号"""
        macd = latest.get("macd", None)
        signal = latest.get("macd_signal", None)
        hist = latest.get("macd_hist", None)

        if macd is None or signal is None:
            return "NEUTRAL", "MACD数据不足"

        # 金叉死叉
        if len(df) >= 2:
            prev_hist = df.iloc[-2].get("macd_hist", 0)
            curr_hist = hist if hist else 0

            if prev_hist < 0 and curr_hist > 0:
                return "BULLISH", f"MACD金叉 (柱状图由负转正)"
            elif prev_hist > 0 and curr_hist < 0:
                return "BEARISH", f"MACD死叉 (柱状图由正转负)"

        # MACD方向
        if hist is not None:
            if hist > 0 and macd > signal:
                return "BULLISH", f"MACD多头 (DIF>{macd:.4f}, DEA>{signal:.4f})"
            elif hist < 0 and macd < signal:
                return "BEARISH", f"MACD空头 (DIF<{macd:.4f}, DEA<{signal:.4f})"

        return "NEUTRAL", "MACD中性"

    def _check_bollinger_bands(self, latest: pd.Series) -> Tuple[str, str]:
        """检查布林带信号"""
        price = latest.get("close", 0)
        bb_upper = latest.get("bb_upper", None)
        bb_lower = latest.get("bb_lower", None)
        bb_position = latest.get("bb_position", None)

        if bb_upper is None or bb_lower is None:
            return "NEUTRAL", "布林带数据不足"

        if price <= bb_lower:
            return "BULLISH", f"价格触及布林带下轨 ({price:.2f} <= {bb_lower:.2f})"
        elif price >= bb_upper:
            return "BEARISH", f"价格触及布林带上轨 ({price:.2f} >= {bb_upper:.2f})"
        elif bb_position is not None:
            if bb_position < 0.2:
                return "BULLISH", f"价格位于布林带低位 ({bb_position:.1%})"
            elif bb_position > 0.8:
                return "BEARISH", f"价格位于布林带高位 ({bb_position:.1%})"

        return "NEUTRAL", "布林带位置中性"

    def _check_volume(self, latest: pd.Series, df: pd.DataFrame) -> Tuple[str, str]:
        """检查成交量信号"""
        volume_ratio = latest.get("volume_ratio", None)
        price_change = latest.get("returns", 0) if "returns" in latest else 0

        if volume_ratio is None:
            return "NEUTRAL", "成交量数据不足"

        if volume_ratio > 2.0 and price_change > 0:
            return (
                "BULLISH",
                f"放量上涨 (量比{volume_ratio:.1f}，涨幅{price_change:.1%})",
            )
        elif volume_ratio > 2.0 and price_change < 0:
            return (
                "BEARISH",
                f"放量下跌 (量比{volume_ratio:.1f}，跌幅{abs(price_change):.1%})",
            )
        elif volume_ratio < 0.5 and price_change < 0:
            return "BULLISH", f"缩量下跌，抛压减轻 (量比{volume_ratio:.1f})"
        elif volume_ratio > 1.5:
            return "BULLISH", f"成交量放大 (量比{volume_ratio:.1f})"

        return "NEUTRAL", f"成交量正常 (量比{volume_ratio:.1f})"

    def _check_momentum(self, latest: pd.Series) -> Tuple[str, str]:
        """检查动量信号"""
        momentum_5 = latest.get("momentum_5", None)
        momentum_10 = latest.get("momentum_10", None)
        momentum_20 = latest.get("momentum_20", None)

        if momentum_5 is None:
            return "NEUTRAL", "动量数据不足"

        if momentum_5 is not None and momentum_5 > 0.03:
            return "BULLISH", f"5日动量强劲 ({momentum_5:.1%})"
        elif momentum_5 is not None and momentum_5 < -0.03:
            return "BEARISH", f"5日动量疲弱 ({momentum_5:.1%})"
        elif momentum_10 is not None and momentum_20 is not None:
            if momentum_5 > momentum_10 > momentum_20:
                return "BULLISH", "动量加速上升"
            elif momentum_5 < momentum_10 < momentum_20:
                return "BEARISH", "动量加速下降"

        return "NEUTRAL", "动量中性"

    def _check_kdj(self, latest: pd.Series) -> Tuple[str, str]:
        """检查KDJ信号"""
        k = latest.get("stoch_k", None)
        d = latest.get("stoch_d", None)

        if k is None or d is None:
            return "NEUTRAL", "KDJ数据不足"

        if k < 20 and d < 20:
            return "BULLISH", f"KDJ超卖 (K={k:.1f}, D={d:.1f})"
        elif k > 80 and d > 80:
            return "BEARISH", f"KDJ超买 (K={k:.1f}, D={d:.1f})"
        elif k > d and k < 50:
            return "BULLISH", f"KDJ金叉向上 (K={k:.1f} > D={d:.1f})"
        elif k < d and k > 50:
            return "BEARISH", f"KDJ死叉向下 (K={k:.1f} < D={d:.1f})"

        return "NEUTRAL", f"KDJ中性 (K={k:.1f}, D={d:.1f})"

    def _check_atr(self, latest: pd.Series, df: pd.DataFrame) -> Tuple[str, str]:
        """检查ATR波动率信号"""
        atr = latest.get("atr", None)
        price = latest.get("close", 0)

        if atr is None or price == 0:
            return "NEUTRAL", "ATR数据不足"

        atr_pct = atr / price * 100

        if atr_pct < 2:
            return "NEUTRAL", f"低波动率 (ATR {atr_pct:.1f}%)，等待突破"
        elif atr_pct > 5:
            return "BEARISH", f"高波动率 (ATR {atr_pct:.1f}%)，风险较高"

        return "NEUTRAL", f"波动率正常 (ATR {atr_pct:.1f}%)"
