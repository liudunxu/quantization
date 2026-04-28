"""技术信号生成器 - 基于技术指标生成交易信号"""

import logging
from typing import Any, Dict, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class TechnicalSignalGenerator:
    """技术信号生成器

    基于多种技术指标生成买入/卖出信号，并提供信号解释。
    """

    def __init__(self):
        """Initialize TechnicalSignalsAnalyzer."""
        self.signals = []
        self.explanations = []

    def analyze(self, df: pd.DataFrame, fast_mode: bool = False) -> Dict[str, Any]:
        """分析技术指标并生成信号

        Args:
            df: 包含技术指标的 DataFrame
            fast_mode: 极速模式，只检查核心指标以减少 CPU 开销

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

        if not fast_mode:
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

            # 9. ADX趋势强度信号
            adx_signal, adx_explanation = self._check_adx(latest)
            if adx_signal == "BULLISH":
                bullish_factors.append(adx_explanation)
            elif adx_signal == "BEARISH":
                bearish_factors.append(adx_explanation)
            else:
                neutral_factors.append(adx_explanation)

            # 10. MFI资金流信号
            mfi_signal, mfi_explanation = self._check_mfi(latest)
            if mfi_signal == "BULLISH":
                bullish_factors.append(mfi_explanation)
            elif mfi_signal == "BEARISH":
                bearish_factors.append(mfi_explanation)
            else:
                neutral_factors.append(mfi_explanation)

            # 11. CCI信号
            cci_signal, cci_explanation = self._check_cci(latest)
            if cci_signal == "BULLISH":
                bullish_factors.append(cci_explanation)
            elif cci_signal == "BEARISH":
                bearish_factors.append(cci_explanation)
            else:
                neutral_factors.append(cci_explanation)

            # 12. DMI信号
            dmi_signal, dmi_explanation = self._check_dmi(latest)
            if dmi_signal == "BULLISH":
                bullish_factors.append(dmi_explanation)
            elif dmi_signal == "BEARISH":
                bearish_factors.append(dmi_explanation)
            else:
                neutral_factors.append(dmi_explanation)

            # 13. RSI背离信号
            rsi_div_signal, rsi_div_explanation = self._check_rsi_divergence(latest, df)
            if rsi_div_signal == "BULLISH":
                bullish_factors.append(rsi_div_explanation)
            elif rsi_div_signal == "BEARISH":
                bearish_factors.append(rsi_div_explanation)
            else:
                neutral_factors.append(rsi_div_explanation)

            # 14. 量价背离信号
            vp_div_signal, vp_div_explanation = self._check_volume_price_divergence(
                latest, df
            )
            if vp_div_signal == "BULLISH":
                bullish_factors.append(vp_div_explanation)
            elif vp_div_signal == "BEARISH":
                bearish_factors.append(vp_div_explanation)
            else:
                neutral_factors.append(vp_div_explanation)

            # 15. Ichimoku云信号
            ichimoku_signal, ichimoku_explanation = self._check_ichimoku(latest)
            if ichimoku_signal == "BULLISH":
                bullish_factors.append(ichimoku_explanation)
            elif ichimoku_signal == "BEARISH":
                bearish_factors.append(ichimoku_explanation)
            else:
                neutral_factors.append(ichimoku_explanation)

            # 16. Williams %R信号
            willr_signal, willr_explanation = self._check_williams_r(latest)
            if willr_signal == "BULLISH":
                bullish_factors.append(willr_explanation)
            elif willr_signal == "BEARISH":
                bearish_factors.append(willr_explanation)
            else:
                neutral_factors.append(willr_explanation)

            # 17. OBV量能信号
            obv_signal, obv_explanation = self._check_obv(latest, df)
            if obv_signal == "BULLISH":
                bullish_factors.append(obv_explanation)
            elif obv_signal == "BEARISH":
                bearish_factors.append(obv_explanation)
            else:
                neutral_factors.append(obv_explanation)

            # 18. 连续涨跌信号
            streak_signal, streak_explanation = self._check_streak(latest)
            if streak_signal == "BULLISH":
                bullish_factors.append(streak_explanation)
            elif streak_signal == "BEARISH":
                bearish_factors.append(streak_explanation)
            else:
                neutral_factors.append(streak_explanation)

        # 综合判断
        bullish_count = len(bullish_factors)
        bearish_count = len(bearish_factors)
        total_signals = bullish_count + bearish_count + len(neutral_factors)

        if bullish_count > bearish_count:
            direction = "UP"
            confidence = (
                0.5 + (bullish_count - bearish_count) / max(total_signals, 1) * 0.5
            )
        elif bearish_count > bullish_count:
            direction = "DOWN"
            confidence = (
                0.5 + (bearish_count - bullish_count) / max(total_signals, 1) * 0.5
            )
        else:
            if bullish_factors:
                direction = "UP"
                confidence = 0.52
            elif bearish_factors:
                direction = "DOWN"
                confidence = 0.52
            else:
                direction = "NEUTRAL"
                confidence = 0.45

        signal_strength = abs(bullish_count - bearish_count)
        if signal_strength >= 3:
            confidence = min(confidence + 0.18, 0.95)
        elif signal_strength >= 2:
            confidence = min(confidence + 0.12, 0.90)
        elif signal_strength >= 1:
            confidence = min(confidence + 0.06, 0.85)

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
        latest.get("ma_60", None)

        if ma5 is None or ma20 is None:
            return "NEUTRAL", "均线数据不足"

        # 价格在均线上方
        above_ma5 = price > ma5 if ma5 else False
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
        """检查RSI信号 - 优化阈值"""
        rsi = latest.get("rsi", None)
        if rsi is None:
            return "NEUTRAL", "RSI数据不足"

        if rsi < 25:
            return "BULLISH", f"RSI极度超卖 ({rsi:.1f}<25)，强反转信号"
        elif rsi < 35:
            return "BULLISH", f"RSI超卖 ({rsi:.1f}<35)，反弹概率大"
        elif rsi > 75:
            return "BEARISH", f"RSI极度超买 ({rsi:.1f}>75)，回调风险高"
        elif rsi > 65:
            return "BEARISH", f"RSI超买 ({rsi:.1f}>65)，需谨慎"
        elif rsi < 45:
            return "BULLISH", f"RSI偏低有反弹机会 ({rsi:.1f})"
        elif rsi > 55:
            return "BEARISH", f"RSI偏强注意风险 ({rsi:.1f})"
        else:
            return "NEUTRAL", f"RSI中性区域 ({rsi:.1f})"

    def _check_macd(self, latest: pd.Series, df: pd.DataFrame) -> Tuple[str, str]:
        """检查MACD信号 - 增强背离检测"""
        macd = latest.get("macd", None)
        signal = latest.get("macd_signal", None)
        hist = latest.get("macd_hist", None)

        if macd is None or signal is None:
            return "NEUTRAL", "MACD数据不足"

        # 金叉死叉检测 (检查最近2天)
        if len(df) >= 2:
            prev_hist = df.iloc[-2].get("macd_hist", 0)
            curr_hist = hist if hist else 0

            if prev_hist < 0 and curr_hist > 0:
                return "BULLISH", "MACD金叉 (柱状图由负转正)"
            elif prev_hist > 0 and curr_hist < 0:
                return "BEARISH", "MACD死叉 (柱状图由正转负)"

        # 柱状图连续放大检测 (趋势确认)
        if len(df) >= 3 and hist is not None:
            hist_2d = df.iloc[-2].get("macd_hist", 0)
            hist_3d = df.iloc[-3].get("macd_hist", 0)
            if not pd.isna(hist_2d) and not pd.isna(hist_3d):
                if hist > 0 and hist > hist_2d > hist_3d:
                    return "BULLISH", f"MACD柱状图连续放大 (DIF={macd:.4f})"
                elif hist < 0 and hist < hist_2d < hist_3d:
                    return "BEARISH", f"MACD柱状图连续放大 (DIF={macd:.4f})"

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
        """检查成交量信号 - 优化阈值"""
        volume_ratio = latest.get("volume_ratio", None)
        price_change = latest.get("returns", 0) if "returns" in latest else 0

        if volume_ratio is None:
            return "NEUTRAL", "成交量数据不足"

        if volume_ratio > 2.5 and price_change > 0.01:
            return (
                "BULLISH",
                f"放量大涨 (量比{volume_ratio:.1f}，涨幅{price_change:.1%})",
            )
        elif volume_ratio > 2.0 and price_change < -0.01:
            return (
                "BEARISH",
                f"放量大跌 (量比{volume_ratio:.1f}，跌幅{abs(price_change):.1%})",
            )
        elif volume_ratio < 0.5 and price_change < 0:
            return "BULLISH", f"缩量下跌，抛压减轻 (量比{volume_ratio:.1f})"
        elif volume_ratio > 1.5 and price_change > 0:
            return (
                "BULLISH",
                f"温和放量上涨 (量比{volume_ratio:.1f}，涨幅{price_change:.1%})",
            )
        elif volume_ratio < 0.3:
            return "NEUTRAL", f"极度缩量，等待变盘 (量比{volume_ratio:.1f})"

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

    def _check_adx(self, latest: pd.Series) -> Tuple[str, str]:
        """检查ADX趋势强度信号"""
        adx = latest.get("adx", None)
        plus_di = latest.get("dmi_plus_di", None)
        minus_di = latest.get("dmi_minus_di", None)

        if adx is None:
            return "NEUTRAL", "ADX数据不足"

        if adx > 25:
            if plus_di is not None and minus_di is not None:
                if plus_di > minus_di:
                    return (
                        "BULLISH",
                        f"ADX趋势强劲 (ADX={adx:.1f}, +DI>{plus_di:.1f}>-DI>{minus_di:.1f})",
                    )
                elif minus_di > plus_di:
                    return (
                        "BEARISH",
                        f"ADX趋势强劲 (ADX={adx:.1f}, -DI>{minus_di:.1f}>+DI>{plus_di:.1f})",
                    )
            return "NEUTRAL", f"ADX趋势中等 (ADX={adx:.1f})"
        elif adx < 20:
            return "NEUTRAL", f"ADX无明确趋势 (ADX={adx:.1f})，观望"

        return "NEUTRAL", f"ADX趋势一般 (ADX={adx:.1f})"

    def _check_mfi(self, latest: pd.Series) -> Tuple[str, str]:
        """检查MFI资金流信号"""
        mfi = latest.get("mfi", None)

        if mfi is None:
            return "NEUTRAL", "MFI数据不足"

        if mfi < 20:
            return "BULLISH", f"MFI严重超卖 (MFI={mfi:.1f}<20)，资金流出极端"
        elif mfi < 30:
            return "BULLISH", f"MFI超卖 (MFI={mfi:.1f})，可能出现反弹"
        elif mfi > 80:
            return "BEARISH", f"MFI严重超买 (MFI={mfi:.1f}>80)，资金流入极端"
        elif mfi > 70:
            return "BEARISH", f"MFI超买 (MFI={mfi:.1f})，需警惕回调"

        return "NEUTRAL", f"MFI正常 (MFI={mfi:.1f})"

    def _check_cci(self, latest: pd.Series) -> Tuple[str, str]:
        """检查CCI信号"""
        cci = latest.get("cci", None)

        if cci is None:
            return "NEUTRAL", "CCI数据不足"

        if cci < -150:
            return "BULLISH", f"CCI极度超卖 (CCI={cci:.0f}<-150)，强反转信号"
        elif cci < -100:
            return "BULLISH", f"CCI超卖 (CCI={cci:.0f})，关注反弹"
        elif cci > 150:
            return "BEARISH", f"CCI极度超买 (CCI={cci:.0f}>150)，注意风险"
        elif cci > 100:
            return "BEARISH", f"CCI超买 (CCI={cci:.0f})，需谨慎"

        return "NEUTRAL", f"CCI正常 (CCI={cci:.0f})"

    def _check_dmi(self, latest: pd.Series) -> Tuple[str, str]:
        """检查DMI方向运动指标信号"""
        plus_di = latest.get("dmi_plus_di", None)
        minus_di = latest.get("dmi_minus_di", None)

        if plus_di is None or minus_di is None:
            return "NEUTRAL", "DMI数据不足"

        diff = plus_di - minus_di

        if diff > 10:
            return (
                "BULLISH",
                f"DMI多头优势 (+DI={plus_di:.1f}, -DI={minus_di:.1f}, 差值={diff:.1f})",
            )
        elif diff < -10:
            return (
                "BEARISH",
                f"DMI空头优势 (+DI={plus_di:.1f}, -DI={minus_di:.1f}, 差值={diff:.1f})",
            )
        elif diff > 0:
            return "BULLISH", f"DMI略偏多头 (+DI={plus_di:.1f}>-DI={minus_di:.1f})"
        elif diff < 0:
            return "BEARISH", f"DMI略偏空头 (+DI={plus_di:.1f}<-DI={minus_di:.1f})"

        return "NEUTRAL", f"DMI均衡 (+DI={plus_di:.1f}, -DI={minus_di:.1f})"

    def _check_rsi_divergence(
        self, latest: pd.Series, df: pd.DataFrame
    ) -> Tuple[str, str]:
        """检查RSI背离信号 - 价格新低但RSI未创新低"""
        if len(df) < 20:
            return "NEUTRAL", "RSI背离检测数据不足"

        rsi = latest.get("rsi", None)
        if rsi is None:
            return "NEUTRAL", "RSI数据不足"

        # 查找近20日的价格和RSI最低点
        recent = df.iloc[-20:]
        price_min_idx = recent["close"].idxmin()
        price_min = recent["close"].min()

        current_price = latest["close"]

        # 底背离：价格接近新低但RSI明显高于之前的RSI低点
        if current_price <= price_min * 1.02:  # 价格在低点附近2%内
            # 找到价格低点时的RSI
            rsi_at_low = recent.loc[price_min_idx, "rsi"]
            if not pd.isna(rsi_at_low) and rsi > rsi_at_low + 5:
                return (
                    "BULLISH",
                    f"RSI底背离 (RSI={rsi:.1f} > 低点RSI={rsi_at_low:.1f})",
                )

        # 顶背离：价格接近新高但RSI未创新高
        price_max_idx = recent["close"].idxmax()
        price_max = recent["close"].max()

        if current_price >= price_max * 0.98:  # 价格在高点附近2%内
            rsi_at_high = recent.loc[price_max_idx, "rsi"]
            if not pd.isna(rsi_at_high) and rsi < rsi_at_high - 5:
                return (
                    "BEARISH",
                    f"RSI顶背离 (RSI={rsi:.1f} < 高点RSI={rsi_at_high:.1f})",
                )

        return "NEUTRAL", f"RSI无明显背离 (RSI={rsi:.1f})"

    def _check_volume_price_divergence(
        self, latest: pd.Series, df: pd.DataFrame
    ) -> Tuple[str, str]:
        """检查量价背离信号 - 价格涨但量缩，或价格跌但量缩"""
        if len(df) < 5:
            return "NEUTRAL", "量价背离检测数据不足"

        price_change_5d = (latest["close"] - df["close"].iloc[-5]) / df["close"].iloc[
            -5
        ]
        vol_ratio = latest.get("volume_ratio", None)

        if vol_ratio is None:
            return "NEUTRAL", "成交量数据不足"

        # 价涨量缩 - 上涨乏力
        if price_change_5d > 0.02 and vol_ratio < 0.7:
            return (
                "BEARISH",
                f"量价背离(价涨量缩) - 5日涨{price_change_5d:.1%}但量比仅{vol_ratio:.1f}",
            )
        # 价跌量缩 - 抛压减轻
        elif price_change_5d < -0.02 and vol_ratio < 0.7:
            return (
                "BULLISH",
                f"量价背离(价跌量缩) - 5日跌{abs(price_change_5d):.1%}且量比仅{vol_ratio:.1f}",
            )
        # 价跌量增 - 主力出逃
        elif price_change_5d < -0.02 and vol_ratio > 1.5:
            return (
                "BEARISH",
                f"放量下跌 - 5日跌{abs(price_change_5d):.1%}且量比{vol_ratio:.1f}",
            )
        # 价涨量增 - 健康上涨
        elif price_change_5d > 0.02 and vol_ratio > 1.2:
            return (
                "BULLISH",
                f"价量齐升 - 5日涨{price_change_5d:.1%}且量比{vol_ratio:.1f}",
            )

        return "NEUTRAL", "量价关系正常"

    def _check_ichimoku(self, latest: pd.Series) -> Tuple[str, str]:
        """检查Ichimoku云信号 - 使用已有指标模拟"""
        # Ichimoku参数: 转换线(9), 基准线(26), 先行带A(26), 先行带B(52)
        # 使用均线近似: MA5 ~ 转换线, MA20 ~ 基准线
        price = latest.get("close", 0)
        ma5 = latest.get("ma_5", None)
        ma10 = latest.get("ma_10", None)
        ma20 = latest.get("ma_20", None)
        ma60 = latest.get("ma_60", None)

        if ma5 is None or ma20 is None:
            return "NEUTRAL", "Ichimoku数据不足"

        # 价格在云层上方 (多头)
        if ma10 is not None and ma60 is not None:
            cloud_top = max(ma10, ma60)  # 模拟先行带上沿
            cloud_bottom = min(ma10, ma60)  # 模拟先行带下沿

            if price > cloud_top and ma5 > ma20:
                return "BULLISH", f"价格在云层上方 (价格>{cloud_top:.2f}>云层)"
            elif price < cloud_bottom and ma5 < ma20:
                return "BEARISH", f"价格在云层下方 (价格<{cloud_bottom:.2f}<云层)"

        # 转换线与基准线交叉
        if ma5 > ma20:
            return "BULLISH", f"Ichimoku多头 (转换线>{ma5:.2f}>基准线>{ma20:.2f})"
        elif ma5 < ma20:
            return "BEARISH", f"Ichimoku空头 (转换线<{ma5:.2f}<基准线<{ma20:.2f})"

        return "NEUTRAL", "Ichimoku中性"

    def _check_williams_r(self, latest: pd.Series) -> Tuple[str, str]:
        """检查Williams %R信号 - 使用RSI和KDJ近似"""
        # Williams %R: -100到0, 超卖<-80, 超买>-20
        # 使用stoch_k近似: willr = stoch_k - 100
        stoch_k = latest.get("stoch_k", None)
        if stoch_k is None:
            return "NEUTRAL", "Williams %R数据不足"

        willr = stoch_k - 100  # 转换为Williams %R范围

        if willr < -80:
            return "BULLISH", f"Williams %R超卖 ({willr:.0f}<-80)"
        elif willr > -20:
            return "BEARISH", f"Williams %R超买 ({willr:.0f}>-20)"
        elif willr < -50:
            return "BULLISH", f"Williams %R偏弱 ({willr:.0f})"
        elif willr > -50:
            return "BEARISH", f"Williams %R偏强 ({willr:.0f})"

        return "NEUTRAL", f"Williams %R中性 ({willr:.0f})"

    def _check_obv(self, latest: pd.Series, df: pd.DataFrame) -> Tuple[str, str]:
        """检查OBV量能信号 - 通过A/D线判断"""
        ad_oscillator = latest.get("ad_oscillator", None)
        if ad_oscillator is None:
            return "NEUTRAL", "OBV数据不足"

        if pd.isna(ad_oscillator):
            return "NEUTRAL", "OBV数据不足"

        if ad_oscillator > 0:
            return "BULLISH", "OBV/A-D线向上 (资金流入)"
        elif ad_oscillator < 0:
            return "BEARISH", "OBV/A-D线向下 (资金流出)"

        return "NEUTRAL", "OBV/A-D线中性"

    def _check_streak(self, latest: pd.Series) -> Tuple[str, str]:
        """检查连续涨跌信号"""
        streak_up_3 = latest.get("streak_up_3", 0)
        streak_down_3 = latest.get("streak_down_3", 0)

        if streak_up_3 == 1:
            return "BEARISH", "连续3日上涨，可能回调"
        elif streak_down_3 == 1:
            return "BULLISH", "连续3日下跌，可能反弹"

        return "NEUTRAL", "无明显连续涨跌"
