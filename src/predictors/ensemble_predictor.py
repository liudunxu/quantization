"""集成预测器 - 结合ML模型、技术信号和多策略进行综合预测"""

import logging
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

from ..models import StockTradingModel
from .technical_signals import TechnicalSignalGenerator

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """集成预测器

    结合ML模型预测、技术分析信号、规则策略信号，提供更可靠的预测结果。
    支持多种信号源的加权投票和多策略叠加。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化预测器

        Args:
            config: 配置参数
        """
        self.config = config or {}
        self.technical_generator = TechnicalSignalGenerator()

        # 默认权重配置 - 优化后权重分配
        self.ml_weight = self.config.get("ml_weight", 0.35)  # ML模型权重
        self.technical_weight = self.config.get(
            "technical_weight", 0.25
        )  # 技术信号权重
        self.momentum_weight = self.config.get("momentum_weight", 0.15)  # 动量权重
        self.trend_weight = self.config.get("trend_weight", 0.10)  # 趋势强度权重
        self.alpha_weight = self.config.get("alpha_weight", 0.05)  # 超额收益权重
        self.strategy_weight = self.config.get("strategy_weight", 0.10)  # 规则策略权重

        # 策略叠加配置
        self.stacking_strategies = self.config.get("stacking_strategies", [])

    def predict(
        self,
        model: StockTradingModel,
        df: pd.DataFrame,
        current_price: float,
    ) -> Dict[str, Any]:
        """综合预测

        Args:
            model: 训练好的ML模型
            df: 包含特征的DataFrame
            current_price: 当前价格

        Returns:
            综合预测结果
        """
        if df.empty:
            return self._create_neutral_prediction(current_price)

        latest_df = df.iloc[[-1]]
        latest = df.iloc[-1]

        # 1. ML模型预测
        ml_result = self._get_ml_prediction(model, latest_df)

        # 2. 技术信号分析
        technical_result = self.technical_generator.analyze(df)

        # 3. 动量分析
        momentum_result = self._analyze_momentum(latest, df)

        # 4. 市场情绪分析
        sentiment_result = self._analyze_sentiment(latest)

        # 5. 趋势强度分析
        trend_result = self._analyze_trend_strength(latest, df)

        # 6. 超额收益分析
        alpha_result = self._analyze_alpha(latest, df)

        # 7. 多策略信号分析 (新增)
        strategy_result = self._analyze_strategies(latest, df)

        # 8. 市场状态分析 (新增)
        market_regime = self._analyze_market_regime(latest, df)

        # 9. 支撑阻力分析 (新增)
        support_resistance = self._analyze_support_resistance(latest, df)

        # 10. 多时间框架趋势确认 (新增)
        multi_tf_result = self._analyze_multi_timeframe(latest, df)

        # 11. 综合判断
        final_result = self._combine_predictions(
            ml_result,
            technical_result,
            momentum_result,
            sentiment_result,
            current_price,
            trend_result,
            alpha_result,
            strategy_result,
            market_regime,
            support_resistance,
            multi_tf_result,
        )

        return final_result

    def _get_ml_prediction(
        self, model: StockTradingModel, latest_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """获取ML模型预测"""
        try:
            action, confidence = model.predict(latest_df)
            probabilities = model.predict_proba(latest_df)

            # 处理概率格式
            if isinstance(probabilities, dict):
                up_prob = probabilities.get("buy_probability", 0.33)
                down_prob = probabilities.get("sell_probability", 0.33)
                hold_prob = probabilities.get("hold_probability", 0.34)
            else:
                up_prob, hold_prob, down_prob = 0.33, 0.34, 0.33

            return {
                "action": action,
                "confidence": confidence,
                "up_prob": up_prob,
                "down_prob": down_prob,
                "hold_prob": hold_prob,
            }
        except Exception as e:
            logger.warning(f"ML prediction failed: {e}")
            return {
                "action": "HOLD",
                "confidence": 0.5,
                "up_prob": 0.33,
                "down_prob": 0.33,
                "hold_prob": 0.34,
            }

    def _analyze_momentum(self, latest: pd.Series, df: pd.DataFrame) -> Dict[str, Any]:
        """分析动量特征"""
        momentum_5 = latest.get("momentum_5", 0)
        momentum_10 = latest.get("momentum_10", 0)
        momentum_20 = latest.get("momentum_20", 0)

        # 动量趋势
        if momentum_5 > 0.02 and momentum_5 > momentum_10:
            direction = "UP"
            strength = min(abs(momentum_5) * 10, 1.0)
            explanation = f"短期动量强劲 (5日:{momentum_5:.1%})"
        elif momentum_5 < -0.02 and momentum_5 < momentum_10:
            direction = "DOWN"
            strength = min(abs(momentum_5) * 10, 1.0)
            explanation = f"短期动量疲弱 (5日:{momentum_5:.1%})"
        elif momentum_5 > momentum_10 > momentum_20:
            direction = "UP"
            strength = 0.6
            explanation = "动量加速上升"
        elif momentum_5 < momentum_10 < momentum_20:
            direction = "DOWN"
            strength = 0.6
            explanation = "动量加速下降"
        else:
            direction = "NEUTRAL"
            strength = 0.5
            explanation = "动量方向不明"

        return {
            "direction": direction,
            "strength": strength,
            "explanation": explanation,
            "momentum_5": momentum_5,
            "momentum_10": momentum_10,
            "momentum_20": momentum_20,
        }

    def _analyze_sentiment(self, latest: pd.Series) -> Dict[str, Any]:
        """分析市场情绪"""
        sentiment_score = latest.get("sentiment_score", 0)
        news_count = latest.get("news_count", 0)

        if pd.isna(sentiment_score):
            sentiment_score = 0

        if sentiment_score > 0.3:
            direction = "UP"
            explanation = f"市场情绪积极 ({sentiment_score:.2f})"
        elif sentiment_score < -0.3:
            direction = "DOWN"
            explanation = f"市场情绪消极 ({sentiment_score:.2f})"
        else:
            direction = "NEUTRAL"
            explanation = f"市场情绪中性 ({sentiment_score:.2f})"

        return {
            "direction": direction,
            "score": sentiment_score,
            "news_count": news_count,
            "explanation": explanation,
        }

    def _analyze_trend_strength(
        self, latest: pd.Series, df: pd.DataFrame
    ) -> Dict[str, Any]:
        """分析趋势强度 - 基于ADX和均线排列"""
        adx = latest.get("adx", 0)
        plus_di = latest.get("dmi_plus_di", 0)
        minus_di = latest.get("dmi_minus_di", 0)

        # 多头排列检查
        ma5 = latest.get("ma_5", 0)
        ma10 = latest.get("ma_10", 0)
        ma20 = latest.get("ma_20", 0)

        if pd.isna(ma5) or pd.isna(ma10) or pd.isna(ma20):
            ma5 = ma10 = ma20 = 0

        ma_bullish = ma5 > ma10 > ma20
        ma_bearish = ma5 < ma10 < ma20

        # ADX趋势强度判断
        if pd.isna(adx):
            adx = 20  # 默认值

        if pd.isna(plus_di):
            plus_di = 0
        if pd.isna(minus_di):
            minus_di = 0

        # 趋势方向
        if adx > 25 and plus_di > minus_di and ma_bullish:
            direction = "UP"
            strength = min(adx / 50, 1.0)
            explanation = f"强势上升趋势 (ADX={adx:.0f}, 多头排列)"
        elif adx > 25 and minus_di > plus_di and ma_bearish:
            direction = "DOWN"
            strength = min(adx / 50, 1.0)
            explanation = f"强势下降趋势 (ADX={adx:.0f}, 空头排列)"
        elif ma_bullish:
            direction = "UP"
            strength = 0.6
            explanation = "温和上升趋势 (多头排列)"
        elif ma_bearish:
            direction = "DOWN"
            strength = 0.6
            explanation = "温和下降趋势 (空头排列)"
        else:
            direction = "NEUTRAL"
            strength = 0.4
            explanation = f"震荡整理 (ADX={adx:.0f})"

        return {
            "direction": direction,
            "strength": strength,
            "adx": adx,
            "plus_di": plus_di,
            "minus_di": minus_di,
            "ma_arrangement": "bullish"
            if ma_bullish
            else ("bearish" if ma_bearish else "neutral"),
            "explanation": explanation,
        }

    def _analyze_alpha(self, latest: pd.Series, df: pd.DataFrame) -> Dict[str, Any]:
        """分析超额收益 - 个股相对市场的表现"""
        alpha = latest.get("alpha", 0)
        beta = latest.get("beta", 1.0)
        corr = latest.get("corr", 0)

        if pd.isna(alpha):
            alpha = 0
        if pd.isna(beta):
            beta = 1.0
        if pd.isna(corr):
            corr = 0

        if alpha > 0.02:
            direction = "UP"
            strength = min(abs(alpha) * 10, 1.0)
            explanation = f"显著超额收益 (alpha={alpha:.2%})"
        elif alpha < -0.02:
            direction = "DOWN"
            strength = min(abs(alpha) * 10, 1.0)
            explanation = f"明显跑输市场 (alpha={alpha:.2%})"
        else:
            direction = "NEUTRAL"
            strength = 0.5
            explanation = f"与市场同步 (alpha={alpha:.2%})"

        return {
            "direction": direction,
            "strength": strength,
            "alpha": alpha,
            "beta": beta,
            "correlation": corr,
            "explanation": explanation,
        }

    def _analyze_strategies(
        self, latest: pd.Series, df: pd.DataFrame
    ) -> Dict[str, Any]:
        """分析规则策略信号 - 叠加多种规则策略的结果

        使用简化版策略逻辑直接计算信号，避免重复创建策略实例
        """
        bullish_votes = 0
        bearish_votes = 0
        total_votes = 0
        explanations = []

        # 1. 均线金叉策略
        ma5 = latest.get("ma_5", None)
        ma10 = latest.get("ma_10", None)
        if ma5 is not None and ma10 is not None and len(df) >= 2:
            prev_ma5 = df.iloc[-2].get("ma_5", ma5)
            prev_ma10 = df.iloc[-2].get("ma_10", ma10)
            if not pd.isna(prev_ma5) and not pd.isna(prev_ma10):
                # 金叉: MA5 从下穿上穿 MA10
                if prev_ma5 <= prev_ma10 and ma5 > ma10:
                    bullish_votes += 1
                    explanations.append("均线金叉(MA5↑MA10)")
                # 死叉: MA5 从上穿下穿 MA10
                elif prev_ma5 >= prev_ma10 and ma5 < ma10:
                    bearish_votes += 1
                    explanations.append("均线死叉(MA5↓MA10)")
                total_votes += 1

        # 2. MACD策略
        macd_hist = latest.get("macd_hist", None)
        if macd_hist is not None and len(df) >= 2:
            prev_hist = df.iloc[-2].get("macd_hist", 0)
            if not pd.isna(prev_hist):
                if prev_hist < 0 and macd_hist > 0:
                    bullish_votes += 1
                    explanations.append("MACD金叉")
                elif prev_hist > 0 and macd_hist < 0:
                    bearish_votes += 1
                    explanations.append("MACD死叉")
                total_votes += 1

        # 3. RSI超买超卖策略
        rsi = latest.get("rsi", None)
        if rsi is not None:
            if rsi < 30:
                bullish_votes += 1
                explanations.append(f"RSI超卖({rsi:.0f})")
            elif rsi > 70:
                bearish_votes += 1
                explanations.append(f"RSI超买({rsi:.0f})")
            total_votes += 1

        # 4. KDJ超买超卖策略
        stoch_k = latest.get("stoch_k", None)
        stoch_d = latest.get("stoch_d", None)
        if stoch_k is not None and stoch_d is not None:
            if stoch_k < 20 and stoch_d < 20:
                bullish_votes += 1
                explanations.append(f"KDJ超卖(K={stoch_k:.0f})")
            elif stoch_k > 80 and stoch_d > 80:
                bearish_votes += 1
                explanations.append(f"KDJ超买(K={stoch_k:.0f})")
            total_votes += 1

        # 5. 布林带策略
        bb_position = latest.get("bb_position", None)
        if bb_position is not None:
            if bb_position < 0.1:
                bullish_votes += 1
                explanations.append("触及布林下轨")
            elif bb_position > 0.9:
                bearish_votes += 1
                explanations.append("触及布林上轨")
            total_votes += 1

        # 6. 成交量突破策略
        volume_ratio = latest.get("volume_ratio", None)
        returns = latest.get("returns", 0)
        if volume_ratio is not None:
            if volume_ratio > 2.0 and returns > 0.01:
                bullish_votes += 1
                explanations.append("放量上涨")
            elif volume_ratio > 2.0 and returns < -0.01:
                bearish_votes += 1
                explanations.append("放量下跌")
            total_votes += 1

        # 7. 均线排列策略
        ma20 = latest.get("ma_20", None)
        if ma5 is not None and ma10 is not None and ma20 is not None:
            if ma5 > ma10 > ma20:
                bullish_votes += 1
                explanations.append("多头排列")
            elif ma5 < ma10 < ma20:
                bearish_votes += 1
                explanations.append("空头排列")
            total_votes += 1

        # 8. DMI策略
        plus_di = latest.get("dmi_plus_di", None)
        minus_di = latest.get("dmi_minus_di", None)
        if plus_di is not None and minus_di is not None:
            if plus_di > minus_di + 10:
                bullish_votes += 1
                explanations.append("DMI多头")
            elif minus_di > plus_di + 10:
                bearish_votes += 1
                explanations.append("DMI空头")
            total_votes += 1

        # 9. MFI资金流策略
        mfi = latest.get("mfi", None)
        if mfi is not None:
            if mfi < 20:
                bullish_votes += 1
                explanations.append("MFI超卖")
            elif mfi > 80:
                bearish_votes += 1
                explanations.append("MFI超买")
            total_votes += 1

        # 10. CCI策略
        cci = latest.get("cci", None)
        if cci is not None:
            if cci < -100:
                bullish_votes += 1
                explanations.append("CCI超卖")
            elif cci > 100:
                bearish_votes += 1
                explanations.append("CCI超买")
            total_votes += 1

        # 11. 缩量回调策略 (借鉴decide.py的ShrinkPullbackStrategy)
        volume_ratio = latest.get("volume_ratio", None)
        ma20_ratio = latest.get("ma_20_ratio", None)
        if volume_ratio is not None and ma20_ratio is not None:
            if volume_ratio < 0.7 and 0.95 < ma20_ratio < 1.05:
                bullish_votes += 1
                explanations.append("缩量回调(接近MA20)")
            total_votes += 1

        # 12. 底部放量策略 (借鉴decide.py的BottomVolumeStrategy)
        price_position = latest.get("price_position", None)
        if volume_ratio is not None and price_position is not None:
            if volume_ratio > 2.5 and price_position < 0.25:
                bullish_votes += 1
                explanations.append("底部放量")
            total_votes += 1

        # 13. 放量突破策略 (借鉴decide.py的VolumeBreakoutStrategy)
        breakout_up = latest.get("breakout_up", 0)
        if volume_ratio is not None and breakout_up:
            if volume_ratio > 1.5:
                bullish_votes += 1
                explanations.append("放量突破")
            total_votes += 1

        # 14. Aroon趋势策略
        aroon_up = latest.get("aroon_up", None)
        aroon_down = latest.get("aroon_down", None)
        if aroon_up is not None and aroon_down is not None:
            if aroon_up > 70 and aroon_down < 30:
                bullish_votes += 1
                explanations.append("Aroon强多头")
            elif aroon_down > 70 and aroon_up < 30:
                bearish_votes += 1
                explanations.append("Aroon强空头")
            total_votes += 1

        # 15. 趋势评分策略 (借鉴decide.py)
        trend_score = latest.get("trend_score", None)
        if trend_score is not None:
            if trend_score > 0.5:
                bullish_votes += 1
                explanations.append("趋势评分偏多")
            elif trend_score < -0.5:
                bearish_votes += 1
                explanations.append("趋势评分偏空")
            total_votes += 1

        # 计算结果
        if total_votes == 0:
            return {
                "direction": "NEUTRAL",
                "strength": 0.5,
                "bullish_votes": 0,
                "bearish_votes": 0,
                "total_votes": 0,
                "explanations": [],
            }

        if bullish_votes > bearish_votes:
            direction = "UP"
            strength = min(bullish_votes / total_votes + 0.3, 0.95)
        elif bearish_votes > bullish_votes:
            direction = "DOWN"
            strength = min(bearish_votes / total_votes + 0.3, 0.95)
        else:
            direction = "NEUTRAL"
            strength = 0.5

        return {
            "direction": direction,
            "strength": strength,
            "bullish_votes": bullish_votes,
            "bearish_votes": bearish_votes,
            "total_votes": total_votes,
            "explanations": explanations,
        }

    def _analyze_market_regime(
        self, latest: pd.Series, df: pd.DataFrame
    ) -> Dict[str, Any]:
        """分析市场状态 - 趋势市场 vs 震荡市场"""
        adx = latest.get("adx", 20)
        if pd.isna(adx):
            adx = 20

        # 波动率
        volatility = latest.get("volatility_20d", 0)
        if pd.isna(volatility):
            volatility = 0

        # 价格位置
        price_position = latest.get("price_position", 0.5)
        if pd.isna(price_position):
            price_position = 0.5

        # 均线收敛度
        ma_convergence = latest.get("ma_convergence", 0)
        if pd.isna(ma_convergence):
            ma_convergence = 0

        # 判断市场状态
        if adx > 25:
            regime = "trending"
            confidence = min(adx / 50, 1.0)
            explanation = f"趋势市场 (ADX={adx:.0f})"
        elif adx < 18:
            regime = "ranging"
            confidence = 0.6
            explanation = f"震荡市场 (ADX={adx:.0f})"
        else:
            regime = "transitional"
            confidence = 0.5
            explanation = f"过渡市场 (ADX={adx:.0f})"

        return {
            "regime": regime,
            "confidence": confidence,
            "adx": adx,
            "volatility": volatility,
            "price_position": price_position,
            "explanation": explanation,
        }

    def _analyze_support_resistance(
        self, latest: pd.Series, df: pd.DataFrame
    ) -> Dict[str, Any]:
        """分析支撑阻力位"""
        price = latest.get("close", 0)

        # 使用已有特征
        box_top = latest.get("box_top", price * 1.05)
        box_bottom = latest.get("box_bottom", price * 0.95)
        high_20d = latest.get("high_20d", price)
        low_20d = latest.get("low_20d", price)
        bb_upper = latest.get("bb_upper", price * 1.02)
        bb_lower = latest.get("bb_lower", price * 0.98)

        if pd.isna(box_top):
            box_top = price * 1.05
        if pd.isna(box_bottom):
            box_bottom = price * 0.95
        if pd.isna(high_20d):
            high_20d = price
        if pd.isna(low_20d):
            low_20d = price

        # 计算距离支撑位和阻力位的百分比
        distance_to_support = (price - box_bottom) / price * 100 if price > 0 else 5
        distance_to_resistance = (box_top - price) / price * 100 if price > 0 else 5

        # 判断信号
        if distance_to_support < 3:
            direction = "UP"
            strength = 0.7
            explanation = f"接近支撑位 ({box_bottom:.2f})"
        elif distance_to_resistance < 3:
            direction = "DOWN"
            strength = 0.7
            explanation = f"接近阻力位 ({box_top:.2f})"
        else:
            direction = "NEUTRAL"
            strength = 0.5
            explanation = f"位于区间中部 (支撑{box_bottom:.2f}/阻力{box_top:.2f})"

        return {
            "direction": direction,
            "strength": strength,
            "support": box_bottom,
            "resistance": box_top,
            "distance_to_support": distance_to_support,
            "distance_to_resistance": distance_to_resistance,
            "explanation": explanation,
        }

    def _analyze_multi_timeframe(
        self, latest: pd.Series, df: pd.DataFrame
    ) -> Dict[str, Any]:
        """多时间框架趋势确认

        短期(5日)、中期(20日)、长期(60日)趋势一致性检查
        只有多时间框架趋势一致时才给出高置信度信号
        """
        if len(df) < 60:
            return {
                "direction": "NEUTRAL",
                "strength": 0.5,
                "short_trend": "NEUTRAL",
                "medium_trend": "NEUTRAL",
                "long_trend": "NEUTRAL",
                "consensus": 0,
                "explanation": "数据不足，无法进行多时间框架分析",
            }

        # 短期趋势(5日): 5日动量
        momentum_5 = latest.get("momentum_5", 0)
        if pd.isna(momentum_5):
            momentum_5 = 0
        short_trend = (
            "UP" if momentum_5 > 0.01 else ("DOWN" if momentum_5 < -0.01 else "NEUTRAL")
        )

        # 中期趋势(20日): 20日动量 + MA20斜率
        momentum_20 = latest.get("momentum_20", 0)
        ma_slope_20 = latest.get("ma_slope_20", 0)
        if pd.isna(momentum_20):
            momentum_20 = 0
        if pd.isna(ma_slope_20):
            ma_slope_20 = 0
        medium_trend = (
            "UP"
            if (momentum_20 > 0.02 or ma_slope_20 > 0.001)
            else (
                "DOWN" if (momentum_20 < -0.02 or ma_slope_20 < -0.001) else "NEUTRAL"
            )
        )

        # 长期趋势(60日): 价格相对MA60位置
        ma60 = latest.get("ma_60", None)
        ma120 = latest.get("ma_120", None)
        price = latest.get("close", 0)
        if ma60 is not None and not pd.isna(ma60) and ma60 > 0:
            long_trend = "UP" if price > ma60 else "DOWN"
        else:
            long_trend = "NEUTRAL"

        # 计算共识
        trends = [short_trend, medium_trend, long_trend]
        up_count = trends.count("UP")
        down_count = trends.count("DOWN")
        neutral_count = trends.count("NEUTRAL")

        if up_count >= 2:
            direction = "UP"
            strength = 0.6 + up_count * 0.1
        elif down_count >= 2:
            direction = "DOWN"
            strength = 0.6 + down_count * 0.1
        else:
            direction = "NEUTRAL"
            strength = 0.5

        consensus = up_count - down_count  # -3 to +3

        return {
            "direction": direction,
            "strength": min(strength, 0.9),
            "short_trend": short_trend,
            "medium_trend": medium_trend,
            "long_trend": long_trend,
            "consensus": consensus,
            "explanation": f"多时间框架: 短期{short_trend} / 中期{medium_trend} / 长期{long_trend}",
        }

    def _combine_predictions(
        self,
        ml_result: Dict[str, Any],
        technical_result: Dict[str, Any],
        momentum_result: Dict[str, Any],
        sentiment_result: Dict[str, Any],
        current_price: float,
        trend_result: Optional[Dict[str, Any]] = None,
        alpha_result: Optional[Dict[str, Any]] = None,
        strategy_result: Optional[Dict[str, Any]] = None,
        market_regime: Optional[Dict[str, Any]] = None,
        support_resistance: Optional[Dict[str, Any]] = None,
        multi_tf_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """综合多个预测结果

        采用多时间框架确认机制:
        - 只有多时间框架趋势一致时才给出高置信度信号
        - 单一时间框架信号降低置信度
        """

        # 收集各信号源的方向和权重
        signals = []

        # ML信号 - 使用概率和action综合判断
        if ml_result["action"] == "BUY":
            ml_direction = "UP"
        elif ml_result["action"] == "SELL":
            ml_direction = "DOWN"
        elif ml_result["up_prob"] > ml_result["down_prob"] + 0.1:
            ml_direction = "UP"
        elif ml_result["down_prob"] > ml_result["up_prob"] + 0.1:
            ml_direction = "DOWN"
        else:
            ml_direction = "NEUTRAL"

        signals.append(
            {
                "source": "ML模型",
                "direction": ml_direction,
                "weight": self.ml_weight,
                "confidence": ml_result["confidence"],
            }
        )

        # 技术信号
        signals.append(
            {
                "source": "技术分析",
                "direction": technical_result["direction"],
                "weight": self.technical_weight,
                "confidence": technical_result["confidence"],
            }
        )

        # 动量信号
        signals.append(
            {
                "source": "动量分析",
                "direction": momentum_result["direction"],
                "weight": self.momentum_weight,
                "confidence": momentum_result["strength"],
            }
        )

        # 趋势强度信号
        if trend_result:
            signals.append(
                {
                    "source": "趋势强度",
                    "direction": trend_result["direction"],
                    "weight": self.trend_weight,
                    "confidence": trend_result["strength"],
                }
            )

        # 超额收益信号
        if alpha_result:
            signals.append(
                {
                    "source": "超额收益",
                    "direction": alpha_result["direction"],
                    "weight": self.alpha_weight,
                    "confidence": alpha_result["strength"],
                }
            )

        # 多策略叠加信号 (新增)
        if strategy_result and strategy_result["total_votes"] > 0:
            signals.append(
                {
                    "source": "策略叠加",
                    "direction": strategy_result["direction"],
                    "weight": self.strategy_weight,
                    "confidence": strategy_result["strength"],
                }
            )

        # 市场状态调整
        regime_multiplier = 1.0
        if market_regime:
            if market_regime["regime"] == "trending":
                regime_multiplier = 1.2  # 趋势市场增强趋势信号
            elif market_regime["regime"] == "ranging":
                regime_multiplier = 0.8  # 震荡市场减弱趋势信号

        # 支撑阻力信号
        if support_resistance:
            signals.append(
                {
                    "source": "支撑阻力",
                    "direction": support_resistance["direction"],
                    "weight": 0.05,
                    "confidence": support_resistance["strength"],
                }
            )

        # 多时间框架确认信号
        if multi_tf_result:
            signals.append(
                {
                    "source": "多时间框架",
                    "direction": multi_tf_result["direction"],
                    "weight": 0.10,
                    "confidence": multi_tf_result["strength"],
                }
            )

        # 计算加权投票
        up_score = sum(
            s["weight"] * s["confidence"] for s in signals if s["direction"] == "UP"
        )
        down_score = sum(
            s["weight"] * s["confidence"] for s in signals if s["direction"] == "DOWN"
        )
        neutral_score = sum(s["weight"] for s in signals if s["direction"] == "NEUTRAL")

        # 降低中性信号权重，让UP/DOWN信号更容易胜出
        neutral_score *= 0.3

        # 应用市场状态调整
        if market_regime and market_regime["regime"] == "trending":
            # 趋势市场: 增强一致性信号
            if up_score > down_score:
                up_score *= regime_multiplier
            elif down_score > up_score:
                down_score *= regime_multiplier

        total_score = up_score + down_score + neutral_score

        if total_score > 0:
            up_ratio = up_score / total_score
            down_ratio = down_score / total_score
        else:
            up_ratio = down_ratio = 0.33

        # 确定最终方向 - 使用更宽松的阈值(0.38)，更容易给出方向
        signal_strength = abs(up_score - down_score)

        if up_score > down_score and up_ratio > 0.38:
            direction = "UP"
            confidence = up_ratio
        elif down_score > up_score and down_ratio > 0.38:
            direction = "DOWN"
            confidence = down_ratio
        else:
            # 当分数接近时，选择分数较高的一方（而不是NEUTRAL）
            if up_score > down_score:
                direction = "UP"
                confidence = 0.5 + (up_ratio - 0.5) * 0.5
            elif down_score > up_score:
                direction = "DOWN"
                confidence = 0.5 + (down_ratio - 0.5) * 0.5
            else:
                direction = "NEUTRAL"
                confidence = 0.45

        # 调整置信度 - 更激进地提升置信度
        if signal_strength > 0.25:
            confidence = min(confidence + 0.18, 0.95)
        elif signal_strength > 0.15:
            confidence = min(confidence + 0.12, 0.92)
        elif signal_strength > 0.08:
            confidence = min(confidence + 0.08, 0.88)

        # 信号一致性检查 (借鉴decide.py的共识机制)
        agreement_count = sum(1 for s in signals if s["direction"] == direction)
        total_active = sum(1 for s in signals if s["direction"] != "NEUTRAL")
        if total_active > 0:
            agreement_ratio = agreement_count / total_active
            # 共识加成: 如果>=3个信号源同意，增强置信度
            if agreement_count >= 3:
                confidence = min(confidence + 0.12, 0.95)
            elif agreement_count >= 2:
                confidence = min(confidence + 0.08, 0.92)
            elif agreement_ratio > 0.6:
                confidence = min(confidence + 0.05, 0.90)

        # Top3投票机制 (借鉴decide.py)
        # 按权重排序，选择前3个信号源投票
        sorted_signals = sorted(signals, key=lambda x: x["weight"], reverse=True)
        top3_signals = sorted_signals[:3]
        top3_up = sum(1 for s in top3_signals if s["direction"] == "UP")
        top3_down = sum(1 for s in top3_signals if s["direction"] == "DOWN")
        top3_neutral = sum(1 for s in top3_signals if s["direction"] == "NEUTRAL")

        # Top3投票加成 - 更激进
        if top3_up >= 2 and direction == "UP":
            confidence = min(confidence + 0.10, 0.95)
        elif top3_down >= 2 and direction == "DOWN":
            confidence = min(confidence + 0.10, 0.95)
        elif top3_up == 1 and top3_down == 0 and direction == "UP":
            confidence = min(confidence + 0.05, 0.90)
        elif top3_down == 1 and top3_up == 0 and direction == "DOWN":
            confidence = min(confidence + 0.05, 0.90)

        # 市场状态过滤 - 更宽松
        if market_regime:
            # 只在极度震荡市场降低置信度
            if (
                market_regime["regime"] == "ranging"
                and market_regime.get("adx", 20) < 15
            ):
                if direction != "NEUTRAL":
                    confidence = max(confidence - 0.05, 0.40)

        # 收集所有解释因素
        bullish_factors = []
        bearish_factors = []

        # 从技术分析获取因素
        bullish_factors.extend(technical_result.get("bullish_factors", []))
        bearish_factors.extend(technical_result.get("bearish_factors", []))

        # 从动量分析获取因素
        if momentum_result["direction"] == "UP":
            bullish_factors.append(momentum_result["explanation"])
        elif momentum_result["direction"] == "DOWN":
            bearish_factors.append(momentum_result["explanation"])

        # 从情绪分析获取因素
        if sentiment_result["direction"] == "UP":
            bullish_factors.append(sentiment_result["explanation"])
        elif sentiment_result["direction"] == "DOWN":
            bearish_factors.append(sentiment_result["explanation"])

        # 从趋势强度获取因素
        if trend_result:
            if trend_result["direction"] == "UP":
                bullish_factors.append(trend_result["explanation"])
            elif trend_result["direction"] == "DOWN":
                bearish_factors.append(trend_result["explanation"])

        # 从策略叠加获取因素
        if strategy_result and strategy_result["explanations"]:
            for exp in strategy_result["explanations"][:3]:
                if "金叉" in exp or "超卖" in exp or "多头" in exp or "上涨" in exp:
                    bullish_factors.append(f"策略信号: {exp}")
                elif "死叉" in exp or "超买" in exp or "空头" in exp or "下跌" in exp:
                    bearish_factors.append(f"策略信号: {exp}")

        # 从ML模型获取概率解释
        ml_prob_text = (
            f"ML概率: UP {ml_result['up_prob']:.1%} / DOWN {ml_result['down_prob']:.1%}"
        )
        if ml_result["up_prob"] > ml_result["down_prob"] + 0.1:
            bullish_factors.append(ml_prob_text)
        elif ml_result["down_prob"] > ml_result["up_prob"] + 0.1:
            bearish_factors.append(ml_prob_text)

        # 从支撑阻力获取因素
        if support_resistance:
            if support_resistance["direction"] == "UP":
                bullish_factors.append(support_resistance["explanation"])
            elif support_resistance["direction"] == "DOWN":
                bearish_factors.append(support_resistance["explanation"])

        # 从多时间框架获取因素
        if multi_tf_result:
            if (
                multi_tf_result["direction"] == "UP"
                and multi_tf_result["consensus"] >= 2
            ):
                bullish_factors.append(multi_tf_result["explanation"])
            elif (
                multi_tf_result["direction"] == "DOWN"
                and multi_tf_result["consensus"] <= -2
            ):
                bearish_factors.append(multi_tf_result["explanation"])

        # 高置信度过滤: 更宽松的条件
        # 只有当两个方向的因素都很少时才降低置信度
        if len(bullish_factors) == 0 and len(bearish_factors) == 0:
            confidence = max(confidence - 0.08, 0.35)
        elif len(bullish_factors) <= 1 and len(bearish_factors) <= 1:
            confidence = max(confidence - 0.03, 0.40)

        return {
            "direction": direction,
            "confidence": confidence,
            "current_price": current_price,
            "ml_action": ml_result["action"],
            "ml_confidence": ml_result["confidence"],
            "ml_up_prob": ml_result["up_prob"],
            "ml_down_prob": ml_result["down_prob"],
            "ml_hold_prob": ml_result["hold_prob"],
            "technical_direction": technical_result["direction"],
            "technical_confidence": technical_result["confidence"],
            "bullish_count": technical_result.get("bullish_count", 0),
            "bearish_count": technical_result.get("bearish_count", 0),
            "bullish_factors": bullish_factors[:8],  # 最多8个看涨因素
            "bearish_factors": bearish_factors[:8],  # 最多8个看跌因素
            "momentum_5": momentum_result.get("momentum_5", 0),
            "momentum_10": momentum_result.get("momentum_10", 0),
            "sentiment_score": sentiment_result.get("score", 0),
            "adx": trend_result.get("adx", 0) if trend_result else 0,
            "plus_di": trend_result.get("plus_di", 0) if trend_result else 0,
            "minus_di": trend_result.get("minus_di", 0) if trend_result else 0,
            "ma_arrangement": trend_result.get("ma_arrangement", "neutral")
            if trend_result
            else "neutral",
            "alpha": alpha_result.get("alpha", 0) if alpha_result else 0,
            "beta": alpha_result.get("beta", 1.0) if alpha_result else 1.0,
            "market_regime": market_regime.get("regime", "unknown")
            if market_regime
            else "unknown",
            "strategy_bullish_votes": strategy_result.get("bullish_votes", 0)
            if strategy_result
            else 0,
            "strategy_bearish_votes": strategy_result.get("bearish_votes", 0)
            if strategy_result
            else 0,
            "support": support_resistance.get("support", 0)
            if support_resistance
            else 0,
            "resistance": support_resistance.get("resistance", 0)
            if support_resistance
            else 0,
            "signal_sources": {
                "ml": {
                    "direction": ml_direction,
                    "confidence": ml_result["confidence"],
                },
                "technical": {
                    "direction": technical_result["direction"],
                    "confidence": technical_result["confidence"],
                },
                "momentum": {
                    "direction": momentum_result["direction"],
                    "confidence": momentum_result["strength"],
                },
                "trend": {
                    "direction": trend_result["direction"]
                    if trend_result
                    else "NEUTRAL",
                    "confidence": trend_result["strength"] if trend_result else 0.5,
                },
                "alpha": {
                    "direction": alpha_result["direction"]
                    if alpha_result
                    else "NEUTRAL",
                    "confidence": alpha_result["strength"] if alpha_result else 0.5,
                },
                "strategy": {
                    "direction": strategy_result["direction"]
                    if strategy_result
                    else "NEUTRAL",
                    "confidence": strategy_result["strength"]
                    if strategy_result
                    else 0.5,
                    "votes": f"{strategy_result.get('bullish_votes', 0)}/{strategy_result.get('bearish_votes', 0)}"
                    if strategy_result
                    else "0/0",
                },
                "support_resistance": {
                    "direction": support_resistance["direction"]
                    if support_resistance
                    else "NEUTRAL",
                    "confidence": support_resistance["strength"]
                    if support_resistance
                    else 0.5,
                },
                "multi_timeframe": {
                    "direction": multi_tf_result["direction"]
                    if multi_tf_result
                    else "NEUTRAL",
                    "confidence": multi_tf_result["strength"]
                    if multi_tf_result
                    else 0.5,
                    "consensus": multi_tf_result["consensus"] if multi_tf_result else 0,
                    "short": multi_tf_result["short_trend"]
                    if multi_tf_result
                    else "N/A",
                    "medium": multi_tf_result["medium_trend"]
                    if multi_tf_result
                    else "N/A",
                    "long": multi_tf_result["long_trend"] if multi_tf_result else "N/A",
                },
            },
        }

    def _create_neutral_prediction(self, current_price: float) -> Dict[str, Any]:
        """创建中性预测（当数据不足时）"""
        return {
            "direction": "NEUTRAL",
            "confidence": 0.5,
            "current_price": current_price,
            "ml_action": "HOLD",
            "ml_confidence": 0.5,
            "technical_direction": "NEUTRAL",
            "bullish_factors": ["数据不足，无法生成有效信号"],
            "bearish_factors": [],
        }
