"""集成预测器 - 结合ML模型和技术信号进行综合预测"""

import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from ..models import StockTradingModel
from .technical_signals import TechnicalSignalGenerator

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """集成预测器

    结合ML模型预测和技术分析信号，提供更可靠的预测结果。
    支持多种信号源的加权投票。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化预测器

        Args:
            config: 配置参数
        """
        self.config = config or {}
        self.technical_generator = TechnicalSignalGenerator()

        # 默认权重配置 - 优化后权重分配
        self.ml_weight = self.config.get("ml_weight", 0.40)  # ML模型权重
        self.technical_weight = self.config.get(
            "technical_weight", 0.30
        )  # 技术信号权重
        self.momentum_weight = self.config.get("momentum_weight", 0.15)  # 动量权重
        self.trend_weight = self.config.get("trend_weight", 0.10)  # 趋势强度权重
        self.alpha_weight = self.config.get("alpha_weight", 0.05)  # 超额收益权重

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

        # 5. 趋势强度分析 (新增)
        trend_result = self._analyze_trend_strength(latest, df)

        # 6. 超额收益分析 (新增)
        alpha_result = self._analyze_alpha(latest, df)

        # 7. 综合判断
        final_result = self._combine_predictions(
            ml_result,
            technical_result,
            momentum_result,
            sentiment_result,
            current_price,
            trend_result,
            alpha_result,
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

    def _combine_predictions(
        self,
        ml_result: Dict[str, Any],
        technical_result: Dict[str, Any],
        momentum_result: Dict[str, Any],
        sentiment_result: Dict[str, Any],
        current_price: float,
        trend_result: Optional[Dict[str, Any]] = None,
        alpha_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """综合多个预测结果"""

        # 收集各信号源的方向和权重
        signals = []

        # ML信号
        ml_direction = (
            "UP"
            if ml_result["action"] == "BUY"
            else ("DOWN" if ml_result["action"] == "SELL" else "NEUTRAL")
        )
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

        # 趋势强度信号 (新增)
        if trend_result:
            signals.append(
                {
                    "source": "趋势强度",
                    "direction": trend_result["direction"],
                    "weight": self.trend_weight,
                    "confidence": trend_result["strength"],
                }
            )

        # 超额收益信号 (新增)
        if alpha_result:
            signals.append(
                {
                    "source": "超额收益",
                    "direction": alpha_result["direction"],
                    "weight": self.alpha_weight,
                    "confidence": alpha_result["strength"],
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

        total_score = up_score + down_score + neutral_score

        if total_score > 0:
            up_ratio = up_score / total_score
            down_ratio = down_score / total_score
        else:
            up_ratio = down_ratio = 0.33

        # 确定最终方向
        if up_score > down_score and up_ratio > 0.45:
            direction = "UP"
            confidence = up_ratio
        elif down_score > up_score and down_ratio > 0.45:
            direction = "DOWN"
            confidence = down_ratio
        else:
            direction = "NEUTRAL"
            confidence = 0.5

        # 调整置信度
        signal_agreement = (
            abs(up_score - down_score) / total_score if total_score > 0 else 0
        )
        if signal_agreement > 0.3:
            confidence = min(confidence + 0.1, 0.95)

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

        # 从ML模型获取概率解释
        ml_prob_text = (
            f"ML概率: UP {ml_result['up_prob']:.1%} / DOWN {ml_result['down_prob']:.1%}"
        )
        if ml_result["up_prob"] > ml_result["down_prob"]:
            bullish_factors.append(ml_prob_text)
        elif ml_result["down_prob"] > ml_result["up_prob"]:
            bearish_factors.append(ml_prob_text)

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
            "bullish_factors": bullish_factors,
            "bearish_factors": bearish_factors,
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
