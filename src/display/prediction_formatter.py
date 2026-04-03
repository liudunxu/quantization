"""预测结果格式化器 - 支持多种输出格式"""

import json
from typing import Any, Dict


class PredictionFormatter:
    """预测结果格式化器

    支持输出格式：text (默认), json, csv
    """

    def format(self, prediction: Dict[str, Any], fmt: str = "text") -> str:
        """根据格式输出

        Args:
            prediction: 预测结果字典
            fmt: 输出格式 ('text', 'json', 'csv')

        Returns:
            格式化后的字符串
        """
        formatters = {
            "text": self._format_text,
            "json": self._format_json,
            "csv": self._format_csv,
        }
        formatter = formatters.get(fmt, self._format_text)
        return formatter(prediction)

    def _format_text(self, p: Dict[str, Any]) -> str:
        """纯文本格式 - 带详细解释"""
        direction_emoji = {
            "UP": "↑",
            "DOWN": "↓",
            "NEUTRAL": "→",
        }
        direction_label = {
            "UP": "看涨 (BULLISH)",
            "DOWN": "看跌 (BEARISH)",
            "NEUTRAL": "中性 (NEUTRAL)",
        }

        # 概率柱状图
        up_prob = p.get("ml_up_prob", p.get("up_prob", 0))
        down_prob = p.get("ml_down_prob", p.get("down_prob", 0))
        hold_prob = p.get("ml_hold_prob", p.get("hold_prob", 0))

        up_bar = "█" * int(up_prob * 20)
        down_bar = "█" * int(down_prob * 20)
        hold_bar = "█" * int(hold_prob * 20)

        # 置信度等级
        confidence = p.get("confidence", 0)
        if confidence >= 0.7:
            confidence_level = "高 (HIGH)"
            confidence_bar = "██████████"
        elif confidence >= 0.5:
            confidence_level = "中 (MEDIUM)"
            confidence_bar = "██████"
        else:
            confidence_level = "低 (LOW)"
            confidence_bar = "███"

        lines = [
            "=" * 60,
            f"  STOCK PREDICTION - {p.get('stock_code', 'N/A')}",
            "=" * 60,
            "",
            f"  Current Price  : {p.get('current_price', 0):.2f}",
            f"  Prediction Date: {p.get('prediction_date', 'N/A')}",
            f"  Target Date    : {p.get('target_date', 'N/A')}",
            "",
            "  ══════════════════════════════════════════════════════",
            "  预测结果 (PREDICTION RESULT)",
            "  ══════════════════════════════════════════════════════",
            "",
            f"  方向 (Direction)  : {p.get('direction', 'N/A')} {direction_emoji.get(p.get('direction'), '')}",
            f"  信号 (Signal)     : {direction_label.get(p.get('direction'), 'N/A')}",
            f"  置信度 (Confidence): {confidence:.1%} [{confidence_level}] {confidence_bar}",
            "",
            "  ══════════════════════════════════════════════════════",
            "  ML模型概率分布 (ML Probability Distribution)",
            "  ══════════════════════════════════════════════════════",
            "",
            f"  看涨 (UP)   : {up_prob * 100:5.1f}% {up_bar}",
            f"  持有 (HOLD) : {hold_prob * 100:5.1f}% {hold_bar}",
            f"  看跌 (DOWN) : {down_prob * 100:5.1f}% {down_bar}",
        ]

        # 添加信号源分析
        signal_sources = p.get("signal_sources", {})
        if signal_sources:
            lines.extend(
                [
                    "",
                    "  ══════════════════════════════════════════════════════",
                    "  信号源分析 (SIGNAL SOURCES)",
                    "  ══════════════════════════════════════════════════════",
                    "",
                ]
            )
            for source_name, source_info in signal_sources.items():
                source_labels = {
                    "ml": "ML模型",
                    "technical": "技术分析",
                    "momentum": "动量分析",
                    "trend": "趋势强度",
                    "alpha": "超额收益",
                    "strategy": "策略叠加",
                    "support_resistance": "支撑阻力",
                }
                label = source_labels.get(source_name, source_name)
                direction = source_info.get("direction", "NEUTRAL")
                conf = source_info.get("confidence", 0)
                direction_symbol = {
                    "UP": "↑ 看涨",
                    "DOWN": "↓ 看跌",
                    "NEUTRAL": "→ 中性",
                }.get(direction, "?")
                votes = source_info.get("votes", "")
                vote_text = f" [{votes}]" if votes else ""
                lines.append(
                    f"  {label:<12} : {direction_symbol} (置信度: {conf:.1%}){vote_text}"
                )

        # 添加看涨因素
        bullish_factors = p.get("bullish_factors", [])
        if bullish_factors:
            lines.extend(
                [
                    "",
                    "  ══════════════════════════════════════════════════════",
                    "  看涨因素 (BULLISH FACTORS)",
                    "  ══════════════════════════════════════════════════════",
                    "",
                ]
            )
            for i, factor in enumerate(bullish_factors[:5], 1):
                lines.append(f"  {i}. {factor}")

        # 添加看跌因素
        bearish_factors = p.get("bearish_factors", [])
        if bearish_factors:
            lines.extend(
                [
                    "",
                    "  ══════════════════════════════════════════════════════",
                    "  看跌因素 (BEARISH FACTORS)",
                    "  ══════════════════════════════════════════════════════",
                    "",
                ]
            )
            for i, factor in enumerate(bearish_factors[:5], 1):
                lines.append(f"  {i}. {factor}")

        # 添加关键指标
        lines.extend(
            [
                "",
                "  ══════════════════════════════════════════════════════",
                "  关键指标 (KEY INDICATORS)",
                "  ══════════════════════════════════════════════════════",
                "",
                f"  5日动量 (Momentum 5d)  : {p.get('momentum_5', 0) * 100:+.2f}%",
                f"  10日动量 (Momentum 10d): {p.get('momentum_10', 0) * 100:+.2f}%",
                f"  市场情绪 (Sentiment)   : {p.get('sentiment_score', 0):.2f}",
                f"  ADX趋势强度            : {p.get('adx', 0):.0f}",
                f"  +DI/-DI                : {p.get('plus_di', 0):.1f} / {p.get('minus_di', 0):.1f}",
                f"  均线排列 (MA Arrangement): {p.get('ma_arrangement', 'neutral')}",
                f"  Alpha超额收益          : {p.get('alpha', 0):.2%}",
                f"  市场状态 (Market Regime): {p.get('market_regime', 'unknown')}",
                f"  支撑位 (Support)       : {p.get('support', 0):.2f}",
                f"  阻力位 (Resistance)    : {p.get('resistance', 0):.2f}",
                f"  看涨信号数 (Bullish)   : {p.get('bullish_count', 0)}",
                f"  看跌信号数 (Bearish)   : {p.get('bearish_count', 0)}",
                f"  策略看涨票数           : {p.get('strategy_bullish_votes', 0)}",
                f"  策略看跌票数           : {p.get('strategy_bearish_votes', 0)}",
            ]
        )

        # 添加模型性能 (包含准召)
        lines.extend(
            [
                "",
                "  ══════════════════════════════════════════════════════",
                "  模型性能 (MODEL PERFORMANCE)",
                "  ══════════════════════════════════════════════════════",
                "",
                f"  评估准确率 (Accuracy)  : {p.get('model_accuracy', 0) * 100:.1f}%",
                f"  ML置信度 (ML Conf)     : {p.get('ml_confidence', 0) * 100:.1f}%",
            ]
        )

        # 添加准召数据
        if p.get("precision_up", None) is not None:
            lines.extend(
                [
                    "",
                    "  ── 看涨(UP)方向 ──",
                    f"  Precision (精准率)  : {p.get('precision_up', 0) * 100:.1f}%",
                    f"  Recall (召回率)     : {p.get('recall_up', 0) * 100:.1f}%",
                    f"  F1 Score            : {p.get('f1_up', 0) * 100:.1f}%",
                    "",
                    "  ── 看跌(DOWN)方向 ──",
                    f"  Precision (精准率)  : {p.get('precision_down', 0) * 100:.1f}%",
                    f"  Recall (召回率)     : {p.get('recall_down', 0) * 100:.1f}%",
                    f"  F1 Score            : {p.get('f1_down', 0) * 100:.1f}%",
                ]
            )

        # 添加交易建议
        direction = p.get("direction", "NEUTRAL")
        lines.extend(
            [
                "",
                "  ══════════════════════════════════════════════════════",
                "  交易建议 (TRADING SUGGESTION)",
                "  ══════════════════════════════════════════════════════",
                "",
            ]
        )
        if direction == "UP":
            lines.extend(
                [
                    "  ✓ 多数指标显示看涨信号",
                    "  ✓ 可考虑逢低买入",
                    "  ✓ 注意设置止损位控制风险",
                ]
            )
        elif direction == "DOWN":
            lines.extend(
                [
                    "  ✗ 多数指标显示看跌信号",
                    "  ✗ 建议观望或减仓",
                    "  ✗ 如有持仓注意止损",
                ]
            )
        else:
            lines.extend(
                [
                    "  - 多空信号胶着，方向不明",
                    "  - 建议观望等待明确信号",
                    "  - 可关注成交量变化",
                ]
            )

        lines.extend(
            [
                "",
                "  ⚠️  免责声明：本预测仅供参考，不构成投资建议",
                "     投资有风险，入市需谨慎",
                "",
                "=" * 60,
            ]
        )

        return "\n".join(lines)

    def _format_json(self, p: Dict[str, Any]) -> str:
        """JSON格式"""
        # 处理不可序列化的对象
        clean_data = {}
        for key, value in p.items():
            if isinstance(value, (list, dict)):
                clean_data[key] = value
            elif hasattr(value, "item"):  # numpy scalar
                clean_data[key] = value.item()
            elif hasattr(value, "tolist"):  # numpy array
                clean_data[key] = value.tolist()
            else:
                clean_data[key] = value
        return json.dumps(clean_data, indent=2, default=str)

    def _format_csv(self, p: Dict[str, Any]) -> str:
        """CSV格式"""
        headers = [
            "stock_code",
            "prediction_date",
            "target_date",
            "current_price",
            "direction",
            "confidence",
            "ml_up_prob",
            "ml_down_prob",
            "ml_hold_prob",
            "ml_confidence",
            "bullish_count",
            "bearish_count",
            "momentum_5",
            "sentiment_score",
            "adx",
            "alpha",
            "model_accuracy",
            "precision_up",
            "recall_up",
            "f1_up",
            "precision_down",
            "recall_down",
            "f1_down",
        ]

        values = []
        for h in headers:
            val = p.get(h, "")
            if isinstance(val, float):
                values.append(f"{val:.4f}")
            else:
                values.append(str(val))

        return ",".join(headers) + "\n" + ",".join(values)
