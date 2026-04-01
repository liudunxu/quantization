"""预测结果格式化器 - 支持多种输出格式"""

import json
from typing import Dict, Any, List


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
        """纯文本格式"""
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
        up_bar = "█" * int(p.get("up_prob", 0) * 20)
        down_bar = "█" * int(p.get("down_prob", 0) * 20)
        hold_bar = "█" * int(p.get("hold_prob", 0) * 20)

        lines = [
            "=" * 60,
            f"  STOCK PREDICTION - {p.get('stock_code', 'N/A')}",
            "=" * 60,
            "",
            f"  Current Price  : {p.get('current_price', 0):.2f}",
            f"  Prediction Date: {p.get('prediction_date', 'N/A')}",
            f"  Target Date    : {p.get('target_date', 'N/A')}",
            "",
            "  === Prediction ===",
            f"  Direction      : {p.get('direction', 'N/A')} {direction_emoji.get(p.get('direction'), '')}",
            f"  Signal         : {direction_label.get(p.get('direction'), 'N/A')}",
            f"  Confidence     : {p.get('confidence', 0):.2f}",
            "",
            "  === Probability Distribution ===",
            f"  UP   : {p.get('up_prob', 0) * 100:5.1f}% {up_bar}",
            f"  DOWN : {p.get('down_prob', 0) * 100:5.1f}% {down_bar}",
            f"  HOLD : {p.get('hold_prob', 0) * 100:5.1f}% {hold_bar}",
            "",
            "  === Model Performance ===",
            f"  Accuracy       : {p.get('model_accuracy', 0) * 100:.1f}%",
        ]

        # 添加特征贡献
        top_features = p.get("top_features", [])
        if top_features:
            lines.append("")
            lines.append("  === Key Features ===")
            for i, feat in enumerate(top_features[:5], 1):
                name = feat.get("feature", "N/A")
                importance = feat.get("importance", 0)
                lines.append(f"  {i}. {name:<25} {importance:.2f}")

        lines.append("=" * 60)
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
            "up_prob",
            "down_prob",
            "hold_prob",
            "confidence",
            "model_accuracy",
        ]

        values = []
        for h in headers:
            val = p.get(h, "")
            if isinstance(val, float):
                values.append(f"{val:.4f}")
            else:
                values.append(str(val))

        return ",".join(headers) + "\n" + ",".join(values)
