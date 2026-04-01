"""模型训练和预测流水线 - 复用现有 StockTradingModel"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from ..models import StockTradingModel

logger = logging.getLogger(__name__)


class ModelPipeline:
    """模型训练和预测流水线

    封装模型训练和预测逻辑，供新脚本复用。
    不影响现有的 decide.py 和 backtest.py。
    """

    def __init__(self, config=None):
        """初始化流水线

        Args:
            config: 配置实例
        """
        self.config = config

    def train(
        self,
        train_df: pd.DataFrame,
        forward_days: int = 1,
        threshold: float = 0.01,
        use_composite_labels: bool = True,
        **kwargs,
    ) -> StockTradingModel:
        """训练模型

        复用现有的 StockTradingModel。

        Args:
            train_df: 训练数据
            forward_days: 预测天数（1表示预测下个交易日）
            threshold: 涨跌阈值
            use_composite_labels: 是否使用复合标签
            **kwargs: 其他参数

        Returns:
            训练好的模型
        """
        logger.info(
            f"Training model: forward_days={forward_days}, threshold={threshold}"
        )

        model = StockTradingModel()
        model.train(
            train_df,
            forward_days=forward_days,
            threshold=threshold,
            use_composite_labels=use_composite_labels,
            **kwargs,
        )

        logger.info("Model training completed")
        return model

    def predict(
        self,
        model: StockTradingModel,
        df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """预测交易动作

        复用 model.predict 和 model.predict_proba。

        Args:
            model: 训练好的模型
            df: 特征数据

        Returns:
            包含 action, confidence, probabilities 的字典
        """
        action, confidence = model.predict(df)
        probabilities = model.predict_proba(df)

        return {
            "action": action,
            "confidence": confidence,
            "probabilities": probabilities,
        }

    def predict_direction(
        self,
        model: StockTradingModel,
        latest_df: pd.DataFrame,
        current_price: float,
    ) -> Dict[str, Any]:
        """预测下个交易日涨跌方向

        Args:
            model: 训练好的模型
            latest_df: 最新一天的特征数据
            current_price: 当前价格

        Returns:
            包含方向预测和概率的字典
        """
        result = self.predict(model, latest_df)

        # 获取概率
        probs = result["probabilities"]

        # 处理字典格式的概率
        if isinstance(probs, dict):
            sell_prob = probs.get("sell_probability", 0.33)
            hold_prob = probs.get("hold_probability", 0.34)
            buy_prob = probs.get("buy_probability", 0.33)
        elif isinstance(probs, list) and len(probs) > 0:
            if isinstance(probs[0], (list, tuple)) and len(probs[0]) >= 3:
                sell_prob, hold_prob, buy_prob = probs[0]
            elif len(probs) >= 3:
                sell_prob, hold_prob, buy_prob = probs[0], probs[1], probs[2]
            else:
                sell_prob, hold_prob, buy_prob = 0.33, 0.34, 0.33
        else:
            sell_prob, hold_prob, buy_prob = 0.33, 0.34, 0.33

        # 确定方向
        if result["action"] == "BUY" or buy_prob > sell_prob:
            direction = "UP"
        elif result["action"] == "SELL" or sell_prob > buy_prob:
            direction = "DOWN"
        else:
            direction = "NEUTRAL"

        return {
            "direction": direction,
            "up_prob": float(buy_prob),
            "hold_prob": float(hold_prob),
            "down_prob": float(sell_prob),
            "confidence": float(result["confidence"]),
            "action": result["action"],
            "current_price": current_price,
        }

    def evaluate_accuracy(
        self,
        model: StockTradingModel,
        eval_df: pd.DataFrame,
        threshold: float = 0.0,
    ) -> float:
        """评估模型准确率

        Args:
            model: 训练好的模型
            eval_df: 评估数据
            threshold: 涨跌阈值

        Returns:
            准确率 (0-1)
        """
        if eval_df.empty or len(eval_df) < 2:
            return 0.0

        correct = 0
        total = 0

        # 逐行评估，跳过最后一行（因为没有下一天的数据）
        for i in range(len(eval_df) - 1):
            # 获取当前特征（用于预测）
            row = eval_df.iloc[[i]]

            # 计算实际涨跌（从当前行到下一行）
            current_close = eval_df["close"].iloc[i]
            next_close = eval_df["close"].iloc[i + 1]
            actual_return = (next_close - current_close) / current_close

            # 确定实际方向
            if actual_return > threshold:
                actual_direction = "UP"
            elif actual_return < -threshold:
                actual_direction = "DOWN"
            else:
                actual_direction = "NEUTRAL"

            # 预测方向
            try:
                prediction = self.predict_direction(model, row, current_close)
                predicted_direction = prediction["direction"]

                # 判断预测是否正确
                if predicted_direction == actual_direction:
                    correct += 1
                total += 1
            except Exception as e:
                logger.debug(f"Prediction error at index {i}: {e}")
                continue

        accuracy = correct / total if total > 0 else 0.0
        logger.info(f"Evaluation accuracy: {accuracy:.2%} ({correct}/{total})")
        return accuracy

    def get_feature_contributions(
        self,
        model: StockTradingModel,
        latest_df: pd.DataFrame,
        top_n: int = 5,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """获取特征贡献

        分析哪些特征对预测影响最大。

        Args:
            model: 训练好的模型
            latest_df: 最新特征数据
            top_n: 返回前N个特征

        Returns:
            包含 positive 和 negative 特征贡献的字典
        """
        importance = model.get_feature_importance()

        if importance.empty:
            return {"positive": [], "negative": []}

        # 获取top特征
        top_features = importance.head(top_n)

        # 分类正负贡献
        positive = []
        negative = []

        for _, row in top_features.iterrows():
            feature_name = row["feature"]
            importance_val = row["importance"]

            # 获取特征值
            if feature_name in latest_df.columns:
                value = latest_df[feature_name].iloc[0]
            else:
                value = None

            feature_info = {
                "name": feature_name,
                "importance": importance_val,
                "value": value,
            }

            # 根据特征类型判断正负贡献
            if any(
                keyword in feature_name
                for keyword in ["rsi", "momentum", "returns", "ma_ratio"]
            ):
                if value is not None and value > 0:
                    positive.append(feature_info)
                else:
                    negative.append(feature_info)
            else:
                positive.append(feature_info)

        return {
            "positive": positive[:top_n],
            "negative": negative[:top_n],
        }
