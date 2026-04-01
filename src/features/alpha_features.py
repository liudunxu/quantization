"""Alpha features inspired by Qlib's Alpha158.

Reference: Microsoft Qlib - Alpha158 feature set
https://github.com/microsoft/qlib
"""

from typing import Optional, List
import pandas as pd
import numpy as np
import logging
from ..utils.cache import FeatureCache
from .base import BaseFeatureExtractor

logger = logging.getLogger(__name__)


class AlphaFeatures(BaseFeatureExtractor):
    """Alpha features inspired by Qlib's Alpha158.

    包含以下类型的特征:
    - 价格类: 收益率、对数收益率、价格位置
    - 成交量类: 量比、成交量变化率
    - 波动率类: 已实现波动率、波动率比率
    - 动量类: 多周期动量、加速度
    - 技术类: VWAP偏离、高低价差、价格效率
    """

    def __init__(self, cache: Optional[FeatureCache] = None):
        super().__init__(cache)

    @property
    def feature_type(self) -> str:
        return "alpha"

    def extract(self, stock_code: str, **kwargs) -> pd.DataFrame:
        """Extract alpha features."""
        df = kwargs.get("df", pd.DataFrame())

        if df.empty or len(df) < 60:
            return pd.DataFrame()

        result = pd.DataFrame()
        result["date"] = df["date"]
        result["stock_code"] = stock_code

        close = df["close"]
        high = df["high"]
        low = df["low"]
        open_price = df["open"]
        volume = df["volume"]

        # ========== Qlib Alpha158 风格特征 ==========

        # 1. 收益率特征 (ROC - Rate of Change)
        for n in [1, 2, 3, 5, 10, 20]:
            result[f"roc_{n}"] = close.pct_change(n)

        # 2. 对数收益率
        for n in [1, 5, 10]:
            result[f"log_return_{n}"] = np.log(close / close.shift(n))

        # 3. 价格位置特征 (相对于N日高低点)
        for n in [5, 10, 20]:
            high_n = high.rolling(n).max()
            low_n = low.rolling(n).min()
            result[f"price_position_{n}"] = (close - low_n) / (high_n - low_n + 1e-10)

        # 4. 高低价差比率
        for n in [5, 10, 20]:
            result[f"hl_ratio_{n}"] = (
                high.rolling(n).max() - low.rolling(n).min()
            ) / close

        # 5. 开盘缺口
        result["gap"] = (open_price - close.shift(1)) / close.shift(1)

        # 6. 日内收益率
        result["intraday_return"] = (close - open_price) / open_price

        # 7. 尾盘收益率 (接近K线形态)
        result["upper_shadow"] = (high - np.maximum(open_price, close)) / close
        result["lower_shadow"] = (np.minimum(open_price, close) - low) / close
        result["body"] = np.abs(close - open_price) / close

        # 8. 成交量特征
        for n in [5, 10, 20]:
            result[f"volume_ma_{n}"] = volume.rolling(n).mean()
            result[f"volume_ratio_{n}"] = volume / (volume.rolling(n).mean() + 1e-10)

        # 9. 成交量变化率
        for n in [1, 5, 10]:
            result[f"volume_roc_{n}"] = volume.pct_change(n)

        # 10. 量价背离
        result["price_volume_corr_5"] = (
            close.pct_change().rolling(5).corr(volume.pct_change())
        )
        result["price_volume_corr_10"] = (
            close.pct_change().rolling(10).corr(volume.pct_change())
        )

        # 11. 已实现波动率
        for n in [5, 10, 20]:
            result[f"realized_vol_{n}"] = close.pct_change().rolling(n).std() * np.sqrt(
                252
            )

        # 12. 波动率比率 (短期/长期)
        result["vol_ratio_5_20"] = result.get("realized_vol_5", 0) / (
            result.get("realized_vol_20", 1e-10) + 1e-10
        )

        # 13. Parkison波动率 (基于高低价)
        for n in [5, 10]:
            log_hl = np.log(high / low)
            result[f"parkinson_vol_{n}"] = np.sqrt(
                (log_hl**2).rolling(n).mean() / (4 * np.log(2))
            )

        # 14. VWAP偏离 (近似)
        typical_price = (high + low + close) / 3
        for n in [5, 10, 20]:
            vwap = (typical_price * volume).rolling(n).sum() / (
                volume.rolling(n).sum() + 1e-10
            )
            result[f"vwap_deviation_{n}"] = (close - vwap) / (vwap + 1e-10)

        # 15. 价格效率 (Hurst指数近似)
        result["efficiency_5"] = np.abs(close - close.shift(5)) / (
            np.abs(close.diff()).rolling(5).sum() + 1e-10
        )
        result["efficiency_10"] = np.abs(close - close.shift(10)) / (
            np.abs(close.diff()).rolling(10).sum() + 1e-10
        )

        # 16. 动量加速度
        for n in [5, 10]:
            momentum = close / close.shift(n) - 1
            result[f"momentum_accel_{n}"] = momentum - momentum.shift(1)

        # 17. 均线斜率
        for n in [5, 10, 20]:
            ma = close.rolling(n).mean()
            result[f"ma_slope_{n}"] = ma / ma.shift(1) - 1

        # 18. 距离均线的偏离
        for n in [5, 10, 20]:
            ma = close.rolling(n).mean()
            result[f"ma_deviation_{n}"] = (close - ma) / (ma + 1e-10)

        # 19. 相对强弱 (RS)
        for n in [14]:
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(n).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(n).mean()
            rs = gain / (loss + 1e-10)
            result[f"rs_{n}"] = rs

        # 20. Amihud非流动性指标
        result["amihud_5"] = (
            (np.abs(close.pct_change()) / (volume * close + 1e-10)).rolling(5).mean()
        )
        result["amihud_20"] = (
            (np.abs(close.pct_change()) / (volume * close + 1e-10)).rolling(20).mean()
        )

        # 21. Kyle's Lambda (价格冲击近似)
        result["kyle_lambda_5"] = np.abs(close.pct_change()).rolling(5).mean() / (
            volume.rolling(5).mean() + 1e-10
        )

        # 22. 连续涨跌天数
        result["consecutive_up"] = (close > close.shift(1)).astype(int).groupby(
            (close <= close.shift(1)).cumsum()
        ).cumcount() + 1
        result["consecutive_up"] = result["consecutive_up"] * (
            close > close.shift(1)
        ).astype(int)

        result["consecutive_down"] = (close < close.shift(1)).astype(int).groupby(
            (close >= close.shift(1)).cumsum()
        ).cumcount() + 1
        result["consecutive_down"] = result["consecutive_down"] * (
            close < close.shift(1)
        ).astype(int)

        # 23. 特征标准化 (Z-score)
        for col in [c for c in result.columns if c not in ["date", "stock_code"]]:
            if result[col].dtype in [np.float64, np.float32]:
                mean = result[col].rolling(60, min_periods=20).mean()
                std = result[col].rolling(60, min_periods=20).std()
                result[f"{col}_zscore"] = (result[col] - mean) / (std + 1e-10)

        # 填充空值
        result = result.ffill().bfill().fillna(0)

        return result


def calculate_ic(feature: pd.Series, label: pd.Series) -> float:
    """计算特征与标签的Information Coefficient (IC).

    IC = corr(feature, label)

    Args:
        feature: 特征序列
        label: 标签序列

    Returns:
        IC值 (-1 到 1)
    """
    # 去除空值
    valid_idx = ~(feature.isna() | label.isna())
    if valid_idx.sum() < 20:
        return 0.0

    return feature[valid_idx].corr(label[valid_idx])


def select_features_by_ic(
    df: pd.DataFrame,
    label_col: str = "label",
    min_ic: float = 0.02,
    max_features: int = 50,
) -> List[str]:
    """基于IC值筛选特征.

    Args:
        df: 包含特征和标签的DataFrame
        label_col: 标签列名
        min_ic: 最小IC阈值
        max_features: 最大特征数量

    Returns:
        筛选后的特征名列表
    """
    if label_col not in df.columns:
        logger.warning(f"Label column '{label_col}' not found")
        return [c for c in df.columns if c not in ["date", "stock_code", label_col]]

    feature_cols = [c for c in df.columns if c not in ["date", "stock_code", label_col]]

    ic_scores = {}
    for col in feature_cols:
        if df[col].dtype in [np.float64, np.float32, np.int64]:
            ic = calculate_ic(df[col], df[label_col])
            if not np.isnan(ic):
                ic_scores[col] = abs(ic)

    # 按IC绝对值排序
    sorted_features = sorted(ic_scores.items(), key=lambda x: x[1], reverse=True)

    # 筛选IC大于阈值的特征
    selected = [f for f, ic in sorted_features if ic >= min_ic]

    # 限制最大数量
    if len(selected) > max_features:
        selected = selected[:max_features]

    logger.info(f"Selected {len(selected)} features by IC (min_ic={min_ic})")

    return selected


def normalize_features(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """市场感知的Z-score标准化.

    使用滚动窗口进行标准化，避免未来信息泄露。

    Args:
        df: 特征DataFrame
        window: 滚动窗口大小

    Returns:
        标准化后的DataFrame
    """
    result = df.copy()

    for col in df.columns:
        if col in ["date", "stock_code"]:
            continue

        if df[col].dtype in [np.float64, np.float32]:
            mean = df[col].rolling(window, min_periods=window // 2).mean()
            std = df[col].rolling(window, min_periods=window // 2).std()
            result[col] = (df[col] - mean) / (std + 1e-10)

    return result.fillna(0)
