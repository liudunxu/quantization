"""Southbound capital flow features for Hong Kong stocks (南向资金)."""

from typing import Optional
import pandas as pd
import numpy as np
from ..utils.cache import FeatureCache
from ..utils.config import get_config
from .base import BaseFeatureExtractor
from ..data_providers import AKShareProvider


class SouthboundFlowFeatures(BaseFeatureExtractor):
    """Extract southbound capital flow features for HK stocks.

    南向资金是指通过港股通从内地流向香港市场的资金。
    这些资金的流向可以反映内地投资者对港股的态度。
    """

    def __init__(self, cache: Optional[FeatureCache] = None):
        super().__init__(cache)
        self._flow_data = None

    @property
    def feature_type(self) -> str:
        return "southbound_flow"

    def _get_southbound_data(self, days: int = 180) -> pd.DataFrame:
        """获取南向资金数据并缓存"""
        if self._flow_data is not None:
            return self._flow_data

        try:
            provider = AKShareProvider()
            self._flow_data = provider.fetch_southbound_flow(days=days)
            return self._flow_data
        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to fetch southbound flow data: {e}")
            return pd.DataFrame()

    def extract(self, stock_code: str, **kwargs) -> pd.DataFrame:
        """Extract southbound flow features for a HK stock.

        Args:
            stock_code: Stock code (e.g., 0700.HK)

        Returns:
            DataFrame with southbound flow features
        """
        days = kwargs.get("days", 30)

        # 只对港股添加南向资金特征
        if not stock_code.endswith(".HK"):
            return pd.DataFrame()

        # 获取南向资金数据
        flow_df = self._get_southbound_data(days=days + 60)

        if flow_df.empty:
            return pd.DataFrame()

        df = pd.DataFrame()
        df["date"] = flow_df["date"]
        df["stock_code"] = stock_code

        # ========== 基础南向资金特征 ==========

        # 净买入额 (亿元)
        df["southbound_net_buy"] = flow_df["southbound_net_buy"]
        df["southbound_sh_net_buy"] = flow_df["southbound_sh_net_buy"]
        df["southbound_sz_net_buy"] = flow_df["southbound_sz_net_buy"]

        # 买入额和卖出额
        df["southbound_buy"] = flow_df["southbound_buy"]
        df["southbound_sell"] = flow_df["southbound_sell"]

        # ========== 移动平均特征 ==========

        # 5/10/20日净买入均线
        for period in [5, 10, 20]:
            df[f"southbound_ma_{period}"] = (
                df["southbound_net_buy"].rolling(window=period).mean()
            )

        # ========== 动量特征 ==========

        # 净买入变化率
        df["southbound_change_1d"] = df["southbound_net_buy"].pct_change()
        df["southbound_change_5d"] = df["southbound_net_buy"].pct_change(5)

        # 净买入动量 (当前值 - N日前值)
        for period in [5, 10]:
            df[f"southbound_momentum_{period}"] = df["southbound_net_buy"] - df[
                "southbound_net_buy"
            ].shift(period)

        # ========== 累计特征 ==========

        # 近N日累计净买入
        for period in [5, 10, 20]:
            df[f"southbound_cumsum_{period}"] = (
                df["southbound_net_buy"].rolling(window=period).sum()
            )

        # ========== 信号特征 ==========

        # 净买入为正/负
        df["southbound_positive"] = (df["southbound_net_buy"] > 0).astype(int)

        # 连续净买入天数
        df["southbound_streak"] = (
            df["southbound_positive"]
            .groupby(
                (
                    df["southbound_positive"] != df["southbound_positive"].shift()
                ).cumsum()
            )
            .cumcount()
            + 1
        ) * df["southbound_positive"]

        # 连续净卖出天数 (负向)
        df["southbound_streak_negative"] = (
            (1 - df["southbound_positive"])
            .groupby(
                (
                    (1 - df["southbound_positive"])
                    != (1 - df["southbound_positive"]).shift()
                ).cumsum()
            )
            .cumcount()
            + 1
        ) * (1 - df["southbound_positive"])

        # ========== 波动率特征 ==========

        # 净买入波动率
        df["southbound_volatility"] = df["southbound_net_buy"].rolling(window=10).std()

        # ========== 相对强弱特征 ==========

        # 沪港通 vs 深港通 比例
        total = df["southbound_sh_net_buy"].abs() + df["southbound_sz_net_buy"].abs()
        df["southbound_sh_ratio"] = np.where(
            total > 0,
            df["southbound_sh_net_buy"].abs() / total,
            0.5,
        )

        # 净买入 / 总成交额比例
        total_volume = df["southbound_buy"] + df["southbound_sell"]
        df["southbound_net_ratio"] = np.where(
            total_volume > 0,
            df["southbound_net_buy"] / total_volume,
            0,
        )

        # ========== 趋势特征 ==========

        # 净买入均线方向
        df["southbound_trend"] = np.where(
            df["southbound_ma_5"] > df["southbound_ma_10"],
            1,
            np.where(df["southbound_ma_5"] < df["southbound_ma_10"], -1, 0),
        )

        # 净买入加速 (当前动量 > 前一日动量)
        df["southbound_acceleration"] = np.where(
            df["southbound_momentum_5"] > df["southbound_momentum_5"].shift(1),
            1,
            np.where(
                df["southbound_momentum_5"] < df["southbound_momentum_5"].shift(1),
                -1,
                0,
            ),
        )

        # 填充空值
        df = df.ffill().bfill().fillna(0)

        return df
