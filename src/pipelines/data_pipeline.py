"""数据获取和特征工程流水线 - 复用现有 combinator"""

import logging
from typing import Optional, Tuple
import pandas as pd

from ..features import get_feature_combinator
from ..utils import get_cache, get_config
from ..data_providers import fetch_realtime_price

logger = logging.getLogger(__name__)


class DataPipeline:
    """数据获取和特征工程流水线

    封装特征获取逻辑，供新脚本复用。
    不影响现有的 decide.py 和 backtest.py。
    """

    def __init__(self, cache=None, config=None):
        """初始化流水线

        Args:
            cache: 特征缓存实例，如果为None则自动创建
            config: 配置实例，如果为None则自动加载
        """
        self.config = config or get_config()
        self.cache = cache or get_cache(self.config.get("data.cache_dir", "cache"))
        self.combinator = get_feature_combinator(self.cache)

    def fetch_features(
        self,
        stock_code: str,
        days: int = 365,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """获取特征数据

        复用现有的 FeatureCombinator 获取所有特征。

        Args:
            stock_code: 股票代码
            days: 获取天数
            force_refresh: 是否强制刷新缓存

        Returns:
            包含所有特征的 DataFrame
        """
        logger.info(f"Fetching features for {stock_code}, days={days}")
        df = self.combinator.get_combined_features(stock_code, days, force_refresh)

        if df.empty:
            logger.warning(f"No data fetched for {stock_code}")
        else:
            logger.info(f"Fetched {len(df)} samples for {stock_code}")

        return df

    def split_train_eval(
        self,
        df: pd.DataFrame,
        backtest_days: int = 30,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """划分训练集和评估集

        Args:
            df: 完整数据
            backtest_days: 评估集天数

        Returns:
            (train_df, eval_df) 元组
        """
        if len(df) <= backtest_days:
            logger.warning("Data too short for splitting, using all for training")
            return df, pd.DataFrame()

        train_df = df.iloc[:-backtest_days]
        eval_df = df.iloc[-backtest_days:]

        logger.info(f"Split: train={len(train_df)}, eval={len(eval_df)}")
        return train_df, eval_df

    def get_latest(self, df: pd.DataFrame) -> pd.DataFrame:
        """获取最新一天的特征

        用于预测下个交易日。

        Args:
            df: 特征数据

        Returns:
            仅包含最新一天的 DataFrame
        """
        if df.empty:
            return df
        return df.iloc[[-1]]

    def get_realtime_price(self, stock_code: str) -> Optional[float]:
        """获取实时价格

        复用现有的 fetch_realtime_price 函数。

        Args:
            stock_code: 股票代码

        Returns:
            实时价格，如果获取失败返回None
        """
        try:
            # fetch_realtime_price 直接返回 price 或 None
            price = fetch_realtime_price(stock_code)
            return price
        except Exception as e:
            logger.warning(f"Failed to get realtime price: {e}")
            return None
