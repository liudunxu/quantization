"""Company events and special dates features for stocks (公司事件/特殊日期)."""

from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import logging
from ..utils.cache import FeatureCache
from ..utils.config import get_config
from .base import BaseFeatureExtractor

logger = logging.getLogger(__name__)


class CompanyEventsFeatures(BaseFeatureExtractor):
    """Extract company events and special dates features.

    包括:
    - 财报发布时间 (季报、半年报、年报)
    - 分红除权日
    - 月末/月初/季末特征
    - 特殊时期特征
    """

    def __init__(self, cache: Optional[FeatureCache] = None):
        super().__init__(cache)
        self._dividend_data = None

    @property
    def feature_type(self) -> str:
        return "company_events"

    def _get_akshare(self):
        """获取akshare库"""
        try:
            import akshare as ak

            return ak
        except ImportError:
            logger.warning("akshare not installed")
            return None

    def _get_dividend_data(self) -> pd.DataFrame:
        """获取分红数据并缓存在内存中"""
        if self._dividend_data is not None:
            return self._dividend_data

        try:
            akshare_lib = self._get_akshare()
            if akshare_lib is None:
                return pd.DataFrame()

            df = akshare_lib.news_trade_notify_dividend_baidu()
            if df is not None and not df.empty:
                df["除权日"] = pd.to_datetime(df["除权日"])
                self._dividend_data = df
                logger.info(f"Fetched {len(df)} dividend records")
                return df
        except Exception as e:
            logger.warning(f"Failed to fetch dividend data: {e}")

        return pd.DataFrame()

    def extract(self, stock_code: str, **kwargs) -> pd.DataFrame:
        """Extract company events features for a stock.

        Args:
            stock_code: Stock code (e.g., 000001.SZ)

        Returns:
            DataFrame with company events features
        """
        days = kwargs.get("days", 30)

        # 获取日期范围
        end_date = pd.Timestamp.today()
        start_date = end_date - pd.Timedelta(days=days + 60)

        # 创建日期序列 (只包含交易日近似 - 工作日)
        dates = pd.bdate_range(start=start_date, end=end_date, freq="D")
        df = pd.DataFrame({"date": dates})
        df["stock_code"] = stock_code

        code = stock_code.split(".")[0]

        # ========== 1. 分红除权特征 ==========
        df["is_near_dividend"] = 0
        df["days_to_next_dividend"] = 0
        df["days_since_last_dividend"] = 0

        try:
            dividend_df = self._get_dividend_data()
            if not dividend_df.empty:
                # 筛选该股票的分红记录
                stock_dividend = dividend_df[dividend_df["股票代码"] == code]

                if not stock_dividend.empty:
                    # 分红日期
                    dividend_dates = stock_dividend["除权日"].dt.date.unique()

                    # 是否接近除权日 (前后5个交易日)
                    for div_date in dividend_dates:
                        if pd.notna(div_date):
                            mask = (
                                df["date"].dt.date >= div_date - pd.Timedelta(days=7)
                            ) & (df["date"].dt.date <= div_date + pd.Timedelta(days=3))
                            df.loc[mask, "is_near_dividend"] = 1

                    # 距离下一个除权日的天数
                    for i, row in df.iterrows():
                        future_dividends = [
                            d
                            for d in dividend_dates
                            if pd.notna(d) and d > row["date"].date()
                        ]
                        if future_dividends:
                            df.loc[i, "days_to_next_dividend"] = (
                                min(future_dividends) - row["date"].date()
                            ).days

                    # 距离上一个除权日的天数
                    for i, row in df.iterrows():
                        past_dividends = [
                            d
                            for d in dividend_dates
                            if pd.notna(d) and d < row["date"].date()
                        ]
                        if past_dividends:
                            df.loc[i, "days_since_last_dividend"] = (
                                row["date"].date() - max(past_dividends)
                            ).days
        except Exception as e:
            logger.debug(f"Failed to process dividend data: {e}")

        # ========== 2. 财报发布特征 ==========
        df["is_earnings_season"] = 0
        df["is_near_earnings"] = 0

        # 财报季标记 (3-4月年报/一季报, 7-8月半年报, 10月三季报)
        df.loc[df["date"].dt.month.isin([3, 4]), "is_earnings_season"] = 1
        df.loc[df["date"].dt.month.isin([7, 8]), "is_earnings_season"] = 1
        df.loc[df["date"].dt.month == 10, "is_earnings_season"] = 1

        # 财报截止日前后标记
        earnings_deadlines = [
            (4, 30),  # 年报/一季报截止
            (8, 31),  # 半年报截止
            (10, 31),  # 三季报截止
        ]

        for month, day in earnings_deadlines:
            for year in df["date"].dt.year.unique():
                try:
                    deadline = pd.Timestamp(year=year, month=month, day=day)
                    mask = (df["date"] >= deadline - pd.Timedelta(days=10)) & (
                        df["date"] <= deadline + pd.Timedelta(days=5)
                    )
                    df.loc[mask, "is_near_earnings"] = 1
                except (ValueError, TypeError) as e:
                    logger.debug(
                        f"Failed to process earnings date for {year}-{month}-{day}: {e}"
                    )

        # ========== 3. 月末/月初特征 ==========
        df["is_month_end"] = (df["date"].dt.day >= 25).astype(int)
        df["is_month_start"] = (df["date"].dt.day <= 5).astype(int)
        df["is_quarter_end"] = (
            (df["date"].dt.month.isin([3, 6, 9, 12])) & (df["date"].dt.day >= 25)
        ).astype(int)

        # ========== 4. 星期特征 ==========
        df["day_of_week"] = df["date"].dt.dayofweek
        df["is_monday"] = (df["day_of_week"] == 0).astype(int)
        df["is_friday"] = (df["day_of_week"] == 4).astype(int)

        # ========== 5. 距离特殊日期的天数 ==========
        df["days_to_month_end"] = df["date"].dt.days_in_month - df["date"].dt.day

        # ========== 6. 年内特殊时期 ==========
        # 春节前后 (1-2月)
        df["is_near_spring_festival"] = df["date"].dt.month.isin([1, 2]).astype(int)

        # 国庆前后 (9月底-10月)
        df["is_near_national_day"] = (
            (df["date"].dt.month == 9) & (df["date"].dt.day >= 25)
            | (df["date"].dt.month == 10)
        ).astype(int)

        # ========== 7. 季度特征 ==========
        df["quarter"] = df["date"].dt.quarter

        # 填充空值
        df = df.fillna(0)

        return df
