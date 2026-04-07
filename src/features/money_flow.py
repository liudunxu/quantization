"""Money flow features extractor for Chinese stocks."""

from typing import TYPE_CHECKING, Optional

import pandas as pd

from .base import BaseFeatureExtractor

if TYPE_CHECKING:
    from ..utils.cache import FeatureCache


class MoneyFlowFeatures(BaseFeatureExtractor):
    """Extract money flow features for stocks.

    Uses akshare to fetch:
    - 主力资金净流入 (Main force net inflow)
    - 超大单/大单/中单/小单 净流入
    - 换手率 (Turnover rate) - from stock hist data
    """

    def __init__(self, cache: Optional["FeatureCache"] = None):
        """Initialize MoneyFlowFeatures."""
        super().__init__(cache)
        self._akshare = None

    def _get_akshare(self):
        """Lazy load akshare."""
        if self._akshare is None:
            try:
                import akshare as akshare_pkg

                self._akshare = akshare_pkg
            except ImportError:
                return None
        return self._akshare

    @property
    def feature_type(self) -> str:
        """Return feature type name."""
        return "money_flow"

    def extract(self, stock_code: str, **kwargs) -> pd.DataFrame:
        """Extract money flow features for a stock.

        Args:
            stock_code: Stock code (e.g., '000001.SZ', '600519.SH', 'MU')
            days: Number of days to fetch
        """
        days = kwargs.get("days", 30)

        # Determine market
        if stock_code.endswith(".HK"):
            # HK stocks don't have easy money flow data via akshare
            return self._extract_hk_turnover(stock_code, days)
        elif (
            stock_code.endswith(".SZ")
            or stock_code.endswith(".SH")
            or stock_code.endswith(".SS")
        ):
            # A-shares - use money flow API
            return self._extract_a_share_money_flow(stock_code, days)
        else:
            # US stocks or other markets - no money flow data available
            return pd.DataFrame()

    def _extract_a_share_money_flow(self, stock_code: str, days: int) -> pd.DataFrame:
        """Extract money flow data for A-shares."""
        akshare = self._get_akshare()
        if akshare is None:
            return pd.DataFrame()

        # Parse stock code
        symbol = stock_code.split(".")[0]
        if stock_code.endswith(".SH") or stock_code.endswith(".SS"):
            market = "sh"
        elif stock_code.endswith(".SZ"):
            market = "sz"
        else:
            market = "sh"  # Default to Shanghai

        try:
            # Fetch money flow data
            mf_df = akshare.stock_individual_fund_flow(stock=symbol, market=market)

            if mf_df is None or mf_df.empty:
                return pd.DataFrame()

            # Parse date
            mf_df["date"] = pd.to_datetime(mf_df["日期"])
            mf_df = mf_df.sort_values("date")

            # Take last N days
            mf_df = mf_df.tail(days * 2)  # Buffer for calculations

            # Rename and select columns
            result = pd.DataFrame()
            result["date"] = mf_df["date"]
            result["stock_code"] = stock_code

            # Main force net inflow (unit: 100 million)
            result["main_net_flow"] = mf_df["主力净流入-净额"] / 1e8
            result["main_net_flow_ratio"] = (
                mf_df["主力净流入-净占比"] / 100
            )  # Convert to ratio

            # Super large order
            result["super_large_net_flow"] = mf_df["超大单净流入-净额"] / 1e8
            result["super_large_net_flow_ratio"] = mf_df["超大单净流入-净占比"] / 100

            # Large order
            result["large_net_flow"] = mf_df["大单净流入-净额"] / 1e8
            result["large_net_flow_ratio"] = mf_df["大单净流入-净占比"] / 100

            # Medium order
            result["medium_net_flow"] = mf_df["中单净流入-净额"] / 1e8
            result["medium_net_flow_ratio"] = mf_df["中单净流入-净占比"] / 100

            # Small order (retail investors)
            result["small_net_flow"] = mf_df["小单净流入-净额"] / 1e8
            result["small_net_flow_ratio"] = mf_df["小单净流入-净占比"] / 100

            # Money flow momentum (3-day sum)
            for col in ["main_net_flow", "large_net_flow", "small_net_flow"]:
                result[f"{col}_momentum_3d"] = result[col].rolling(window=3).sum()

            # Money flow trend (5-day moving average of ratio)
            result["main_net_flow_ratio_ma5"] = (
                result["main_net_flow_ratio"].rolling(window=5).mean()
            )

            # Institutional vs Retail ratio
            result["institutional_ratio"] = (
                result["main_net_flow_ratio"] + result["super_large_net_flow_ratio"]
            )

            # Net flow volatility (10-day std)
            result["main_net_flow_volatility"] = (
                result["main_net_flow"].rolling(window=10).std()
            )

            # Fill NaN
            result = result.ffill().bfill()

            return result

        except Exception as e:
            import logging

            logging.warning(
                f"[MoneyFlowFeatures] Failed to fetch money flow for {stock_code}: {e}"
            )
            return pd.DataFrame()

    def _extract_hk_turnover(self, stock_code: str, days: int) -> pd.DataFrame:
        """Extract basic turnover data for HK stocks.

        HK stocks don't have easy money flow data, but we can still get
        some volume-related features from the main stock data.
        """
        # This will be handled by the main data fetcher with volume features
        # For now, return empty - the technical features already include volume
        return pd.DataFrame()
