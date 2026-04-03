"""AKShare data provider for Chinese stocks."""

import logging
import time

import pandas as pd

from .base import BaseDataProvider

logger = logging.getLogger(__name__)

# Lazy import akshare to make it optional
akshare = None


def _get_akshare():
    """Lazy load akshare."""
    global akshare
    if akshare is None:
        try:
            import akshare as akshare_pkg

            akshare = akshare_pkg
        except ImportError:
            logger.warning(
                "[akshare] akshare not installed. Install with: pip install akshare"
            )
            return None
    return akshare


class AKShareProvider(BaseDataProvider):
    """AKShare data provider for Chinese A-shares and HK stocks."""

    def __init__(self):
        self._name = "akshare"

    @property
    def name(self) -> str:
        return self._name

    def _convert_a_share_code(self, stock_code: str) -> tuple:
        """Convert standard code to akshare format.

        Returns:
            tuple: (symbol, exchange) e.g., ('000001', 'sz') or ('600519', 'sh')
        """
        code = stock_code.split(".")[0]

        if stock_code.endswith(".SH") or stock_code.endswith(".SS"):
            return code, "sh"
        elif stock_code.endswith(".SZ"):
            return code, "sz"
        elif stock_code.endswith(".HK"):
            # Remove leading zeros from HK stock codes
            return code.lstrip("0") or "0", "hk"
        return code, None

    def fetch(
        self,
        stock_code: str,
        days: int = 120,
        retry_count: int = 3,
        retry_delay: float = 1.0,
    ) -> pd.DataFrame:
        """Fetch stock data from AKShare."""
        akshare_lib = _get_akshare()
        if akshare_lib is None:
            return pd.DataFrame()

        symbol, exchange = self._convert_a_share_code(stock_code)

        if exchange is None:
            logger.warning(f"[{self.name}] Unknown exchange for {stock_code}")
            return pd.DataFrame()

        last_error = None

        # Calculate date range based on days needed (add buffer for indicators)
        start_date = (pd.Timestamp.today() - pd.Timedelta(days=days * 2)).strftime(
            "%Y%m%d"
        )
        end_date = pd.Timestamp.today().strftime("%Y%m%d")

        for attempt in range(retry_count):
            try:
                if exchange == "hk":
                    # HK stock
                    df = akshare_lib.stock_hk_hist(
                        symbol=symbol,
                        period="daily",
                        start_date=start_date,
                        end_date=end_date,
                        adjust="",
                    )
                else:
                    # A-share
                    df = akshare_lib.stock_zh_a_hist(
                        symbol=symbol,
                        period="daily",
                        start_date=start_date,
                        end_date=end_date,
                        adjust="qfq",
                    )

                if df is None or df.empty:
                    last_error = "Empty data returned"
                    if attempt < retry_count - 1:
                        time.sleep(retry_delay)
                        continue
                    break

                # Rename columns to standard names
                # AKShare A-share columns: 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅, 涨跌幅, 涨跌额,换手率
                # HK columns: 日期, 开盘, 收盘, 最高, 最低, 成交量
                column_mapping = {
                    "日期": "date",
                    "开盘": "open",
                    "收盘": "close",
                    "最高": "high",
                    "最低": "low",
                    "成交量": "volume",
                }

                # Handle MultiIndex if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] for col in df.columns]

                df = df.rename(columns=column_mapping)

                # Keep only required columns
                required = ["date", "open", "close", "high", "low", "volume"]
                df = df[[c for c in required if c in df.columns]]

                if not self._validate_data(df):
                    last_error = "Missing required columns"
                    if attempt < retry_count - 1:
                        time.sleep(retry_delay)
                        continue
                    break

                # Convert date to datetime
                df["date"] = pd.to_datetime(df["date"])

                # Sort by date and take last N days
                df = df.sort_values("date").tail(days)

                logger.info(
                    f"[{self.name}] Successfully fetched {len(df)} rows for {stock_code}"
                )
                return df.reset_index(drop=True)

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"[{self.name}] Attempt {attempt + 1}/{retry_count} "
                    f"failed for {stock_code}: {last_error}. Retrying..."
                )
                if attempt < retry_count - 1:
                    time.sleep(retry_delay)

        logger.error(
            f"[{self.name}] All {retry_count} attempts failed for {stock_code}: {last_error}"
        )
        return pd.DataFrame()

    def fetch_index(
        self,
        index_code: str,
        days: int = 120,
        retry_count: int = 3,
        retry_delay: float = 1.0,
    ) -> pd.DataFrame:
        """Fetch A-share index data from AKShare.

        Args:
            index_code: Index code (e.g., '000001' for 上证指数, '399001' for 深证成指, '399006' for 创业板指)
            days: Number of days to fetch
            retry_count: Number of retry attempts
            retry_delay: Delay between retries

        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        akshare_lib = _get_akshare()
        if akshare_lib is None:
            return pd.DataFrame()

        last_error = None

        for attempt in range(retry_count):
            try:
                df = akshare_lib.stock_zh_index_daily(
                    symbol=f"sh{index_code}"
                    if index_code.startswith("000") or index_code.startswith("688")
                    else f"sz{index_code}"
                )
                if df is None or df.empty:
                    last_error = "Empty data returned"
                    if attempt < retry_count - 1:
                        time.sleep(retry_delay)
                        continue
                    break

                column_mapping = {
                    "date": "date",
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume",
                }
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] for col in df.columns]
                df = df.rename(columns=column_mapping)

                required = ["date", "open", "close", "high", "low", "volume"]
                df = df[[c for c in required if c in df.columns]]

                if not self._validate_data(df):
                    last_error = "Missing required columns"
                    if attempt < retry_count - 1:
                        time.sleep(retry_delay)
                        continue
                    break

                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").tail(days)
                logger.info(
                    f"[{self.name}] Successfully fetched {len(df)} rows for index {index_code}"
                )
                return df.reset_index(drop=True)

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"[{self.name}] Attempt {attempt + 1}/{retry_count} failed for index {index_code}: {last_error}"
                )
                if attempt < retry_count - 1:
                    time.sleep(retry_delay)

        logger.error(
            f"[{self.name}] All {retry_count} attempts failed for index {index_code}: {last_error}"
        )
        return pd.DataFrame()

    def fetch_southbound_flow(self, days: int = 120) -> pd.DataFrame:
        """Fetch southbound capital flow data (港股通资金流向).

        Args:
            days: Number of days to fetch

        Returns:
            DataFrame with columns: date, southbound_net_buy, southbound_buy,
                                    southbound_sell, southbound_net_flow
        """
        akshare_lib = _get_akshare()
        if akshare_lib is None:
            logger.warning("[akshare] akshare not installed")
            return pd.DataFrame()

        try:
            # 获取沪港通-港股通沪数据
            df_sh = akshare_lib.stock_hsgt_hist_em(symbol="港股通沪")
            # 获取深港通-港股通深数据
            df_sz = akshare_lib.stock_hsgt_hist_em(symbol="港股通深")

            if df_sh is None or df_sh.empty:
                logger.warning("[akshare] No southbound flow data (港股通沪)")
                return pd.DataFrame()

            # 合并沪深港股通数据
            result = pd.DataFrame()
            result["date"] = pd.to_datetime(df_sh["日期"])

            # 沪港通-港股通数据
            result["southbound_sh_net_buy"] = df_sh["当日成交净买额"].values
            result["southbound_sh_buy"] = df_sh["买入成交额"].values
            result["southbound_sh_sell"] = df_sh["卖出成交额"].values

            # 深港通-港股通数据
            if df_sz is not None and not df_sz.empty:
                df_sz["日期"] = pd.to_datetime(df_sz["日期"])
                # 合并数据
                result = result.merge(
                    df_sz[["日期", "当日成交净买额", "买入成交额", "卖出成交额"]],
                    left_on="date",
                    right_on="日期",
                    how="left",
                    suffixes=("", "_sz"),
                )
                result["southbound_sz_net_buy"] = result["当日成交净买额_sz"].fillna(0)
                result["southbound_sz_buy"] = result["买入成交额_sz"].fillna(0)
                result["southbound_sz_sell"] = result["卖出成交额_sz"].fillna(0)
                # 清理多余列
                result = result.drop(
                    columns=[
                        "日期",
                        "当日成交净买额_sz",
                        "买入成交额_sz",
                        "卖出成交额_sz",
                    ],
                    errors="ignore",
                )
            else:
                result["southbound_sz_net_buy"] = 0
                result["southbound_sz_buy"] = 0
                result["southbound_sz_sell"] = 0

            # 计算总南向资金
            result["southbound_net_buy"] = (
                result["southbound_sh_net_buy"] + result["southbound_sz_net_buy"]
            )
            result["southbound_buy"] = (
                result["southbound_sh_buy"] + result["southbound_sz_buy"]
            )
            result["southbound_sell"] = (
                result["southbound_sh_sell"] + result["southbound_sz_sell"]
            )

            # 排序并取最近N天
            result = result.sort_values("date").tail(days)

            logger.info(
                f"[{self.name}] Successfully fetched {len(result)} days of southbound flow data"
            )
            return result.reset_index(drop=True)

        except Exception as e:
            logger.error(f"[akshare] Failed to fetch southbound flow: {e}")
            return pd.DataFrame()
