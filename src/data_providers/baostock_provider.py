"""BaoStock data provider for Chinese A-shares."""

import time
import logging
from typing import Optional
import pandas as pd
from .base import BaseDataProvider

logger = logging.getLogger(__name__)

# Lazy import baostock
_baostock = None


def _get_baostock():
    """Lazy load baostock."""
    global _baostock
    if _baostock is None:
        try:
            import baostock as bs

            _baostock = bs
        except ImportError:
            logger.warning(
                "[baostock] Not installed. Install with: pip install baostock"
            )
            return None
    return _baostock


class BaostockProvider(BaseDataProvider):
    """BaoStock data provider for Chinese A-shares.

    BaoStock provides free, high-quality A-share data with:
    - Daily K-line data
    - Forward/backward adjusted prices
    - Historical data going back many years
    """

    def __init__(self, adjustflag: str = "2"):
        """Initialize Baostock provider.

        Args:
            adjustflag: Price adjustment flag
                - "1": Backward adjusted (后复权)
                - "2": Forward adjusted (前复权) - recommended
                - "3": No adjustment (不复权)
        """
        self._name = "baostock"
        self.adjustflag = adjustflag

    @property
    def name(self) -> str:
        return self._name

    def _convert_code(self, stock_code: str) -> str:
        """Convert stock code to baostock format.

        Examples:
            000001.SZ -> sz.000001
            600519.SH -> sh.600519
        """
        code = stock_code.split(".")[0]
        if stock_code.endswith(".SH"):
            return f"sh.{code}"
        elif stock_code.endswith(".SZ"):
            return f"sz.{code}"
        return stock_code

    def _is_a_share(self, stock_code: str) -> bool:
        """Check if stock code is A-share."""
        return stock_code.endswith(".SH") or stock_code.endswith(".SZ")

    def fetch(
        self,
        stock_code: str,
        days: int = 120,
        retry_count: int = 3,
        retry_delay: float = 1.0,
    ) -> pd.DataFrame:
        """Fetch stock data from BaoStock.

        Args:
            stock_code: Stock code (e.g., '000001.SZ', '600519.SH')
            days: Number of days to fetch
            retry_count: Number of retries on failure
            retry_delay: Delay between retries in seconds

        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        # Only support A-shares
        if not self._is_a_share(stock_code):
            logger.debug(f"[{self.name}] Only supports A-shares, skipping {stock_code}")
            return pd.DataFrame()

        bs = _get_baostock()
        if bs is None:
            return pd.DataFrame()

        last_error = None

        for attempt in range(retry_count):
            try:
                # Login to baostock
                login_result = bs.login()
                if login_result.error_code != "0":
                    last_error = f"Login failed: {login_result.error_msg}"
                    logger.warning(f"[{self.name}] {last_error}")
                    if attempt < retry_count - 1:
                        time.sleep(retry_delay)
                        continue
                    break

                # Convert stock code
                code = self._convert_code(stock_code)

                # Calculate date range (add buffer for indicators)
                start_date = (
                    pd.Timestamp.today() - pd.Timedelta(days=days * 2)
                ).strftime("%Y-%m-%d")
                end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

                # Query historical data
                rs = bs.query_history_k_data_plus(
                    code,
                    "date,open,high,low,close,volume",
                    start_date=start_date,
                    end_date=end_date,
                    frequency="d",
                    adjustflag=self.adjustflag,
                )

                if rs.error_code != "0":
                    last_error = f"Query failed: {rs.error_msg}"
                    logger.warning(f"[{self.name}] {last_error}")
                    bs.logout()
                    if attempt < retry_count - 1:
                        time.sleep(retry_delay)
                        continue
                    break

                # Parse results
                data_list = []
                while (rs.error_code == "0") & rs.next():
                    data_list.append(rs.get_row_data())

                # Logout
                bs.logout()

                if not data_list:
                    last_error = "No data returned"
                    logger.warning(f"[{self.name}] {last_error} for {stock_code}")
                    if attempt < retry_count - 1:
                        time.sleep(retry_delay)
                        continue
                    break

                # Create DataFrame
                df = pd.DataFrame(data_list, columns=rs.fields)

                # Convert types
                df["date"] = pd.to_datetime(df["date"])
                for col in ["open", "high", "low", "close"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

                # Remove rows with NaN
                df = df.dropna()

                # Sort by date and take last N days
                df = df.sort_values("date").tail(days)

                # Reset index
                df = df.reset_index(drop=True)

                if self._validate_data(df):
                    logger.info(
                        f"[{self.name}] Successfully fetched {len(df)} rows for {stock_code}"
                    )
                    return df

                last_error = "Missing required columns after processing"
                if attempt < retry_count - 1:
                    time.sleep(retry_delay)

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
