"""AKShare data provider for Chinese stocks."""

import time
import logging
from typing import Optional
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
            logger.warning("[akshare] akshare not installed. Install with: pip install akshare")
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
        code = stock_code.split('.')[0]

        if stock_code.endswith('.SH') or stock_code.endswith('.SS'):
            return code, 'sh'
        elif stock_code.endswith('.SZ'):
            return code, 'sz'
        elif stock_code.endswith('.HK'):
            # Remove leading zeros from HK stock codes
            return code.lstrip('0') or '0', 'hk'
        return code, None

    def fetch(
        self,
        stock_code: str,
        days: int = 120,
        retry_count: int = 3,
        retry_delay: float = 1.0
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

        for attempt in range(retry_count):
            try:
                if exchange == 'hk':
                    # HK stock
                    df = akshare_lib.stock_hk_hist(
                        symbol=symbol,
                        period="daily",
                        start_date="20200101",
                        end_date="20300101",
                        adjust=""
                    )
                else:
                    # A-share
                    df = akshare_lib.stock_zh_a_hist(
                        symbol=symbol,
                        period="daily",
                        start_date="20200101",
                        end_date="20300101",
                        adjust="qfq"
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
                    '日期': 'date',
                    '开盘': 'open',
                    '收盘': 'close',
                    '最高': 'high',
                    '最低': 'low',
                    '成交量': 'volume',
                }

                # Handle MultiIndex if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] for col in df.columns]

                df = df.rename(columns=column_mapping)

                # Keep only required columns
                required = ['date', 'open', 'close', 'high', 'low', 'volume']
                df = df[[c for c in required if c in df.columns]]

                if not self._validate_data(df):
                    last_error = "Missing required columns"
                    if attempt < retry_count - 1:
                        time.sleep(retry_delay)
                        continue
                    break

                # Convert date to datetime
                df['date'] = pd.to_datetime(df['date'])

                # Sort by date and take last N days
                df = df.sort_values('date').tail(days)

                logger.info(f"[{self.name}] Successfully fetched {len(df)} rows for {stock_code}")
                return df.reset_index(drop=True)

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"[{self.name}] Attempt {attempt + 1}/{retry_count} "
                    f"failed for {stock_code}: {last_error}. Retrying..."
                )
                if attempt < retry_count - 1:
                    time.sleep(retry_delay)

        logger.error(f"[{self.name}] All {retry_count} attempts failed for {stock_code}: {last_error}")
        return pd.DataFrame()
