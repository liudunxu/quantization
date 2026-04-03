"""Tushare data provider for Chinese stocks."""

import logging
import os
import time
from typing import Optional

import pandas as pd

from .base import BaseDataProvider

logger = logging.getLogger(__name__)

# Lazy import tushare to make it optional
tushare = None


def _get_tushare():
    """Lazy load tushare."""
    global tushare
    if tushare is None:
        try:
            import tushare as ts
            tushare = ts
        except ImportError:
            logger.warning("[tushare] tushare not installed. Install with: pip install tushare")
            return None
    return tushare


class TushareProvider(BaseDataProvider):
    """Tushare data provider for Chinese A-shares and HK stocks."""

    def __init__(self, token: Optional[str] = None):
        """Initialize Tushare provider.

        Args:
            token: Tushare API token. If not provided, tries TUSHARE_TOKEN env var.
        """
        self._name = "tushare"
        self._token = token or os.environ.get('TUSHARE_TOKEN')
        self._pro = None

    @property
    def name(self) -> str:
        return self._name

    def _init_pro(self):
        """Initialize Tushare Pro API."""
        if self._pro is None:
            ts = _get_tushare()
            if ts is None:
                return False
            if self._token:
                ts.set_token(self._token)
                self._pro = ts.pro_api()
            else:
                # Use basic tushare without token
                self._pro = None
        return True

    def _convert_code(self, stock_code: str) -> tuple:
        """Convert standard code to tushare format.

        Returns:
            tuple: (ts_code, api_code) e.g., ('000001.SZ', 'SZ.000001')
        """
        code = stock_code.split('.')[0]
        suffix = stock_code.upper().split('.')[-1] if '.' in stock_code else ''

        if suffix == 'SH' or suffix == 'SS':
            return f'{code}.SH', 'SH.{code}'
        elif suffix == 'SZ':
            return f'{code}.SZ', 'SZ.{code}'
        elif suffix == 'HK':
            # Remove leading zeros from HK stock codes
            code_clean = code.lstrip('0') or '0'
            return f'{code_clean}.HK', 'HK.{code_clean}'
        return stock_code, None

    def fetch(
        self,
        stock_code: str,
        days: int = 120,
        retry_count: int = 3,
        retry_delay: float = 1.0
    ) -> pd.DataFrame:
        """Fetch stock data from Tushare."""
        if not self._init_pro():
            return pd.DataFrame()

        ts_code, api_code = self._convert_code(stock_code)
        if api_code is None:
            logger.warning(f"[{self.name}] Unknown exchange for {stock_code}")
            return pd.DataFrame()

        ts = _get_tushare()
        last_error = None

        for attempt in range(retry_count):
            try:
                # Calculate date range
                end_date = pd.Timestamp.today().strftime('%Y%m%d')
                start_date = (pd.Timestamp.today() - pd.Timedelta(days=days * 2)).strftime('%Y%m%d')

                if 'HK' in stock_code:
                    # HK stock
                    df = ts.hk_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
                else:
                    # A-share
                    df = ts.pro_bar(
                        ts_code=ts_code,
                        start_date=start_date,
                        end_date=end_date,
                        adj='qfq'
                    )

                if df is None or df.empty:
                    last_error = "Empty data returned"
                    if attempt < retry_count - 1:
                        time.sleep(retry_delay)
                        continue
                    break

                # Rename columns to standard names
                # Tushare columns: ts_code, trade_date, open, high, low, close, vol
                column_mapping = {
                    'trade_date': 'date',
                    'vol': 'volume',
                }

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
