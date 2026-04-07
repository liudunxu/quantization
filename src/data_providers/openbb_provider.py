"""OpenBB ODP data provider for global stocks."""

import logging
import time
from typing import Optional

import pandas as pd

from .base import BaseDataProvider

logger = logging.getLogger(__name__)

# Lazy import openbb
_obb = None


def _get_openbb():
    """Lazy load OpenBB."""
    global _obb
    if _obb is None:
        try:
            from openbb import obb

            _obb = obb
        except ImportError:
            logger.warning("[openbb] Not installed. Install with: pip install openbb")
            return None
    return _obb


class OpenBBProvider(BaseDataProvider):
    """OpenBB ODP data provider with multi-source support.

    OpenBB provides access to multiple data providers through a unified interface:
    - yfinance (default)
    - fmp
    - alpha_vantage
    - polygon
    - and many more

    Supports stocks worldwide including US, HK, and A-shares.
    """

    def __init__(self, default_provider: str = "yfinance"):
        """Initialize OpenBB provider.

        Args:
            default_provider: Default data source within OpenBB
                Options: 'yfinance', 'fmp', 'alpha_vantage', 'polygon', etc.
        """
        self._name = "openbb"
        self.default_provider = default_provider

    @property
    def name(self) -> str:
        """Return provider name."""
        return self._name

    def _convert_symbol(self, stock_code: str) -> str:
        """Convert stock code to OpenBB format.

        OpenBB uses the same format as yfinance for most markets.
        """
        return stock_code

    def fetch(
        self,
        stock_code: str,
        days: int = 120,
        retry_count: int = 3,
        retry_delay: float = 1.0,
    ) -> pd.DataFrame:
        """Fetch stock data from OpenBB.

        Args:
            stock_code: Stock code (e.g., 'AAPL', '0700.HK', '000001.SZ')
            days: Number of days to fetch
            retry_count: Number of retries on failure
            retry_delay: Delay between retries in seconds

        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        obb = _get_openbb()
        if obb is None:
            return pd.DataFrame()

        last_error = None

        for attempt in range(retry_count):
            try:
                symbol = self._convert_symbol(stock_code)

                # Calculate date range
                start_date = (
                    pd.Timestamp.today() - pd.Timedelta(days=days + 30)
                ).strftime("%Y-%m-%d")
                end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

                # Try to fetch historical data
                # OpenBB will automatically fallback to available providers
                data = obb.equity.price.historical(
                    symbol=symbol,
                    provider=self.default_provider,
                    start_date=start_date,
                    end_date=end_date,
                    interval="1d",
                )

                if data is None or not hasattr(data, "to_df"):
                    last_error = "No data returned"
                    logger.warning(f"[{self.name}] {last_error} for {stock_code}")
                    if attempt < retry_count - 1:
                        time.sleep(retry_delay)
                        continue
                    break

                # Convert to DataFrame
                df = data.to_df()

                if df.empty:
                    last_error = "Empty DataFrame returned"
                    logger.warning(f"[{self.name}] {last_error} for {stock_code}")
                    if attempt < retry_count - 1:
                        time.sleep(retry_delay)
                        continue
                    break

                # Reset index to get date as column
                df = df.reset_index()

                # Standardize column names
                column_mapping = {
                    "date": "date",
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume",
                }

                # Rename columns if needed
                for old_col, new_col in column_mapping.items():
                    if old_col in df.columns:
                        df = df.rename(columns={old_col: new_col})

                # Handle case where date is in index
                if "date" not in df.columns and df.index.name == "date":
                    df = df.reset_index()

                # Keep only required columns
                required_cols = ["date", "open", "high", "low", "close", "volume"]
                available_cols = [col for col in required_cols if col in df.columns]
                df = df[available_cols]

                # Convert date to datetime
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])

                # Sort by date and take last N days
                if "date" in df.columns:
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

    def fetch_realtime_price(self, stock_code: str) -> Optional[float]:
        """Fetch real-time price using OpenBB.

        Args:
            stock_code: Stock code

        Returns:
            Real-time price or None if failed
        """
        obb = _get_openbb()
        if obb is None:
            return None

        try:
            symbol = self._convert_symbol(stock_code)

            # Get current quote
            quote = obb.equity.price.quote(
                symbol=symbol, provider=self.default_provider
            )

            if quote and hasattr(quote, "to_df"):
                df = quote.to_df()
                if not df.empty and "last_price" in df.columns:
                    price = float(df["last_price"].iloc[0])
                    if price > 0:
                        logger.info(
                            f"[{self.name}] Got realtime price {price} for {stock_code}"
                        )
                        return price

            return None

        except Exception as e:
            logger.warning(
                f"[{self.name}] Failed to fetch realtime price for {stock_code}: {e}"
            )
            return None
