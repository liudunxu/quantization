"""Yahoo Finance data provider with retry logic."""

import logging
import time

import pandas as pd
import yfinance as yf

from .base import BaseDataProvider

logger = logging.getLogger(__name__)


class YFinanceProvider(BaseDataProvider):
    """Yahoo Finance data provider with retry logic."""

    def __init__(self):
        """Initialize YFinance provider."""
        self._name = "yfinance"

    @property
    def name(self) -> str:
        """Return provider name."""
        return self._name

    def _convert_symbol(self, stock_code: str) -> str:
        """Convert stock code to yfinance format.

        A-share SH suffix -> SS (yfinance uses .SS for Shanghai)
        """
        if stock_code.endswith(".SH"):
            return stock_code.replace(".SH", ".SS")
        return stock_code

    def fetch(
        self,
        stock_code: str,
        days: int = 120,
        retry_count: int = 3,
        retry_delay: float = 1.0,
    ) -> pd.DataFrame:
        """Fetch stock data from Yahoo Finance with retry logic."""
        last_error = None
        symbol = self._convert_symbol(stock_code)

        for attempt in range(retry_count):
            try:
                data = yf.download(
                    symbol, period=f"{days}d", auto_adjust=False, progress=False, timeout=5
                )

                if data.empty:
                    last_error = "Empty data returned"
                    if attempt < retry_count - 1:
                        logger.warning(
                            f"[{self.name}] Attempt {attempt + 1}/{retry_count} "
                            f"failed for {stock_code}: {last_error}. Retrying..."
                        )
                        time.sleep(retry_delay)
                        continue
                    break

                # Flatten multi-level columns if present
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = [col[0] for col in data.columns]

                # Ensure date is index
                if "Date" in data.columns:
                    data = data.set_index("Date")
                elif "date" not in data.index.name.lower():
                    data.index = pd.to_datetime(data.index)

                # Build result DataFrame with required columns
                df = pd.DataFrame(index=data.index)
                df.index.name = "date"

                # Handle both DataFrame and Series cases for each column
                close = data["Close"]
                df["close"] = (
                    close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close
                )

                open_ = data["Open"]
                df["open"] = (
                    open_.iloc[:, 0] if isinstance(open_, pd.DataFrame) else open_
                )

                high = data["High"]
                df["high"] = high.iloc[:, 0] if isinstance(high, pd.DataFrame) else high

                low = data["Low"]
                df["low"] = low.iloc[:, 0] if isinstance(low, pd.DataFrame) else low

                volume = data["Volume"]
                df["volume"] = (
                    volume.iloc[:, 0] if isinstance(volume, pd.DataFrame) else volume
                )

                df = df.reset_index()

                if self._validate_data(df):
                    logger.info(
                        f"[{self.name}] Successfully fetched {len(df)} rows for {stock_code}"
                    )
                    return df

                last_error = "Missing required columns after processing"
                if attempt < retry_count - 1:
                    logger.warning(
                        f"[{self.name}] Attempt {attempt + 1}/{retry_count} "
                        f"failed for {stock_code}: {last_error}. Retrying..."
                    )
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
