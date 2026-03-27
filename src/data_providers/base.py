"""Base data provider interface."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import pandas as pd


class BaseDataProvider(ABC):
    """Base class for stock data providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass

    @abstractmethod
    def fetch(
        self,
        stock_code: str,
        days: int = 120,
        retry_count: int = 3,
        retry_delay: float = 1.0
    ) -> pd.DataFrame:
        """Fetch stock data with retry logic.

        Args:
            stock_code: Stock code (e.g., 000001.SZ, 0700.HK, AAPL)
            days: Number of days to fetch
            retry_count: Number of retries on failure
            retry_delay: Delay between retries in seconds

        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        pass

    def _validate_data(self, df: pd.DataFrame) -> bool:
        """Validate that data has required columns and is not empty."""
        if df.empty:
            return False
        required_cols = ['close', 'open', 'high', 'low', 'volume']
        return all(col in df.columns for col in required_cols)
