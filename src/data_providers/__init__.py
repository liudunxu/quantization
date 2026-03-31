"""Data providers module.

Multi-provider stock data fetching with automatic fallback:
- yfinance: Global stocks (US, HK, etc.)
- akshare: Chinese A-shares and HK stocks
- tushare: Chinese A-shares/HK (requires TUSHARE_TOKEN env var)
- baostock: Chinese A-shares (free, high-quality)
- openbb: Global stocks via OpenBB ODP (multi-source)
"""

from .base import BaseDataProvider
from .yfinance_provider import YFinanceProvider
from .akshare_provider import AKShareProvider
from .tushare_provider import TushareProvider
from .baostock_provider import BaostockProvider
from .openbb_provider import OpenBBProvider
from .fetch_stock_data import (
    StockDataFetcher,
    get_data_fetcher,
    fetch_stock_data,
    fetch_realtime_price,
)

__all__ = [
    "BaseDataProvider",
    "YFinanceProvider",
    "AKShareProvider",
    "TushareProvider",
    "BaostockProvider",
    "OpenBBProvider",
    "StockDataFetcher",
    "get_data_fetcher",
    "fetch_stock_data",
    "fetch_realtime_price",
]
