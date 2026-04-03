"""Data providers module.

Multi-provider stock data fetching with automatic fallback:
- yfinance: Global stocks (US, HK, etc.)
- akshare: Chinese A-shares and HK stocks
- tushare: Chinese A-shares/HK (requires TUSHARE_TOKEN env var)
- baostock: Chinese A-shares (free, high-quality)
- openbb: Global stocks via OpenBB ODP (multi-source)
"""

from .akshare_provider import AKShareProvider
from .baostock_provider import BaostockProvider
from .base import BaseDataProvider
from .fetch_stock_data import (
    StockDataFetcher,
    fetch_realtime_price,
    fetch_stock_data,
    get_data_fetcher,
)
from .openbb_provider import OpenBBProvider
from .sentiment_provider import SentimentProvider
from .tushare_provider import TushareProvider
from .yfinance_provider import YFinanceProvider

__all__ = [
    "BaseDataProvider",
    "YFinanceProvider",
    "AKShareProvider",
    "TushareProvider",
    "BaostockProvider",
    "OpenBBProvider",
    "SentimentProvider",
    "StockDataFetcher",
    "get_data_fetcher",
    "fetch_stock_data",
    "fetch_realtime_price",
]
