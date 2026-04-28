"""Stock data fetcher with multi-provider fallback."""

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import List, Optional

import pandas as pd
import yfinance as yf

from .akshare_provider import AKShareProvider
from .baostock_provider import BaostockProvider
from .base import BaseDataProvider
from .openbb_provider import OpenBBProvider
from .tushare_provider import TushareProvider
from .yfinance_provider import YFinanceProvider

logger = logging.getLogger(__name__)

# Default provider order by market
MARKET_PROVIDER_ORDER = {
    "a_share": ["baostock", "akshare", "tushare", "openbb", "yfinance"],
    "hk": ["yfinance", "openbb", "akshare", "tushare"],
    "us": ["openbb", "yfinance"],
}


def fetch_realtime_price(stock_code: str) -> Optional[float]:
    """Fetch real-time price for a stock.

    Args:
        stock_code: Stock code (e.g., '000001.SZ', 'AAPL', '0700.HK')

    Returns:
        Real-time price or None if failed
    """
    # Try OpenBB first (if available)
    try:
        openbb_provider = OpenBBProvider()
        price = openbb_provider.fetch_realtime_price(stock_code)
        if price is not None:
            return price
    except Exception:
        pass

    # Fallback to yfinance
    try:
        ticker = yf.Ticker(stock_code)
        info = ticker.info

        # Try different price fields
        price_fields = ["currentPrice", "regularMarketPrice", "previousClose"]
        for field in price_fields:
            if field in info and info[field] is not None:
                price = float(info[field])
                if price > 0:
                    logger.info(
                        f"[Realtime] Got price {price} for {stock_code} from {field}"
                    )
                    return price

        # Fallback: try fast_info
        try:
            price = ticker.fast_info.get("lastPrice") or ticker.fast_info.get(
                "previousClose"
            )
            if price and price > 0:
                logger.info(
                    f"[Realtime] Got price {price} for {stock_code} from fast_info"
                )
                return float(price)
        except (AttributeError, TypeError, KeyError) as e:
            logger.debug(f"fast_info fallback failed for {stock_code}: {e}")

        logger.warning(f"[Realtime] No price found for {stock_code}")
        return None

    except Exception as e:
        logger.warning(f"[Realtime] Failed to fetch price for {stock_code}: {e}")
        return None


class StockDataFetcher:
    """Multi-provider stock data fetcher with automatic fallback.

    Provider priority by market:
    - A-shares: baostock → akshare → tushare → openbb → yfinance
    - HK: openbb → yfinance → akshare → tushare
    - US: openbb → yfinance
    """

    def __init__(self, providers: Optional[List[BaseDataProvider]] = None):
        """Initialize with providers list.

        Args:
            providers: List of providers to try in order. If None, uses default order.
        """
        if providers is None:
            self.providers: List[BaseDataProvider] = [
                BaostockProvider(),
                OpenBBProvider(),
                YFinanceProvider(),
                AKShareProvider(),
                TushareProvider(),
            ]
        else:
            self.providers = providers

    def _get_market(self, stock_code: str) -> str:
        """Determine market from stock code."""
        code_upper = stock_code.upper()
        if code_upper.endswith(".SH") or code_upper.endswith(".SZ"):
            return "a_share"
        elif code_upper.endswith(".HK"):
            return "hk"
        else:
            return "us"

    def _get_provider_by_name(self, name: str) -> Optional[BaseDataProvider]:
        """Get provider instance by name."""
        name_map = {
            "baostock": BaostockProvider,
            "openbb": OpenBBProvider,
            "yfinance": YFinanceProvider,
            "akshare": AKShareProvider,
            "tushare": TushareProvider,
        }
        provider_cls = name_map.get(name.lower())
        if provider_cls:
            return provider_cls()
        return None

    def fetch(
        self, stock_code: str, days: int = 120, providers: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Fetch stock data with automatic provider fallback.

        Args:
            stock_code: Stock code (e.g., 000001.SZ, 0700.HK, AAPL)
            days: Number of days to fetch
            providers: Optional list of provider names to use (e.g., ['baostock', 'akshare'])
                     If None, tries all providers in market-specific order.

        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        tried_providers = []
        market = self._get_market(stock_code)

        # Get provider order for this market
        if providers:
            provider_order = providers
        else:
            provider_order = MARKET_PROVIDER_ORDER.get(market, ["yfinance"])

        logger.info(
            f"[StockDataFetcher] Market: {market}, Provider order: {provider_order}"
        )

        for provider_name in provider_order:
            provider = self._get_provider_by_name(provider_name)
            if provider is None:
                logger.debug(
                    f"[StockDataFetcher] Provider {provider_name} not available"
                )
                continue

            tried_providers.append(provider.name)
            logger.info(f"[StockDataFetcher] Trying {provider.name} for {stock_code}")

            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(provider.fetch, stock_code, days, 1, 1.0)
            try:
                df = future.result(timeout=10)
            except FutureTimeoutError:
                logger.warning(
                    f"[StockDataFetcher] {provider.name} timed out after 10s for {stock_code}"
                )
                executor.shutdown(wait=False)
                continue
            finally:
                executor.shutdown(wait=False)

            if not df.empty:
                logger.info(
                    f"[StockDataFetcher] Success with {provider.name} ({len(df)} rows)"
                )
                return df

            logger.warning(
                f"[StockDataFetcher] {provider.name} returned empty data for {stock_code}"
            )

        logger.error(
            f"[StockDataFetcher] All providers failed for {stock_code}. "
            f"Tried: {', '.join(tried_providers)}"
        )
        return pd.DataFrame()


# Global instance
_fetcher: Optional[StockDataFetcher] = None


def get_data_fetcher() -> StockDataFetcher:
    """Get global data fetcher instance."""
    global _fetcher
    if _fetcher is None:
        _fetcher = StockDataFetcher()
    return _fetcher


def fetch_stock_data(
    stock_code: str, days: int = 120, providers: Optional[List[str]] = None
) -> pd.DataFrame:
    """Convenience function to fetch stock data.

    Args:
        stock_code: Stock code
        days: Number of days to fetch
        providers: Optional list of provider names to use

    Returns:
        DataFrame with stock data or empty DataFrame on failure
    """
    return get_data_fetcher().fetch(stock_code, days=days, providers=providers)


# A-share index code mapping
A_SHARE_INDICES = {
    "000001": "上证指数",
    "000002": "上证A股",
    "000003": "上证B股",
    "000016": "上证50",
    "000300": "沪深300",
    "000905": "中证500",
    "000852": "中证1000",
    "399001": "深证成指",
    "399005": "中小板指",
    "399006": "创业板指",
    "399673": "创业板50",
    "899050": "北证50",
}


def fetch_index_data(index_code: str, days: int = 120) -> pd.DataFrame:
    """Fetch A-share index data.

    Args:
        index_code: Index code (e.g., '000001', '000300', '399001', '399006')
        days: Number of days to fetch

    Returns:
        DataFrame with index data or empty DataFrame on failure
    """
    ak = AKShareProvider()
    return ak.fetch_index(index_code, days=days)
