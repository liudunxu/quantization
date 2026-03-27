"""Stock data fetcher with multi-provider fallback."""

import logging
from typing import Optional, List
import pandas as pd
from .base import BaseDataProvider
from .yfinance_provider import YFinanceProvider
from .akshare_provider import AKShareProvider
from .tushare_provider import TushareProvider

logger = logging.getLogger(__name__)


class StockDataFetcher:
    """Multi-provider stock data fetcher with automatic fallback.

    Tries providers in order:
    1. yfinance (global stocks - US, HK)
    2. akshare (Chinese A-shares fallback)
    3. tushare (Chinese A-shares/HK - requires token)
    """

    def __init__(self, providers: Optional[List[BaseDataProvider]] = None):
        """Initialize with providers list.

        Args:
            providers: List of providers to try in order. Defaults to [YFinance, AKShare, Tushare]
        """
        if providers is None:
            self.providers: List[BaseDataProvider] = [
                YFinanceProvider(),
                AKShareProvider(),
                TushareProvider()
            ]
        else:
            self.providers = providers

    def _is_chinese_stock(self, stock_code: str) -> bool:
        """Check if stock code is for Chinese market (A-share or HK)."""
        return any(ext in stock_code.upper() for ext in ['.SH', '.SZ', '.SS', '.HK'])

    def fetch(
        self,
        stock_code: str,
        days: int = 120,
        providers: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Fetch stock data with automatic provider fallback.

        Args:
            stock_code: Stock code (e.g., 000001.SZ, 0700.HK, AAPL)
            days: Number of days to fetch
            providers: Optional list of provider names to use (e.g., ['yfinance', 'akshare'])
                     If None, tries all providers in order.

        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        tried_providers = []

        # For Chinese stocks, try akshare first if yfinance fails
        if self._is_chinese_stock(stock_code):
            provider_order = []
            if providers:
                provider_order = [p for p in providers if p in ['yfinance', 'akshare', 'tushare']]
            else:
                # Default: try yfinance first for HK, akshare for A-shares, tushare last
                if stock_code.endswith('.HK'):
                    provider_order = ['yfinance', 'akshare', 'tushare']
                else:
                    # A-share: akshare first, then tushare, then yfinance
                    provider_order = ['akshare', 'tushare', 'yfinance']

            for provider_name in provider_order:
                provider = self._get_provider(provider_name)
                if provider is None:
                    continue

                tried_providers.append(provider.name)
                logger.info(f"[StockDataFetcher] Trying {provider.name} for {stock_code}")

                df = provider.fetch(stock_code, days=days)
                if not df.empty:
                    logger.info(f"[StockDataFetcher] Success with {provider.name}")
                    return df

                logger.warning(f"[StockDataFetcher] {provider.name} returned empty data for {stock_code}")
        else:
            # Non-Chinese stocks: yfinance only
            if providers:
                for provider_name in providers:
                    provider = self._get_provider(provider_name)
                    if provider:
                        tried_providers.append(provider.name)
                        df = provider.fetch(stock_code, days=days)
                        if not df.empty:
                            return df
            else:
                # Try all providers
                for provider in self.providers:
                    tried_providers.append(provider.name)
                    df = provider.fetch(stock_code, days=days)
                    if not df.empty:
                        return df

        logger.error(
            f"[StockDataFetcher] All providers failed for {stock_code}. "
            f"Tried: {', '.join(tried_providers)}"
        )
        return pd.DataFrame()

    def _get_provider(self, name: str) -> Optional[BaseDataProvider]:
        """Get provider by name."""
        name_map = {
            'yfinance': YFinanceProvider,
            'akshare': AKShareProvider,
            'tushare': TushareProvider
        }
        provider_cls = name_map.get(name.lower())
        if provider_cls:
            return provider_cls()
        return None


# Global instance
_fetcher: Optional[StockDataFetcher] = None


def get_data_fetcher() -> StockDataFetcher:
    """Get global data fetcher instance."""
    global _fetcher
    if _fetcher is None:
        _fetcher = StockDataFetcher()
    return _fetcher


def fetch_stock_data(
    stock_code: str,
    days: int = 120,
    providers: Optional[List[str]] = None
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
