"""Stock data source utilities."""

import re
from typing import Tuple
from dataclasses import dataclass


@dataclass
class StockInfo:
    """Stock information container."""
    code: str
    market: str  # 'a_share', 'hk_share', 'us_share'
    exchange: str  # 'SZ', 'SH', 'HK', 'NASDAQ', 'NYSE'
    symbol: str  # Original symbol without exchange


class StockInfoResolver:
    """Resolve stock code to market and exchange."""

    @staticmethod
    def resolve(stock_code: str) -> StockInfo:
        """
        Resolve stock code to StockInfo.

        Examples:
        - 000001.SZ -> A-share, SZ
        - 600000.SH -> A-share, SH
        - 0700.HK -> HK-share, HK
        - AAPL -> US-share, NASDAQ
        """
        stock_code = stock_code.upper().strip()

        # A-share pattern: 6-digit.SZ or 6-digit.SH
        a_share_match = re.match(r'^(\d{6})\.(SZ|SH)$', stock_code)
        if a_share_match:
            return StockInfo(
                code=stock_code,
                market='a_share',
                exchange=a_share_match.group(2),
                symbol=a_share_match.group(1)
            )

        # HK-share pattern: 4-5-digit.HK
        hk_match = re.match(r'^(\d{4,5})\.HK$', stock_code)
        if hk_match:
            return StockInfo(
                code=stock_code,
                market='hk_share',
                exchange='HK',
                symbol=hk_match.group(1)
            )

        # US stock (no extension) - default to NASDAQ
        if '.' not in stock_code:
            # Check if it's a valid US stock symbol format
            if re.match(r'^[A-Z]{1,5}$', stock_code):
                return StockInfo(
                    code=stock_code,
                    market='us_share',
                    exchange='NASDAQ',  # Default, could be NYSE
                    symbol=stock_code
                )

        raise ValueError(f"Unknown stock code format: {stock_code}")

    @staticmethod
    def get_index_code(stock_info: StockInfo) -> str:
        """Get corresponding market index code."""
        if stock_info.market == 'a_share':
            return '000001.SH'  # Shanghai Composite
        elif stock_info.market == 'hk_share':
            return '^HSI'  # Hang Seng Index (yfinance format)
        elif stock_info.market == 'us_share':
            return '^GSPC'  # S&P 500
        return '^GSPC'


def format_stock_code(code: str, exchange: str) -> str:
    """Format stock code with exchange suffix."""
    if exchange in ['SZ', 'SH']:
        return f"{code}.{exchange}"
    elif exchange == 'HK':
        return f"{code}.HK"
    return code
