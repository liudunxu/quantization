"""Market-wide features extractor."""

from typing import Optional
import pandas as pd
import numpy as np
from ..utils.cache import FeatureCache
from ..utils.stock_info import StockInfoResolver
from ..utils.config import get_config
from .base import BaseFeatureExtractor
from ..data_providers import fetch_stock_data


class MarketFeatures(BaseFeatureExtractor):
    """Extract market-wide features."""

    def __init__(self, cache: Optional[FeatureCache] = None):
        super().__init__(cache)

    @property
    def feature_type(self) -> str:
        return 'market'

    def extract(self, stock_code: str, **kwargs) -> pd.DataFrame:
        """Extract market features for a stock."""
        days = kwargs.get('days', 30)

        try:
            stock_info = StockInfoResolver.resolve(stock_code)
            index_code = StockInfoResolver.get_index_code(stock_info)
        except ValueError:
            index_code = '^GSPC'

        # Use multi-provider data fetcher for index
        index_data = fetch_stock_data(index_code, days=days + 60)
        if index_data.empty:
            return pd.DataFrame()

        df = pd.DataFrame(index=range(len(index_data)))
        df['stock_code'] = stock_code
        df['index_code'] = index_code

        # Index prices
        df['index_close'] = index_data['close']
        df['index_open'] = index_data['open']
        df['index_high'] = index_data['high']
        df['index_low'] = index_data['low']
        df['index_volume'] = index_data['volume']

        # Index returns
        df['index_returns'] = df['index_close'].pct_change()
        df['index_log_returns'] = np.log(df['index_close'] / df['index_close'].shift(1))

        # Index moving averages
        for period in [5, 10, 20]:
            df[f'index_ma_{period}'] = df['index_close'].rolling(window=period).mean()
            df[f'index_ma_{period}_ratio'] = df['index_close'] / df[f'index_ma_{period}']

        # Index RSI
        delta = df['index_close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['index_rsi'] = 100 - (100 / (1 + rs))

        # Index volatility
        df['index_volatility_5'] = df['index_returns'].rolling(window=5).std()
        df['index_volatility_20'] = df['index_returns'].rolling(window=20).std()

        # Index momentum
        for period in [5, 10, 20]:
            df[f'index_momentum_{period}'] = df['index_close'] / df['index_close'].shift(period) - 1

        # Index volume
        df['index_volume_ma20'] = df['index_volume'].rolling(window=20).mean()
        df['index_volume_ratio'] = df['index_volume'] / df['index_volume_ma20']

        # Index 52-week (only if enough data)
        if len(df) >= 252:
            df['index_high_52w'] = df['index_close'].rolling(window=252).max()
            df['index_low_52w'] = df['index_close'].rolling(window=252).min()
            df['index_position_52w'] = (df['index_close'] - df['index_low_52w']) / (df['index_high_52w'] - df['index_low_52w'] + 1e-10)
        else:
            # Use available data for position calculation
            df['index_high_52w'] = df['index_close'].rolling(window=min(60, len(df))).max()
            df['index_low_52w'] = df['index_close'].rolling(window=min(60, len(df))).min()
            df['index_position_52w'] = (df['index_close'] - df['index_low_52w']) / (df['index_high_52w'] - df['index_low_52w'] + 1e-10)

        # Fill NaN values with forward fill then backward fill for remaining
        df = df.ffill().bfill()
        df = df.reset_index()
        # Ensure date column name is lowercase
        if 'Date' in df.columns:
            df.rename(columns={'Date': 'date'}, inplace=True)
        elif 'index' in df.columns:
            df.rename(columns={'index': 'date'}, inplace=True)

        return df
