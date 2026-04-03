"""Industry features extractor."""

from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from ..utils.cache import FeatureCache
from .base import BaseFeatureExtractor


class IndustryFeatures(BaseFeatureExtractor):
    """Extract industry/sector features."""

    def __init__(self, cache: Optional[FeatureCache] = None):
        super().__init__(cache)

    @property
    def feature_type(self) -> str:
        return 'industry'

    def extract(self, stock_code: str, **kwargs) -> pd.DataFrame:
        """Extract industry features for a stock."""
        days = kwargs.get('days', 30)

        try:
            ticker = yf.Ticker(stock_code)
            info = ticker.info
            sector = info.get('sector', None)
            industry = info.get('industry', None)
        except Exception:
            return pd.DataFrame()

        if not sector:
            return pd.DataFrame()

        date_index = pd.date_range(end=pd.Timestamp.today(), periods=days)
        df = pd.DataFrame(index=date_index)
        df['stock_code'] = stock_code
        df['sector'] = sector
        df['industry'] = industry
        df = df.reset_index()
        df.rename(columns={'index': 'date'}, inplace=True)
        # Normalize to date only for consistent merging
        df['date'] = pd.to_datetime(df['date'].dt.date)

        # Sector ETF proxies (approximate mappings)
        sector_etfs = {
            'Technology': 'XLK',
            'Healthcare': 'XLV',
            'Financials': 'XLF',
            'Consumer Discretionary': 'XLY',
            'Consumer Staples': 'XLP',
            'Energy': 'XLE',
            'Industrials': 'XLI',
            'Materials': 'XLB',
            'Real Estate': 'XLRE',
            'Utilities': 'XLU',
            'Communication Services': 'XLC'
        }

        if sector in sector_etfs:
            try:
                sector_data = yf.download(sector_etfs[sector], period=f"{days + 60}d", auto_adjust=False, progress=False)
                if not sector_data.empty:
                    # Flatten multi-level columns if present
                    if isinstance(sector_data.columns, pd.MultiIndex):
                        sector_data.columns = [col[0] for col in sector_data.columns]

                    def get_col(col):
                        return col.iloc[:, 0] if isinstance(col, pd.DataFrame) else col

                    # Create df_sector with explicit date column (not from index)
                    df_sector = pd.DataFrame()
                    df_sector['date'] = pd.to_datetime(sector_data.index)
                    df_sector['sector_close'] = get_col(sector_data['Close']).values
                    df_sector['sector_returns'] = df_sector['sector_close'].pct_change()
                    df_sector['sector_volume'] = get_col(sector_data['Volume']).values
                    df_sector['sector_ma20'] = df_sector['sector_close'].rolling(window=20).mean()
                    df_sector['sector_ma_ratio'] = df_sector['sector_close'] / df_sector['sector_ma20']

                    # Sector RSI
                    delta = df_sector['sector_close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss.replace(0, np.nan)
                    df_sector['sector_rsi'] = 100 - (100 / (1 + rs))

                    df_sector = df_sector.dropna()
                    # Normalize to date only for consistent merging
                    df_sector['date'] = pd.to_datetime(df_sector['date'].dt.date)

                    # Merge on date
                    df = pd.merge(df, df_sector, on=['date'], how='left')
                    df = df.dropna()
                else:
                    df['sector_close'] = np.nan
                    df['sector_returns'] = np.nan
                    df['sector_volume'] = np.nan
                    df['sector_ma20'] = np.nan
                    df['sector_ma_ratio'] = np.nan
                    df['sector_rsi'] = np.nan
            except Exception:
                df['sector_close'] = np.nan
                df['sector_returns'] = np.nan
                df['sector_volume'] = np.nan
                df['sector_ma20'] = np.nan
                df['sector_ma_ratio'] = np.nan
                df['sector_rsi'] = np.nan
        else:
            df['sector_close'] = np.nan
            df['sector_returns'] = np.nan
            df['sector_volume'] = np.nan
            df['sector_ma20'] = np.nan
            df['sector_ma_ratio'] = np.nan
            df['sector_rsi'] = np.nan

        return df
