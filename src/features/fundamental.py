"""Fundamental features extractor."""

from typing import Optional
import pandas as pd
import numpy as np
import yfinance as yf
from ..utils.cache import FeatureCache
from ..utils.stock_info import StockInfoResolver, StockInfo
from .base import BaseFeatureExtractor


class FundamentalFeatures(BaseFeatureExtractor):
    """Extract fundamental features for a stock."""

    def __init__(self, cache: Optional[FeatureCache] = None):
        super().__init__(cache)

    @property
    def feature_type(self) -> str:
        return 'fundamental'

    def extract(self, stock_code: str, **kwargs) -> pd.DataFrame:
        """Extract fundamental features for a stock."""
        try:
            ticker = yf.Ticker(stock_code)
            info = ticker.info

            if not info or len(info) < 5:
                return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

        df = pd.DataFrame(index=[0])
        df['stock_code'] = stock_code
        df['date'] = pd.Timestamp.today()

        # Valuation metrics
        df['pe_ratio'] = info.get('trailingPE', np.nan)
        df['forward_pe'] = info.get('forwardPE', np.nan)
        df['peg_ratio'] = info.get('pegRatio', np.nan)
        df['pb_ratio'] = info.get('priceToBook', np.nan)
        df['ps_ratio'] = info.get('priceToSalesTrailing12Months', np.nan)

        # Profitability
        df['roe'] = info.get('returnOnEquity', np.nan)
        df['roa'] = info.get('returnOnAssets', np.nan)
        df['gross_margin'] = info.get('grossMargins', np.nan)
        df['operating_margin'] = info.get('operatingMargins', np.nan)
        df['net_margin'] = info.get('profitMargins', np.nan)

        # Growth
        df['revenue_growth'] = info.get('revenueGrowth', np.nan)
        df['earnings_growth'] = info.get('earningsGrowth', np.nan)
        df['earnings_quarterly_growth'] = info.get('earningsQuarterlyGrowth', np.nan)

        # Financial health
        df['debt_to_equity'] = info.get('debtToEquity', np.nan)
        df['current_ratio'] = info.get('currentRatio', np.nan)
        df['quick_ratio'] = info.get('quickRatio', np.nan)

        # Dividends
        df['dividend_yield'] = info.get('dividendYield', np.nan)
        df['dividend_rate'] = info.get('dividendRate', np.nan)
        df['payout_ratio'] = info.get('payoutRatio', np.nan)

        # Share structure
        df['shares_outstanding'] = info.get('sharesOutstanding', np.nan)
        df['market_cap'] = info.get('marketCap', np.nan)
        df['enterprise_value'] = info.get('enterpriseValue', np.nan)

        # Analyst recommendations
        df['recommendation_key'] = info.get('recommendationKey', np.nan)
        df['number_of_analyst_recommendations'] = info.get('numberOfAnalystRecommendations', np.nan)

        # Price targets
        df['target_mean_price'] = info.get('targetMeanPrice', np.nan)
        df['target_high_price'] = info.get('targetHighPrice', np.nan)
        df['target_low_price'] = info.get('targetLowPrice', np.nan)
        df['current_price'] = info.get('currentPrice', np.nan)

        # Valuation vs target
        if df['current_price'].notna().any() and df['target_mean_price'].notna().any():
            df['price_to_target'] = df['current_price'] / df['target_mean_price']

        # 52-week
        df['week_52_high'] = info.get('fiftyTwoWeekHigh', np.nan)
        df['week_52_low'] = info.get('fiftyTwoWeekLow', np.nan)
        df['week_52_high_ratio'] = df['current_price'] / df['week_52_high']
        df['week_52_low_ratio'] = df['current_price'] / df['week_52_low']

        df = df.dropna(subset=['pe_ratio', 'pb_ratio'], how='all')

        return df
