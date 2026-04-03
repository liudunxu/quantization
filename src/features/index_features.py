"""Index feature extractor for A-share market indices.

Simplified feature extraction for indices (no fundamental, money flow, or
southbound flow features since those don't apply to indices).
"""

import logging

import pandas as pd

from ..data_providers.fetch_stock_data import fetch_index_data
from ..features.alpha_features import AlphaFeatures
from ..features.company_events import CompanyEventsFeatures
from ..features.technical import TechnicalFeatures

logger = logging.getLogger(__name__)

# A-share index display names
INDEX_NAMES = {
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


def get_index_name(code: str) -> str:
    """Get display name for an index code."""
    return INDEX_NAMES.get(code, code)


def extract_index_features(index_code: str, days: int = 120) -> pd.DataFrame:
    """Extract features for an A-share index.

    Uses a simplified feature set suitable for indices:
    - Technical indicators (MA, RSI, MACD, Bollinger, ATR, etc.)
    - Alpha features (Qlib-style)
    - Company events (earnings seasons, holidays, etc.)
    - Market features (index-level metrics)

    Excludes features that don't apply to indices:
    - Fundamental (P/E, P/B, etc. - not meaningful for indices)
    - Money flow (主力资金 - individual stock concept)
    - Southbound flow (港股通 - HK specific)
    - Industry features

    Args:
        index_code: Index code (e.g., '000001', '000300', '399001', '399006')
        days: Number of days of data to fetch

    Returns:
        DataFrame with index features
    """
    logger.info(
        f"[IndexFeatures] Fetching data for index {index_code} ({get_index_name(index_code)})"
    )

    # Fetch index price data
    data = fetch_index_data(index_code, days=days)
    if data.empty:
        logger.error(f"[IndexFeatures] Failed to fetch data for index {index_code}")
        return pd.DataFrame()

    logger.info(f"[IndexFeatures] Fetched {len(data)} rows for {index_code}")

    # Use technical features extractor (works on any OHLCV data)
    tech = TechnicalFeatures()
    # Monkey-patch the fetch method to use index data
    df = tech.extract(index_code, days=days, _preloaded_data=data)

    # Add alpha features
    if not df.empty and len(df) >= 60:
        alpha = AlphaFeatures()
        alpha_df = alpha.extract(index_code, df=df, days=days)
        if not alpha_df.empty:
            # Merge alpha features
            alpha_cols = [
                c
                for c in alpha_df.columns
                if c not in df.columns and c not in ("date", "stock_code")
            ]
            if alpha_cols:
                df = df.merge(alpha_df[["date"] + alpha_cols], on="date", how="left")

    # Add company events (earnings seasons, holidays, etc.)
    events = CompanyEventsFeatures()
    events_df = events.extract(index_code, days=days, df=df)
    if not events_df.empty:
        event_cols = [
            c
            for c in events_df.columns
            if c not in df.columns and c not in ("date", "stock_code")
        ]
        if event_cols:
            df = df.merge(events_df[["date"] + event_cols], on="date", how="left")

    # Fill NaN values
    df = df.ffill().bfill().fillna(0)
    df = df.replace([float("inf"), float("-inf")], 0)

    logger.info(
        f"[IndexFeatures] Extracted {len(df.columns)} features for {index_code}"
    )
    return df
