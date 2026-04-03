"""Base feature extractor class."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import pandas as pd
from ..utils.cache import FeatureCache


class BaseFeatureExtractor(ABC):
    """Base class for feature extractors."""

    def __init__(self, cache: Optional[FeatureCache] = None):
        self.cache = cache

    @property
    @abstractmethod
    def feature_type(self) -> str:
        """Return feature type name."""
        pass

    @abstractmethod
    def extract(self, stock_code: str, **kwargs) -> pd.DataFrame:
        """Extract features for a stock."""
        pass

    def get_or_extract(
        self, stock_code: str, force_refresh: bool = False, **kwargs
    ) -> pd.DataFrame:
        """Get features from cache or extract if not available.

        Supports incremental updates: only fetches missing data since last cache date.
        """
        days = kwargs.get("days")

        if self.cache and not force_refresh:
            cached = self.cache.get(stock_code, self.feature_type)
            if cached is not None and not cached.empty:
                # Check if cached data has enough rows
                if days and len(cached) >= days:
                    cached = (
                        cached.sort_values("date").tail(days).reset_index(drop=True)
                    )
                    return cached
                elif not days:
                    return cached
                # else: need more data, fall through to extract

        df = self.extract(stock_code, **kwargs)

        if df.empty:
            return df

        # Merge with existing cache (incremental update)
        if self.cache is not None:
            df = self.cache.merge_and_update(
                stock_code, self.feature_type, df, kwargs.get("params")
            )

        # Filter by days if specified
        if days and "date" in df.columns:
            df = df.sort_values("date").tail(days).reset_index(drop=True)

        return df

        # Merge with existing cache (incremental update)
        if self.cache is not None:
            df = self.cache.merge_and_update(
                stock_code, self.feature_type, df, kwargs.get("params")
            )

        # Filter by days if specified
        if days and "date" in df.columns:
            df = df.sort_values("date").tail(days).reset_index(drop=True)

        return df
