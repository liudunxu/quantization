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
        """Get features from cache or extract if not available."""
        days = kwargs.get("days")

        if self.cache and not force_refresh:
            cached = self.cache.get(stock_code, self.feature_type)
            if cached is not None:
                # Filter by days if specified
                if days and "date" in cached.columns:
                    cached = (
                        cached.sort_values("date").tail(days).reset_index(drop=True)
                    )
                return cached

        df = self.extract(stock_code, **kwargs)

        if self.cache is not None and not df.empty:
            self.cache.set(stock_code, self.feature_type, df)

        # Filter by days if specified
        if days and "date" in df.columns:
            df = df.sort_values("date").tail(days).reset_index(drop=True)

        return df
