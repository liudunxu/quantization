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

    def get_or_extract(self, stock_code: str, force_refresh: bool = False, **kwargs) -> pd.DataFrame:
        """Get features from cache or extract if not available."""
        if self.cache and not force_refresh:
            cached = self.cache.get(stock_code, self.feature_type)
            if cached is not None:
                return cached

        df = self.extract(stock_code, **kwargs)

        if self.cache is not None and not df.empty:
            self.cache.set(stock_code, self.feature_type, df)

        return df
