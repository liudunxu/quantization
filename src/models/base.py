"""Base model class for stock trading decisions."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd


class BaseModel(ABC):
    """Base class for all stock trading models."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.feature_names: Optional[List[str]] = None
        self.selected_features: Optional[List[str]] = None
        self.is_trained = False

    @abstractmethod
    def train(
        self,
        df: pd.DataFrame,
        forward_days: int = 5,
        threshold: float = 0.01,
        eval_df: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> tuple:
        """Predict trading action."""
        pass

    @abstractmethod
    def predict_proba(self, df: pd.DataFrame) -> dict:
        """Predict trading action probabilities."""
        pass

    @abstractmethod
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model name."""
        pass
