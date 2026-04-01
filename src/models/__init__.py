"""Models package."""

from .trainer import StockTradingModel, get_model
from .base import BaseModel

# Optional imports (requires pip install lightgbm xgboost)
try:
    from .lgbm_model import LightGBMModel

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from .xgboost_model import XGBoostModel

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from .multi_model import MultiModelEnsemble

__all__ = [
    "StockTradingModel",
    "get_model",
    "BaseModel",
    "MultiModelEnsemble",
]

if HAS_LIGHTGBM:
    __all__.append("LightGBMModel")
if HAS_XGBOOST:
    __all__.append("XGBoostModel")
