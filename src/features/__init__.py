"""Features package."""

from .base import BaseFeatureExtractor
from .technical import TechnicalFeatures
from .fundamental import FundamentalFeatures
from .market import MarketFeatures
from .industry import IndustryFeatures
from .sentiment import SentimentFeatures
from .southbound_flow import SouthboundFlowFeatures
from .company_events import CompanyEventsFeatures
from .alpha_features import (
    AlphaFeatures,
    calculate_ic,
    select_features_by_ic,
    normalize_features,
)
from .combinator import FeatureCombinator, get_feature_combinator

__all__ = [
    "BaseFeatureExtractor",
    "TechnicalFeatures",
    "FundamentalFeatures",
    "MarketFeatures",
    "IndustryFeatures",
    "SentimentFeatures",
    "SouthboundFlowFeatures",
    "CompanyEventsFeatures",
    "AlphaFeatures",
    "calculate_ic",
    "select_features_by_ic",
    "normalize_features",
    "FeatureCombinator",
    "get_feature_combinator",
]
