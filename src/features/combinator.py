"""Feature combinator that merges all features."""

from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
from ..utils.cache import FeatureCache
from .technical import TechnicalFeatures
from .fundamental import FundamentalFeatures
from .market import MarketFeatures
from .industry import IndustryFeatures
from .money_flow import MoneyFlowFeatures
from .sentiment import SentimentFeatures
from .southbound_flow import SouthboundFlowFeatures
from .company_events import CompanyEventsFeatures


class FeatureCombinator:
    """Combine all features into a single DataFrame."""

    def __init__(self, cache: Optional[FeatureCache] = None):
        self.cache = cache
        self.extractors = {
            "technical": TechnicalFeatures(cache),
            "fundamental": FundamentalFeatures(cache),
            "market": MarketFeatures(cache),
            "industry": IndustryFeatures(cache),
            "money_flow": MoneyFlowFeatures(cache),
            "sentiment": SentimentFeatures(cache),
            "southbound_flow": SouthboundFlowFeatures(cache),
            "company_events": CompanyEventsFeatures(cache),
        }

    def get_combined_features(
        self, stock_code: str, days: int = 120, force_refresh: bool = False
    ) -> pd.DataFrame:
        """Get combined features for a stock."""
        features = {}

        # Technical features (time series)
        tech_df = self.extractors["technical"].get_or_extract(
            stock_code, force_refresh=force_refresh, days=days
        )
        if not tech_df.empty:
            features["technical"] = tech_df

        # Market features (time series)
        market_df = self.extractors["market"].get_or_extract(
            stock_code, force_refresh=force_refresh, days=days
        )
        if not market_df.empty:
            features["market"] = market_df

        # Fundamental features (point in time - latest)
        fund_df = self.extractors["fundamental"].get_or_extract(
            stock_code, force_refresh=force_refresh
        )
        if not fund_df.empty:
            features["fundamental"] = fund_df

        # Industry features (time series)
        industry_df = self.extractors["industry"].get_or_extract(
            stock_code, force_refresh=force_refresh, days=days
        )
        if not industry_df.empty:
            features["industry"] = industry_df

        # Sentiment features (time series)
        sentiment_df = self.extractors["sentiment"].get_or_extract(
            stock_code, force_refresh=force_refresh, days=days
        )
        if not sentiment_df.empty:
            features["sentiment"] = sentiment_df

        # Money flow features (time series) - A-share only
        if not stock_code.endswith(".HK"):
            mf_df = self.extractors["money_flow"].get_or_extract(
                stock_code, force_refresh=force_refresh, days=days
            )
            if not mf_df.empty:
                features["money_flow"] = mf_df

        # Southbound flow features (time series) - HK only
        if stock_code.endswith(".HK"):
            sb_df = self.extractors["southbound_flow"].get_or_extract(
                stock_code, force_refresh=force_refresh, days=days
            )
            if not sb_df.empty:
                features["southbound_flow"] = sb_df

        # Company events features (time series) - All markets
        events_df = self.extractors["company_events"].get_or_extract(
            stock_code, force_refresh=force_refresh, days=days
        )
        if not events_df.empty:
            features["company_events"] = events_df

        if not features:
            return pd.DataFrame()

        # Start with technical features
        combined = features["technical"].copy()

        # Merge market features
        if "market" in features and not features["market"].empty:
            market_cols = [
                c for c in features["market"].columns if c not in ["stock_code", "date"]
            ]
            combined = combined.merge(
                features["market"][["date"] + market_cols],
                on="date",
                how="left",
                suffixes=("", "_market"),
            )

        # Merge industry features
        if "industry" in features and not features["industry"].empty:
            industry_cols = [
                c
                for c in features["industry"].columns
                if c not in ["stock_code", "date", "sector", "industry"]
            ]
            combined = combined.merge(
                features["industry"][["date"] + industry_cols],
                on="date",
                how="left",
                suffixes=("", "_industry"),
            )

        # Merge money flow features
        if "money_flow" in features and not features["money_flow"].empty:
            mf_cols = [
                c
                for c in features["money_flow"].columns
                if c not in ["stock_code", "date"]
            ]
            combined = combined.merge(
                features["money_flow"][["date"] + mf_cols],
                on="date",
                how="left",
                suffixes=("", "_mf"),
            )

        # Merge sentiment features
        if "sentiment" in features and not features["sentiment"].empty:
            sentiment_cols = [
                c
                for c in features["sentiment"].columns
                if c not in ["stock_code", "date"]
            ]
            combined = combined.merge(
                features["sentiment"][["date"] + sentiment_cols],
                on="date",
                how="left",
                suffixes=("", "_sentiment"),
            )

        # Merge fundamental features (broadcast to all rows)
        if "fundamental" in features and not features["fundamental"].empty:
            fund_cols = [
                c
                for c in features["fundamental"].columns
                if c not in ["stock_code", "date"]
            ]
            for col in fund_cols:
                combined[col] = features["fundamental"][col].iloc[0]

        # Add sector and industry if available
        if "industry" in features and not features["industry"].empty:
            combined["sector"] = features["industry"]["sector"].iloc[0]
            combined["industry"] = features["industry"]["industry"].iloc[0]

        # ========== Relative performance features (相对表现) ==========
        # Stock vs market (Alpha)
        if "returns" in combined.columns and "index_returns" in combined.columns:
            combined["alpha"] = combined["returns"] - combined["index_returns"]
            combined["alpha_5d"] = combined["momentum_5"] - combined.get(
                "index_momentum_5", 0
            )
            combined["alpha_10d"] = combined["momentum_10"] - combined.get(
                "index_momentum_10", 0
            )

        # Stock vs sector
        if "returns" in combined.columns and "sector_returns" in combined.columns:
            combined["sector_relative"] = (
                combined["returns"] - combined["sector_returns"]
            )
            combined["sector_relative_5d"] = combined["momentum_5"] - combined.get(
                "sector_momentum_5", 0
            )

        # Market correlation (rolling)
        if "returns" in combined.columns and "index_returns" in combined.columns:
            combined["market_corr_5"] = (
                combined["returns"].rolling(window=5).corr(combined["index_returns"])
            )
            combined["market_corr_10"] = (
                combined["returns"].rolling(window=10).corr(combined["index_returns"])
            )

        # Beta (stock volatility vs market)
        if "returns" in combined.columns and "index_returns" in combined.columns:
            combined["beta_5"] = combined["returns"].rolling(window=5).cov(
                combined["index_returns"]
            ) / (combined["index_returns"].rolling(window=5).std() + 1e-8)
            combined["beta_20"] = combined["returns"].rolling(window=20).cov(
                combined["index_returns"]
            ) / (combined["index_returns"].rolling(window=20).std() + 1e-8)

        # Volume vs market volume
        if (
            "volume_change" in combined.columns
            and "index_volume_ratio" in combined.columns
        ):
            combined["volume_vs_market"] = (
                combined["volume_change"] - combined["index_volume_ratio"]
            )

        # Clean up column names
        combined.columns = [
            c.replace("_market", "").replace("_industry", "") for c in combined.columns
        ]

        return combined

    def get_latest_features(
        self, stock_code: str, force_refresh: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Get latest features for a stock as a dictionary."""
        df = self.get_combined_features(
            stock_code, days=120, force_refresh=force_refresh
        )
        if df.empty:
            return None

        latest = df.iloc[-1].to_dict()

        # Get latest fundamental separately
        fund_df = self.extractors["fundamental"].get_or_extract(
            stock_code, force_refresh=force_refresh
        )
        if not fund_df.empty:
            for col in fund_df.columns:
                if col not in ["stock_code", "date"]:
                    latest[col] = fund_df[col].iloc[0]

        return latest


# Global instance
_combinator: Optional[FeatureCombinator] = None


def get_feature_combinator(cache: Optional[FeatureCache] = None) -> FeatureCombinator:
    """Get global feature combinator instance."""
    global _combinator
    if _combinator is None:
        _combinator = FeatureCombinator(cache)
    return _combinator
