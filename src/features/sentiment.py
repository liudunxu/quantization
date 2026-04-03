"""Sentiment features extractor for stock news and social media analysis."""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from ..data_providers.sentiment_provider import SentimentProvider
from ..utils.cache import FeatureCache
from .base import BaseFeatureExtractor

logger = logging.getLogger(__name__)


class SentimentFeatures(BaseFeatureExtractor):
    """Extract sentiment features from news and social media data."""

    def __init__(self, cache: Optional[FeatureCache] = None):
        super().__init__(cache)
        self.provider = SentimentProvider()

    @property
    def feature_type(self) -> str:
        return "sentiment"

    def extract(self, stock_code: str, days: int = 120, **kwargs) -> pd.DataFrame:
        """Extract sentiment features for a stock.

        Args:
            stock_code: Stock code
            days: Number of days to look back

        Returns:
            DataFrame with sentiment features
        """
        try:
            # Fetch sentiment data
            sentiment_df = self.provider.fetch(stock_code, days)

            if sentiment_df.empty:
                logger.warning(
                    f"[{self.feature_type}] No sentiment data for {stock_code}"
                )
                return self._create_default_features(days)

            # Create features
            features = self._create_features(sentiment_df)

            # Add stock code
            features["stock_code"] = stock_code

            logger.info(
                f"[{self.feature_type}] Extracted {len(features.columns)} features for {stock_code}"
            )
            return features

        except Exception as e:
            logger.error(
                f"[{self.feature_type}] Failed to extract features for {stock_code}: {e}"
            )
            return self._create_default_features(days)

    def _create_features(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """Create sentiment features from raw sentiment data."""
        features = pd.DataFrame()

        # Ensure date column exists
        if "date" not in sentiment_df.columns:
            sentiment_df["date"] = pd.Timestamp.today()

        # Basic sentiment features
        features["date"] = sentiment_df["date"]
        features["sentiment_score"] = sentiment_df.get("sentiment_score", 0.0)
        features["news_count"] = sentiment_df.get("news_count", 0)

        # Rolling sentiment features
        if "sentiment_score" in sentiment_df.columns:
            features["sentiment_ma3"] = (
                sentiment_df["sentiment_score"].rolling(window=3, min_periods=1).mean()
            )
            features["sentiment_ma7"] = (
                sentiment_df["sentiment_score"].rolling(window=7, min_periods=1).mean()
            )
            features["sentiment_ma14"] = (
                sentiment_df["sentiment_score"].rolling(window=14, min_periods=1).mean()
            )

            # Sentiment volatility
            features["sentiment_std7"] = (
                sentiment_df["sentiment_score"].rolling(window=7, min_periods=1).std()
            )
            features["sentiment_std14"] = (
                sentiment_df["sentiment_score"].rolling(window=14, min_periods=1).std()
            )

            # Sentiment momentum
            features["sentiment_momentum"] = sentiment_df["sentiment_score"].diff()
            features["sentiment_momentum3"] = sentiment_df["sentiment_score"].diff(3)

            # Sentiment acceleration
            features["sentiment_acceleration"] = features["sentiment_momentum"].diff()

            # Extreme sentiment flags
            features["sentiment_extreme_positive"] = (
                sentiment_df["sentiment_score"] > 0.5
            ).astype(int)
            features["sentiment_extreme_negative"] = (
                sentiment_df["sentiment_score"] < -0.5
            ).astype(int)
            features["sentiment_neutral"] = (
                (sentiment_df["sentiment_score"] >= -0.1)
                & (sentiment_df["sentiment_score"] <= 0.1)
            ).astype(int)

            # Sentiment regime
            features["sentiment_regime"] = np.where(
                sentiment_df["sentiment_score"] > 0.2,
                2,  # Bullish
                np.where(
                    sentiment_df["sentiment_score"] < -0.2, 0, 1
                ),  # Bearish, Neutral
            )

            # Sentiment trend (comparing short-term vs long-term)
            features["sentiment_trend"] = (
                features["sentiment_ma3"] - features["sentiment_ma14"]
            )

        # News volume features
        if "news_count" in sentiment_df.columns:
            features["news_count_ma3"] = (
                sentiment_df["news_count"].rolling(window=3, min_periods=1).mean()
            )
            features["news_count_ma7"] = (
                sentiment_df["news_count"].rolling(window=7, min_periods=1).mean()
            )

            # News volume change
            features["news_volume_change"] = sentiment_df["news_count"].pct_change()

            # News volume spike (current > 2x average)
            features["news_volume_spike"] = (
                sentiment_df["news_count"] > 2 * features["news_count_ma7"]
            ).astype(int)

        # Combined sentiment features
        if (
            "sentiment_score" in sentiment_df.columns
            and "news_count" in sentiment_df.columns
        ):
            # Weighted sentiment (weighted by news volume)
            features["weighted_sentiment"] = sentiment_df["sentiment_score"] * np.log1p(
                sentiment_df["news_count"]
            )

            # Sentiment divergence (difference between weighted and simple sentiment)
            features["sentiment_divergence"] = (
                features["weighted_sentiment"] - sentiment_df["sentiment_score"]
            )

        # Fill NaN values
        features = features.fillna(0)

        return features

    def _create_default_features(self, days: int = 120) -> pd.DataFrame:
        """Create default neutral sentiment features when data is unavailable."""
        dates = pd.date_range(end=pd.Timestamp.today(), periods=days, freq="D")

        features = pd.DataFrame(
            {
                "date": dates,
                "sentiment_score": 0.0,
                "news_count": 0,
                "sentiment_ma3": 0.0,
                "sentiment_ma7": 0.0,
                "sentiment_ma14": 0.0,
                "sentiment_std7": 0.0,
                "sentiment_std14": 0.0,
                "sentiment_momentum": 0.0,
                "sentiment_momentum3": 0.0,
                "sentiment_acceleration": 0.0,
                "sentiment_extreme_positive": 0,
                "sentiment_extreme_negative": 0,
                "sentiment_neutral": 1,
                "sentiment_regime": 1,  # Neutral
                "sentiment_trend": 0.0,
                "news_count_ma3": 0.0,
                "news_count_ma7": 0.0,
                "news_volume_change": 0.0,
                "news_volume_spike": 0,
                "weighted_sentiment": 0.0,
                "sentiment_divergence": 0.0,
            }
        )

        return features

    def get_latest_sentiment(self, stock_code: str) -> Optional[dict]:
        """Get latest sentiment summary for a stock."""
        try:
            df = self.get_or_extract(stock_code, days=30)
            if df.empty:
                return None

            latest = df.iloc[-1].to_dict()

            # Add interpretation
            sentiment_score = latest.get("sentiment_score", 0)
            if sentiment_score > 0.3:
                latest["sentiment_label"] = "Positive"
                latest["sentiment_emoji"] = "🟢"
            elif sentiment_score < -0.3:
                latest["sentiment_label"] = "Negative"
                latest["sentiment_emoji"] = "🔴"
            else:
                latest["sentiment_label"] = "Neutral"
                latest["sentiment_emoji"] = "⚪"

            # Add news summary
            news_count = latest.get("news_count", 0)
            if news_count > 10:
                latest["news_activity"] = "High"
            elif news_count > 3:
                latest["news_activity"] = "Moderate"
            else:
                latest["news_activity"] = "Low"

            return latest

        except Exception as e:
            logger.error(
                f"[{self.feature_type}] Failed to get latest sentiment for {stock_code}: {e}"
            )
            return None


# Convenience function
def get_sentiment_features(
    stock_code: str, days: int = 120, cache: Optional[FeatureCache] = None
) -> pd.DataFrame:
    """Get sentiment features for a stock."""
    extractor = SentimentFeatures(cache)
    return extractor.get_or_extract(stock_code, days=days)
