"""US market sentiment features from VIX, Fear & Greed Index, and Put/Call Ratio."""

import logging
from typing import Optional

import pandas as pd

from ..utils.cache import FeatureCache
from .base import BaseFeatureExtractor

logger = logging.getLogger(__name__)


class USMarketSentimentFeatures(BaseFeatureExtractor):
    """Extract US market sentiment features from free data sources.

    Sources:
    - CBOE VIX (Volatility Index / Fear Gauge) via yfinance
    - CNN Fear & Greed Index via public API
    - CBOE Put/Call Ratio via yfinance
    """

    def __init__(self, cache: Optional[FeatureCache] = None):
        super().__init__(cache)

    @property
    def feature_type(self) -> str:
        return "us_market_sentiment"

    def extract(self, stock_code: str, days: int = 120, **kwargs) -> pd.DataFrame:
        """Extract US market sentiment features.

        Only produces meaningful features for US stocks (e.g., AAPL, MSFT).
        For other markets, returns default neutral features.
        """
        if not self._is_us_stock(stock_code):
            return self._create_default_features(days)

        try:
            sentiment_df = self._fetch_sentiment_data(days)

            if sentiment_df.empty:
                logger.warning(
                    f"[{self.feature_type}] No sentiment data for {stock_code}"
                )
                return self._create_default_features(days)

            features = self._create_features(sentiment_df)
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

    def _is_us_stock(self, stock_code: str) -> bool:
        """Check if stock is a US stock."""
        # US stocks don't have exchange suffix like .SZ/.SH/.HK
        return not stock_code.endswith((".SZ", ".SH", ".HK"))

    def _fetch_sentiment_data(self, days: int) -> pd.DataFrame:
        """Fetch sentiment data from multiple sources."""
        result = pd.DataFrame()

        # 1. VIX data
        vix_df = self._fetch_vix(days)

        # 2. CNN Fear & Greed Index
        fng_df = self._fetch_fear_greed(days)

        # 3. Put/Call Ratio
        pcr_df = self._fetch_put_call_ratio(days)

        # Merge all sources on date
        if not vix_df.empty:
            result = vix_df

        if not fng_df.empty:
            if result.empty:
                result = fng_df
            else:
                result = result.merge(fng_df, on="date", how="outer")

        if not pcr_df.empty:
            if result.empty:
                result = pcr_df
            else:
                result = result.merge(pcr_df, on="date", how="outer")

        if not result.empty:
            result = result.sort_values("date").reset_index(drop=True)

        return result

    def _fetch_vix(self, days: int) -> pd.DataFrame:
        """Fetch VIX data from yfinance."""
        try:
            import yfinance as yf

            period_map = {
                range(0, 8): "7d",
                range(8, 32): "1mo",
                range(32, 92): "3mo",
                range(92, 200): "6mo",
                range(200, 400): "1y",
                range(400, 750): "2y",
            }

            period = "1y"
            for r, p in period_map.items():
                if days in r:
                    period = p
                    break

            vix = yf.download("^VIX", period=period, progress=False)

            if vix.empty or len(vix) < 2:
                return pd.DataFrame()

            vix = vix.copy()
            vix.index = pd.to_datetime(vix.index)
            vix = vix[~vix.index.duplicated(keep="first")]
            vix = vix.tail(days)

            df = pd.DataFrame()
            df["date"] = vix.index
            df["vix_close"] = vix["Close"].values
            df["vix_open"] = vix["Open"].values
            df["vix_high"] = vix["High"].values
            df["vix_low"] = vix["Low"].values

            return df

        except ImportError:
            logger.warning("yfinance not available for VIX data")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to fetch VIX data: {e}")
            return pd.DataFrame()

    def _fetch_fear_greed(self, days: int) -> pd.DataFrame:
        """Fetch CNN Fear & Greed Index data."""
        try:
            import requests

            url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
            response = requests.get(url, timeout=10)

            if response.status_code != 200:
                return pd.DataFrame()

            data = response.json()
            fng_list = data.get("fear_and_greed", [])

            if not fng_list:
                return pd.DataFrame()

            records = []
            for item in fng_list:
                date_str = item.get("date")
                if not date_str:
                    continue
                try:
                    date = pd.to_datetime(date_str)
                    records.append(
                        {
                            "date": date,
                            "fear_greed_score": float(item.get("rating", 50)),
                            "fear_greed_label": item.get("rating_text", "Neutral"),
                        }
                    )
                except (ValueError, TypeError):
                    continue

            if not records:
                return pd.DataFrame()

            df = pd.DataFrame(records)
            df = df.drop_duplicates(subset=["date"]).sort_values("date")
            df = df.tail(days)

            return df

        except ImportError:
            logger.warning("requests not available for Fear & Greed data")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to fetch Fear & Greed data: {e}")
            return pd.DataFrame()

    def _fetch_put_call_ratio(self, days: int) -> pd.DataFrame:
        """Fetch CBOE Put/Call Ratio from yfinance."""
        try:
            import yfinance as yf

            period_map = {
                range(0, 8): "7d",
                range(8, 32): "1mo",
                range(32, 92): "3mo",
                range(92, 200): "6mo",
                range(200, 400): "1y",
                range(400, 750): "2y",
            }

            period = "1y"
            for r, p in period_map.items():
                if days in r:
                    period = p
                    break

            pcr = yf.download("^CPC", period=period, progress=False)

            if pcr.empty or len(pcr) < 2:
                return pd.DataFrame()

            pcr = pcr.copy()
            pcr.index = pd.to_datetime(pcr.index)
            pcr = pcr[~pcr.index.duplicated(keep="first")]
            pcr = pcr.tail(days)

            df = pd.DataFrame()
            df["date"] = pcr.index
            df["put_call_ratio"] = pcr["Close"].values

            return df

        except ImportError:
            logger.warning("yfinance not available for Put/Call Ratio data")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to fetch Put/Call Ratio data: {e}")
            return pd.DataFrame()

    def _create_features(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """Create US market sentiment features."""
        features = pd.DataFrame()

        if "date" not in sentiment_df.columns:
            sentiment_df = sentiment_df.copy()
            sentiment_df["date"] = pd.Timestamp.today()

        features["date"] = sentiment_df["date"]

        # === VIX features ===
        if "vix_close" in sentiment_df.columns:
            features["vix_close"] = sentiment_df["vix_close"]
            features["vix_ma5"] = (
                sentiment_df["vix_close"].rolling(window=5, min_periods=1).mean()
            )
            features["vix_ma10"] = (
                sentiment_df["vix_close"].rolling(window=10, min_periods=1).mean()
            )
            features["vix_ma20"] = (
                sentiment_df["vix_close"].rolling(window=20, min_periods=1).mean()
            )

            # VIX change
            features["vix_change"] = sentiment_df["vix_close"].pct_change()
            features["vix_change_3d"] = sentiment_df["vix_close"].pct_change(3)

            # VIX level classification
            features["vix_fear_flag"] = (sentiment_df["vix_close"] > 30).astype(int)
            features["vix_calm_flag"] = (sentiment_df["vix_close"] < 15).astype(int)
            features["vix_spike_flag"] = (
                sentiment_df["vix_close"]
                > sentiment_df["vix_close"].rolling(20).mean() * 1.5
            ).astype(int)

        # === Fear & Greed features ===
        if "fear_greed_score" in sentiment_df.columns:
            features["fear_greed_score"] = sentiment_df["fear_greed_score"]
            features["fear_greed_ma5"] = (
                sentiment_df["fear_greed_score"].rolling(window=5, min_periods=1).mean()
            )
            features["fear_greed_ma10"] = (
                sentiment_df["fear_greed_score"]
                .rolling(window=10, min_periods=1)
                .mean()
            )

            # Fear & Greed change
            features["fear_greed_change"] = sentiment_df["fear_greed_score"].diff()
            features["fear_greed_change_3d"] = sentiment_df["fear_greed_score"].diff(3)

            # Extreme flags
            features["fear_greed_extreme_fear"] = (
                sentiment_df["fear_greed_score"] < 20
            ).astype(int)
            features["fear_greed_extreme_greed"] = (
                sentiment_df["fear_greed_score"] > 80
            ).astype(int)

        # === Put/Call Ratio features ===
        if "put_call_ratio" in sentiment_df.columns:
            features["put_call_ratio"] = sentiment_df["put_call_ratio"]
            features["put_call_ma5"] = (
                sentiment_df["put_call_ratio"].rolling(window=5, min_periods=1).mean()
            )
            features["put_call_ma10"] = (
                sentiment_df["put_call_ratio"].rolling(window=10, min_periods=1).mean()
            )

            # PCR extremes (contrarian signals)
            features["pcr_extreme_bearish"] = (
                sentiment_df["put_call_ratio"] > 1.2
            ).astype(int)
            features["pcr_extreme_bullish"] = (
                sentiment_df["put_call_ratio"] < 0.7
            ).astype(int)

        # === Combined sentiment score ===
        # Normalize each component to [0, 1] and average
        scores = []

        if "vix_close" in features.columns:
            # VIX: higher = more fear, invert it (0-100 scale)
            vix = features["vix_close"].clip(10, 50)
            vix_norm = ((50 - vix) / 40) * 100
            scores.append(vix_norm)

        if "fear_greed_score" in features.columns:
            scores.append(features["fear_greed_score"])

        if "put_call_ratio" in features.columns:
            # PCR: higher = more bearish, invert it
            pcr = features["put_call_ratio"].clip(0.5, 1.5)
            pcr_norm = ((1.5 - pcr) / 1.0) * 100
            scores.append(pcr_norm)

        if scores:
            features["us_sentiment_score"] = pd.concat(scores, axis=1).mean(axis=1)
            features["us_sentiment_ma5"] = (
                features["us_sentiment_score"].rolling(window=5, min_periods=1).mean()
            )
            features["us_sentiment_ma10"] = (
                features["us_sentiment_score"].rolling(window=10, min_periods=1).mean()
            )
            features["us_sentiment_trend"] = (
                features["us_sentiment_ma5"] - features["us_sentiment_ma10"]
            )

        # Fill NaN values
        features = features.fillna(0)

        return features

    def _create_default_features(self, days: int = 120) -> pd.DataFrame:
        """Create default neutral features when data is unavailable."""
        dates = pd.date_range(end=pd.Timestamp.today(), periods=days, freq="D")

        features = pd.DataFrame(
            {
                "date": dates,
                "vix_close": 0.0,
                "vix_ma5": 0.0,
                "vix_ma10": 0.0,
                "vix_ma20": 0.0,
                "vix_change": 0.0,
                "vix_change_3d": 0.0,
                "vix_fear_flag": 0,
                "vix_calm_flag": 0,
                "vix_spike_flag": 0,
                "fear_greed_score": 0.0,
                "fear_greed_ma5": 0.0,
                "fear_greed_ma10": 0.0,
                "fear_greed_change": 0.0,
                "fear_greed_change_3d": 0.0,
                "fear_greed_extreme_fear": 0,
                "fear_greed_extreme_greed": 0,
                "put_call_ratio": 0.0,
                "put_call_ma5": 0.0,
                "put_call_ma10": 0.0,
                "pcr_extreme_bearish": 0,
                "pcr_extreme_bullish": 0,
                "us_sentiment_score": 0.0,
                "us_sentiment_ma5": 0.0,
                "us_sentiment_ma10": 0.0,
                "us_sentiment_trend": 0.0,
            }
        )

        return features


def get_us_market_sentiment_features(
    stock_code: str, days: int = 120, cache: Optional[FeatureCache] = None
) -> pd.DataFrame:
    """Get US market sentiment features for a stock."""
    extractor = USMarketSentimentFeatures(cache)
    return extractor.get_or_extract(stock_code, days=days)
