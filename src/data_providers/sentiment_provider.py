"""Sentiment data provider for stock news and social media sentiment analysis."""

import time
import logging
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from .base import BaseDataProvider

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
akshare = None
snownlp = None


def _get_akshare():
    """Lazy load akshare."""
    global akshare
    if akshare is None:
        try:
            import akshare as akshare_pkg

            akshare = akshare_pkg
        except ImportError:
            logger.warning(
                "[sentiment] akshare not installed. Install with: pip install akshare"
            )
            return None
    return akshare


def _get_snownlp():
    """Lazy load snownlp for Chinese sentiment analysis."""
    global snownlp
    if snownlp is None:
        try:
            from snownlp import SnowNLP

            snownlp = SnowNLP
        except ImportError:
            logger.warning(
                "[sentiment] snownlp not installed. Install with: pip install snownlp"
            )
            return None
    return snownlp


# Chinese sentiment keywords
POSITIVE_KEYWORDS = [
    "利好",
    "上涨",
    "增长",
    "突破",
    "涨停",
    "强势",
    "看好",
    "买入",
    "增持",
    "业绩大增",
    "盈利",
    "利润增长",
    "营收增长",
    "超预期",
    "创新高",
    "反弹",
    "反转",
    "牛市",
    "机会",
    "潜力",
    "优质",
    "龙头",
    "热点",
    "爆发",
    "recommend",
    "buy",
    "growth",
    "positive",
    "bullish",
    "upgrade",
    "outperform",
]

NEGATIVE_KEYWORDS = [
    "利空",
    "下跌",
    "下降",
    "跌破",
    "跌停",
    "弱势",
    "看空",
    "卖出",
    "减持",
    "业绩下滑",
    "亏损",
    "利润下降",
    "营收下降",
    "不及预期",
    "创新低",
    "暴跌",
    "崩盘",
    "熊市",
    "风险",
    "警告",
    "问题",
    "危机",
    "暴雷",
    "减持",
    "sell",
    "decline",
    "negative",
    "bearish",
    "downgrade",
    "underperform",
]


class SentimentProvider(BaseDataProvider):
    """Sentiment data provider for stock news analysis."""

    def __init__(self):
        self._name = "sentiment"
        self._cache_duration = timedelta(hours=6)  # Cache for 6 hours
        self._news_cache: Dict[str, Tuple[datetime, pd.DataFrame]] = {}

    @property
    def name(self) -> str:
        return self._name

    def _convert_code(self, stock_code: str) -> str:
        """Convert stock code to akshare format."""
        code = stock_code.split(".")[0]
        return code

    def _analyze_sentiment_keywords(self, text: str) -> float:
        """Analyze sentiment using keyword matching.

        Returns:
            float: Sentiment score between -1 (negative) and 1 (positive)
        """
        if not text:
            return 0.0

        text_lower = text.lower()
        positive_count = sum(1 for kw in POSITIVE_KEYWORDS if kw.lower() in text_lower)
        negative_count = sum(1 for kw in NEGATIVE_KEYWORDS if kw.lower() in text_lower)

        total = positive_count + negative_count
        if total == 0:
            return 0.0

        return (positive_count - negative_count) / total

    def _analyze_sentiment_snownlp(self, text: str) -> float:
        """Analyze sentiment using SnowNLP for Chinese text.

        Returns:
            float: Sentiment score between -1 (negative) and 1 (positive)
        """
        SnowNLP = _get_snownlp()
        if SnowNLP is None:
            return 0.0

        try:
            if not text:
                return 0.0
            s = SnowNLP(text)
            # SnowNLP returns 0-1, convert to -1 to 1
            return (s.sentiments - 0.5) * 2
        except Exception:
            return 0.0

    def _analyze_sentiment(self, text: str, use_snownlp: bool = True) -> float:
        """Combine keyword and SnowNLP sentiment analysis."""
        keyword_score = self._analyze_sentiment_keywords(text)
        snownlp_score = self._analyze_sentiment_snownlp(text) if use_snownlp else 0.0

        # Weighted average: 60% keywords, 40% SnowNLP
        if use_snownlp and snownlp != 0:
            return keyword_score * 0.6 + snownlp_score * 0.4
        return keyword_score

    def _fetch_news_akshare(self, stock_code: str, days: int = 30) -> pd.DataFrame:
        """Fetch news data from AKShare.

        使用可用的新闻API:
        1. stock_news_main_cx - 财新新闻 (通用财经)
        2. news_cctv - 央视新闻 (通用新闻)
        3. news_economic_baidu - 百度财经日历
        """
        ak = _get_akshare()
        if ak is None:
            return pd.DataFrame()

        try:
            # Method 1: stock_news_main_cx (财新新闻 - 通用财经新闻)
            try:
                df_cx = ak.stock_news_main_cx()
                if df_cx is not None and not df_cx.empty:
                    df_cx = df_cx.rename(columns={"summary": "title"})
                    df_cx["content"] = df_cx["title"]  # 财新只有摘要
                    df_cx["source"] = "caixin"
                    df_cx["date"] = pd.Timestamp.today().strftime("%Y-%m-%d")
                    logger.info(f"[{self.name}] Fetched {len(df_cx)} news from caixin")
                    return df_cx[["date", "title", "content", "source"]].head(100)
            except Exception as e:
                logger.debug(f"[{self.name}] Failed to fetch news from caixin: {e}")

            # Method 2: news_cctv (央视新闻)
            try:
                df_cctv = ak.news_cctv()
                if df_cctv is not None and not df_cctv.empty:
                    df_cctv["source"] = "cctv"
                    df_cctv = df_cctv.rename(columns={"date": "date"})
                    logger.info(f"[{self.name}] Fetched {len(df_cctv)} news from cctv")
                    return df_cctv[["date", "title", "content", "source"]].head(100)
            except Exception as e:
                logger.debug(f"[{self.name}] Failed to fetch news from cctv: {e}")

            # Method 3: news_economic_baidu (百度财经日历 - 虽然不是新闻但有事件)
            try:
                df_baidu = ak.news_economic_baidu()
                if df_baidu is not None and not df_baidu.empty:
                    df_baidu = df_baidu.rename(columns={"事件": "title"})
                    df_baidu["content"] = df_baidu["title"]
                    df_baidu["source"] = "baidu_calendar"
                    df_baidu = df_baidu.rename(columns={"日期": "date"})
                    logger.info(
                        f"[{self.name}] Fetched {len(df_baidu)} events from baidu"
                    )
                    return df_baidu[["date", "title", "content", "source"]].head(100)
            except Exception as e:
                logger.debug(f"[{self.name}] Failed to fetch from baidu: {e}")

            logger.warning(f"[{self.name}] No news data available for {stock_code}")
            return pd.DataFrame()

        except Exception as e:
            logger.warning(f"[{self.name}] Failed to fetch news for {stock_code}: {e}")
            return pd.DataFrame()

    def fetch_news_sentiment(
        self, stock_code: str, days: int = 30, use_snownlp: bool = True
    ) -> pd.DataFrame:
        """Fetch news and calculate sentiment scores.

        Args:
            stock_code: Stock code
            days: Number of days to look back
            use_snownlp: Whether to use SnowNLP for sentiment analysis

        Returns:
            DataFrame with columns: date, sentiment_score, news_count, title, source
        """
        # Check cache
        if stock_code in self._news_cache:
            cache_time, cached_df = self._news_cache[stock_code]
            if datetime.now() - cache_time < self._cache_duration:
                return cached_df

        news_df = self._fetch_news_akshare(stock_code, days)

        # 创建完整的日期范围
        dates = pd.date_range(end=pd.Timestamp.today(), periods=days, freq="D")
        dates = dates.normalize()

        if news_df.empty:
            # Return default neutral sentiment for past N days
            result = pd.DataFrame(
                {
                    "date": dates,
                    "sentiment_score": 0.0,
                    "news_count": 0,
                    "title": "No news available",
                    "source": "none",
                }
            )
        else:
            # Calculate sentiment for each news item
            news_df["sentiment_score"] = news_df.apply(
                lambda row: self._analyze_sentiment(
                    f"{row.get('title', '')} {row.get('content', '')}",
                    use_snownlp=use_snownlp,
                ),
                axis=1,
            )

            # 解析日期
            if "date" in news_df.columns:
                news_df["date"] = pd.to_datetime(news_df["date"], errors="coerce")
                news_df["date"] = news_df["date"].fillna(pd.Timestamp.today())
                news_df["date"] = news_df["date"].dt.normalize()
            else:
                news_df["date"] = pd.Timestamp.today().normalize()

            # 聚合每日情绪
            daily_sentiment = (
                news_df.groupby("date")
                .agg({"sentiment_score": "mean", "title": "count"})
                .reset_index()
            )
            daily_sentiment = daily_sentiment.rename(columns={"title": "news_count"})

            # 添加到完整日期范围
            result = pd.DataFrame({"date": dates})
            result = result.merge(daily_sentiment, on="date", how="left")
            result["sentiment_score"] = result["sentiment_score"].fillna(0.0)
            result["news_count"] = result["news_count"].fillna(0).astype(int)

            # 如果只有今天的新闻，使用今天的平均情绪填充所有日期
            # 这是因为财新新闻不提供历史新闻的日期
            if (
                result["sentiment_score"].sum() != 0
                and result[result["sentiment_score"] != 0].shape[0] <= 1
            ):
                avg_sentiment = result[result["sentiment_score"] != 0][
                    "sentiment_score"
                ].mean()
                total_news = result["news_count"].sum()
                # 使用衰减的历史情绪 (今天是100%，昨天是80%，前天是60%...)
                for i, row in result.iterrows():
                    days_ago = result.index[-1] - i
                    decay = max(0.5 ** (days_ago / 7), 0.1)  # 每周衰减50%
                    result.loc[i, "sentiment_score"] = avg_sentiment * decay

            # 添加标题和来源 (使用最近的新闻)
            if not news_df.empty:
                result["title"] = "Aggregated"
                result["source"] = (
                    news_df["source"].iloc[0]
                    if "source" in news_df.columns
                    else "unknown"
                )
            else:
                result["title"] = "No news"
                result["source"] = "none"

        # Cache the result
        self._news_cache[stock_code] = (datetime.now(), result)

        logger.info(
            f"[{self.name}] Fetched sentiment for {stock_code}: {len(result)} days, avg_score={result['sentiment_score'].mean():.3f}"
        )
        return result

    def get_sentiment_features(self, stock_code: str, days: int = 30) -> pd.DataFrame:
        """Get sentiment features for a stock.

        Returns:
            DataFrame with sentiment features
        """
        sentiment_df = self.fetch_news_sentiment(stock_code, days)

        if sentiment_df.empty:
            return pd.DataFrame()

        # Create feature columns
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

    def fetch(
        self,
        stock_code: str,
        days: int = 120,
        retry_count: int = 3,
        retry_delay: float = 1.0,
    ) -> pd.DataFrame:
        """Fetch sentiment data for a stock.

        This is the main fetch method required by BaseDataProvider.
        Returns sentiment features aligned with price data.
        """
        for attempt in range(retry_count):
            try:
                features = self.get_sentiment_features(stock_code, days)

                if features.empty:
                    # Return default neutral sentiment features
                    # Use yesterday as end date to match technical features
                    dates = pd.date_range(
                        end=pd.Timestamp.today() - pd.Timedelta(days=1),
                        periods=days,
                        freq="D",
                    )
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
                else:
                    # Ensure date column exists
                    if "date" not in features.columns:
                        features["date"] = (
                            pd.Timestamp.today() - pd.Timedelta(days=1)
                        ).strftime("%Y-%m-%d")

                # Convert date to datetime
                features["date"] = pd.to_datetime(features["date"])

                # Sort by date
                features = features.sort_values("date").reset_index(drop=True)

                logger.info(
                    f"[{self.name}] Successfully fetched sentiment for {stock_code}"
                )
                return features

            except Exception as e:
                logger.warning(
                    f"[{self.name}] Attempt {attempt + 1}/{retry_count} "
                    f"failed for {stock_code}: {e}"
                )
                if attempt < retry_count - 1:
                    time.sleep(retry_delay)

        # Return empty DataFrame on failure
        logger.error(f"[{self.name}] All attempts failed for {stock_code}")
        return pd.DataFrame()


# Convenience function
def fetch_stock_sentiment(stock_code: str, days: int = 30) -> pd.DataFrame:
    """Fetch sentiment data for a stock."""
    provider = SentimentProvider()
    return provider.fetch(stock_code, days)
