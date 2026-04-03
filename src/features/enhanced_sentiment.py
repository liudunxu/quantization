"""Enhanced sentiment analysis module with multi-source news, social media, and search trends.

Provides:
- NewsSentimentAnalyzer: Multi-source news aggregation with keyword + ML sentiment
- SocialMediaAnalyzer: Weibo, Reddit, X/Twitter sentiment analysis
- SearchTrendAnalyzer: Baidu Index, Google Trends analysis
- CombinedSentiment: Unified sentiment scoring from all sources
"""

import logging
import re
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Lazy imports
snownlp = None
jieba = None


def _get_snownlp():
    """Lazy load snownlp."""
    global snownlp
    if snownlp is None:
        try:
            from snownlp import SnowNLP

            snownlp = SnowNLP
        except ImportError:
            logger.debug("snownlp not installed: pip install snownlp")
    return snownlp


def _get_jieba():
    """Lazy load jieba."""
    global jieba
    if jieba is None:
        try:
            import jieba as jieba_pkg

            jieba = jieba_pkg
        except ImportError:
            logger.debug("jieba not installed: pip install jieba")
    return jieba


# ========== Sentiment Lexicons ==========

# Financial-specific sentiment keywords (Chinese + English)
FINANCIAL_POSITIVE = {
    # Chinese - strong positive
    "涨停": 1.0,
    "利好": 0.8,
    "超预期": 0.7,
    "创新高": 0.6,
    "强势": 0.5,
    "突破": 0.6,
    "暴涨": 0.8,
    "大涨": 0.7,
    "飙升": 0.8,
    "反弹": 0.4,
    "买入": 0.6,
    "增持": 0.5,
    "看好": 0.5,
    "推荐": 0.5,
    "上调": 0.4,
    "业绩大增": 0.7,
    "利润增长": 0.6,
    "营收增长": 0.5,
    "盈利": 0.5,
    "牛市": 0.6,
    "龙头": 0.4,
    "优质": 0.4,
    "潜力": 0.3,
    "机会": 0.3,
    "爆发": 0.6,
    "利好消息": 0.7,
    "重大利好": 0.8,
    "资金流入": 0.5,
    "主力买入": 0.5,
    "机构看好": 0.5,
    "北向资金": 0.3,
    "外资加仓": 0.4,
    # English
    "bullish": 0.5,
    "buy": 0.5,
    "upgrade": 0.5,
    "outperform": 0.5,
    "overweight": 0.4,
    "beat": 0.4,
    "surge": 0.6,
    "rally": 0.5,
    "breakout": 0.5,
    "growth": 0.4,
    "profit": 0.4,
    "record": 0.4,
}

FINANCIAL_NEGATIVE = {
    # Chinese - strong negative
    "跌停": -1.0,
    "利空": -0.8,
    "不及预期": -0.7,
    "创新低": -0.6,
    "弱势": -0.5,
    "跌破": -0.6,
    "暴跌": -0.8,
    "大跌": -0.7,
    "崩盘": -0.9,
    "闪崩": -0.9,
    "卖出": -0.6,
    "减持": -0.5,
    "看空": -0.5,
    "下调": -0.4,
    "评级下调": -0.5,
    "业绩下滑": -0.7,
    "亏损": -0.7,
    "利润下降": -0.6,
    "营收下降": -0.5,
    "熊市": -0.6,
    "风险": -0.4,
    "警告": -0.4,
    "危机": -0.7,
    "暴雷": -0.8,
    "利空消息": -0.7,
    "重大利空": -0.8,
    "资金流出": -0.5,
    "主力卖出": -0.5,
    "机构看空": -0.5,
    "停牌": -0.3,
    "退市": -0.8,
    "调查": -0.5,
    "处罚": -0.6,
    # English
    "bearish": -0.5,
    "sell": -0.5,
    "downgrade": -0.5,
    "underperform": -0.5,
    "underweight": -0.4,
    "miss": -0.4,
    "plunge": -0.6,
    "crash": -0.8,
    "breakdown": -0.5,
    "loss": -0.5,
    "decline": -0.4,
    "risk": -0.3,
}

# Negation words that flip sentiment
NEGATION_WORDS = [
    "不",
    "没",
    "非",
    "无",
    "未",
    "否",
    "别",
    "莫",
    "勿",
    "缺乏",
    "难以",
    "未能",
]

# Intensifiers that amplify sentiment
INTENSIFIERS = {
    "大幅": 1.5,
    "显著": 1.3,
    "持续": 1.2,
    "加速": 1.3,
    "再次": 1.2,
    "突然": 1.4,
    "急剧": 1.4,
    "全面": 1.2,
    "严重": 1.5,
    "极其": 1.5,
    "very": 1.5,
    "extremely": 1.5,
    "significantly": 1.3,
    "sharply": 1.4,
}

# Social media specific keywords
SOCIAL_POSITIVE = [
    "冲啊",
    "加仓",
    "满仓",
    "allin",
    "yyds",
    "牛逼",
    "起飞",
    "吃肉",
    "涨停板",
    "打板",
    "连板",
    "龙头",
    "主线",
    "热点",
]
SOCIAL_NEGATIVE = [
    "割肉",
    "跑路",
    "清仓",
    "被套",
    "踏空",
    "韭菜",
    "凉凉",
    "关灯",
    "吃面",
    "爆仓",
    "跌停板",
    "跑路",
    "血亏",
]


class NewsSentimentAnalyzer:
    """Multi-source news sentiment analyzer with weighted scoring."""

    def __init__(self):
        self._source_weights = {
            "caixin": 0.9,  # 财新 - 高质量财经
            "xinhua": 0.85,  # 新华社 - 权威
            "cctv": 0.8,  # 央视 - 权威
            "eastmoney": 0.7,  # 东方财富 - 财经垂直
            "sina": 0.65,  # 新浪财经
            "cls": 0.75,  # 财联社 - 快速
            "baidu": 0.6,  # 百度财经
            "yahoo": 0.65,  # Yahoo Finance
            "reuters": 0.85,  # Reuters
            "bloomberg": 0.9,  # Bloomberg
            "default": 0.5,
        }

    def analyze_text(self, text: str) -> float:
        """Analyze sentiment of a single text with weighted keyword + NLP approach.

        Args:
            text: Text to analyze

        Returns:
            Sentiment score in [-1.0, 1.0]
        """
        if not text or not isinstance(text, str):
            return 0.0

        text_lower = text.lower()

        # Method 1: Weighted keyword scoring with negation detection
        keyword_score = self._keyword_sentiment(text, text_lower)

        # Method 2: SnowNLP NLP sentiment
        nlp_score = self._nlp_sentiment(text)

        # Combine: 70% keyword (domain-specific), 30% NLP (general)
        if nlp_score != 0.0:
            return keyword_score * 0.7 + nlp_score * 0.3
        return keyword_score

    def _keyword_sentiment(self, text: str, text_lower: str) -> float:
        """Weighted keyword sentiment with negation and intensifier handling."""
        total_score = 0.0
        total_weight = 0.0

        # Tokenize if jieba available, otherwise use character-level matching
        words = self._tokenize(text)

        for i, word in enumerate(words):
            word_lower = word.lower()

            # Check if negated
            is_negated = False
            for j in range(max(0, i - 3), i):
                if words[j] in NEGATION_WORDS:
                    is_negated = True
                    break

            # Check for intensifier
            intensity = 1.0
            for j in range(max(0, i - 2), i):
                if words[j] in INTENSIFIERS:
                    intensity = INTENSIFIERS[words[j]]
                    break

            # Score from financial lexicons
            score = 0.0
            if word in FINANCIAL_POSITIVE:
                score = FINANCIAL_POSITIVE[word]
            elif word in FINANCIAL_NEGATIVE:
                score = FINANCIAL_NEGATIVE[word]
            elif word_lower in FINANCIAL_POSITIVE:
                score = FINANCIAL_POSITIVE[word_lower]
            elif word_lower in FINANCIAL_NEGATIVE:
                score = FINANCIAL_NEGATIVE[word_lower]

            if score != 0.0:
                if is_negated:
                    score *= -0.5  # Negation reduces and flips
                score *= intensity
                total_score += score
                total_weight += abs(score)

        if total_weight == 0:
            return 0.0

        # Normalize to [-1, 1]
        return max(-1.0, min(1.0, total_score / total_weight))

    def _nlp_sentiment(self, text: str) -> float:
        """SnowNLP-based sentiment analysis."""
        SnowNLP = _get_snownlp()
        if SnowNLP is None:
            return 0.0

        try:
            # Truncate long texts for performance
            text = text[:2000]
            s = SnowNLP(text)
            return (s.sentiments - 0.5) * 2  # Convert [0,1] to [-1,1]
        except Exception as e:
            logger.debug(f"SnowNLP analysis failed: {e}")
            return 0.0

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        jieba_mod = _get_jieba()
        if jieba_mod:
            try:
                return list(jieba_mod.cut(text))
            except Exception:
                pass

        # Fallback: extract 2-4 character phrases + English words
        tokens = re.findall(r"[a-zA-Z]+|[\u4e00-\u9fff]{2,4}", text)
        return tokens

    def analyze_news_batch(
        self,
        news_items: List[Dict],
        source: str = "default",
    ) -> pd.DataFrame:
        """Analyze sentiment for a batch of news items.

        Args:
            news_items: List of dicts with 'title', 'content', 'date' keys
            source: News source name for weighting

        Returns:
            DataFrame with sentiment scores
        """
        weight = self._source_weights.get(source, self._source_weights["default"])
        results = []

        for item in news_items:
            text = f"{item.get('title', '')} {item.get('content', '')}"
            raw_score = self.analyze_text(text)
            weighted_score = raw_score * weight

            results.append(
                {
                    "date": item.get("date", datetime.now().strftime("%Y-%m-%d")),
                    "title": item.get("title", ""),
                    "source": source,
                    "raw_sentiment": round(raw_score, 4),
                    "weighted_sentiment": round(weighted_score, 4),
                    "source_weight": weight,
                }
            )

        return pd.DataFrame(results)


class SocialMediaAnalyzer:
    """Social media sentiment analyzer for Weibo, Reddit, X/Twitter."""

    def __init__(self):
        self._platform_weights = {
            "weibo": 0.8,  # 微博 - A股情绪风向标
            "xueqiu": 0.85,  # 雪球 - 专业投资者
            "guba_eastmoney": 0.6,  # 股吧 - 散户情绪
            "reddit_wsb": 0.7,  # WallStreetBets - 美股情绪
            "reddit_stocks": 0.65,
            "twitter": 0.5,  # X/Twitter - 噪声较大
        }

    def analyze_weibo_sentiment(self, stock_name: str, posts: List[str]) -> Dict:
        """Analyze Weibo sentiment for a stock.

        Args:
            stock_name: Stock name or code for filtering
            posts: List of Weibo post texts

        Returns:
            Dict with sentiment metrics
        """
        if not posts:
            return self._empty_result("weibo")

        platform_weight = self._platform_weights["weibo"]
        scores = []

        for post in posts:
            score = self._analyze_social_text(post)
            scores.append(score * platform_weight)

        return self._aggregate_scores(scores, "weibo", len(posts))

    def analyze_reddit_sentiment(self, stock_ticker: str, posts: List[str]) -> Dict:
        """Analyze Reddit sentiment (WSB, stocks).

        Args:
            stock_ticker: Stock ticker (e.g., AAPL, TSLA)
            posts: List of Reddit post texts

        Returns:
            Dict with sentiment metrics
        """
        if not posts:
            return self._empty_result("reddit")

        platform_weight = self._platform_weights.get("reddit_wsb", 0.7)
        scores = []

        for post in posts:
            # Check if post is relevant to the ticker
            if stock_ticker.lower() in post.lower() or f"${stock_ticker}" in post:
                score = self._analyze_social_text(post, is_english=True)
                scores.append(score * platform_weight)

        if not scores:
            return self._empty_result("reddit")

        return self._aggregate_scores(scores, "reddit", len(posts))

    def _analyze_social_text(self, text: str, is_english: bool = False) -> float:
        """Analyze social media text with slang awareness."""
        if not text:
            return 0.0

        text_lower = text.lower()
        score = 0.0
        count = 0

        # Check social media specific keywords
        social_words = SOCIAL_POSITIVE + SOCIAL_NEGATIVE
        for word in social_words:
            if word.lower() in text_lower:
                if word in SOCIAL_POSITIVE:
                    score += 0.6
                    count += 1
                else:
                    score -= 0.6
                    count += 1

        # Also check financial keywords
        analyzer = NewsSentimentAnalyzer()
        financial_score = analyzer.analyze_text(text)

        # Combine: 50% social-specific, 50% financial
        if count > 0:
            social_avg = score / count
            return social_avg * 0.5 + financial_score * 0.5
        return financial_score

    def _aggregate_scores(
        self, scores: List[float], platform: str, post_count: int
    ) -> Dict:
        """Aggregate sentiment scores into metrics."""
        if not scores:
            return self._empty_result(platform)

        arr = np.array(scores)
        return {
            "platform": platform,
            "sentiment_score": round(float(np.mean(arr)), 4),
            "sentiment_std": round(float(np.std(arr)), 4),
            "post_count": post_count,
            "positive_ratio": round(float(np.mean(arr > 0.1)), 4),
            "negative_ratio": round(float(np.mean(arr < -0.1)), 4),
            "neutral_ratio": round(float(np.mean(np.abs(arr) <= 0.1)), 4),
            "bullish_intensity": round(
                float(np.mean(arr[arr > 0])) if np.any(arr > 0) else 0, 4
            ),
            "bearish_intensity": round(
                float(np.mean(arr[arr < 0])) if np.any(arr < 0) else 0, 4
            ),
            "volatility": round(float(np.std(arr) / (abs(np.mean(arr)) + 1e-8)), 4),
        }

    def _empty_result(self, platform: str) -> Dict:
        """Return empty result for a platform."""
        return {
            "platform": platform,
            "sentiment_score": 0.0,
            "sentiment_std": 0.0,
            "post_count": 0,
            "positive_ratio": 0.0,
            "negative_ratio": 0.0,
            "neutral_ratio": 1.0,
            "bullish_intensity": 0.0,
            "bearish_intensity": 0.0,
            "volatility": 0.0,
        }

    def analyze_multiple_platforms(
        self,
        stock_code: str,
        weibo_posts: Optional[List[str]] = None,
        reddit_posts: Optional[List[str]] = None,
        twitter_posts: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Analyze sentiment across multiple social platforms.

        Args:
            stock_code: Stock code
            weibo_posts: List of Weibo posts
            reddit_posts: List of Reddit posts
            twitter_posts: List of Twitter posts

        Returns:
            DataFrame with per-platform sentiment metrics
        """
        results = []

        if weibo_posts:
            stock_name = stock_code.split(".")[0]
            results.append(self.analyze_weibo_sentiment(stock_name, weibo_posts))

        if reddit_posts:
            ticker = stock_code.split(".")[0]
            results.append(self.analyze_reddit_sentiment(ticker, reddit_posts))

        if twitter_posts:
            # Twitter uses same analysis as social text
            platform_weight = self._platform_weights.get("twitter", 0.5)
            scores = [
                self._analyze_social_text(t) * platform_weight for t in twitter_posts
            ]
            results.append(
                self._aggregate_scores(scores, "twitter", len(twitter_posts))
            )

        if not results:
            return pd.DataFrame()

        return pd.DataFrame(results)


class SearchTrendAnalyzer:
    """Analyze search trends as sentiment proxy."""

    def __init__(self):
        self._trend_cache: Dict[str, pd.DataFrame] = {}

    def analyze_search_trend(
        self,
        keywords: List[str],
        trend_data: pd.DataFrame,
    ) -> Dict:
        """Analyze search trend data for sentiment signals.

        Args:
            keywords: Search keywords
            trend_data: DataFrame with 'date' and 'search_volume' columns

        Returns:
            Dict with trend-based sentiment metrics
        """
        if trend_data.empty or "search_volume" not in trend_data.columns:
            return self._empty_result()

        volume = trend_data["search_volume"].astype(float)

        # Trend direction (rising = increasing interest)
        ma_short = volume.rolling(3, min_periods=1).mean()
        ma_long = volume.rolling(7, min_periods=1).mean()
        trend_direction = (ma_short.iloc[-1] - ma_long.iloc[-1]) / (
            ma_long.iloc[-1] + 1e-8
        )

        # Volume spike (sudden interest surge)
        vol_mean = volume.mean()
        vol_std = volume.std()
        is_spike = (
            bool(volume.iloc[-1] > vol_mean + 2 * vol_std) if vol_std > 0 else False
        )

        # Acceleration (rate of change of trend)
        changes = volume.pct_change().dropna()
        acceleration = changes.mean() if len(changes) > 0 else 0.0

        # Consistency (how stable the trend is)
        cv = float(vol_std / (vol_mean + 1e-8))  # Coefficient of variation

        return {
            "trend_direction": round(trend_direction, 4),
            "trend_strength": round(abs(trend_direction), 4),
            "is_volume_spike": is_spike,
            "spike_ratio": round(float(volume.iloc[-1] / (vol_mean + 1e-8)), 4),
            "acceleration": round(acceleration, 4),
            "consistency": round(1.0 - min(cv, 1.0), 4),  # Higher = more stable
            "avg_volume": round(float(vol_mean), 2),
            "keywords": keywords,
        }

    def _empty_result(self) -> Dict:
        """Return empty trend result."""
        return {
            "trend_direction": 0.0,
            "trend_strength": 0.0,
            "is_volume_spike": False,
            "spike_ratio": 1.0,
            "acceleration": 0.0,
            "consistency": 0.0,
            "avg_volume": 0.0,
            "keywords": [],
        }


class CombinedSentiment:
    """Combine multiple sentiment sources into a unified score."""

    def __init__(self):
        # Source weights for final combination
        self.source_weights = {
            "news": 0.35,
            "social": 0.30,
            "search_trend": 0.15,
            "technical_sentiment": 0.20,  # Market-based sentiment proxy
        }

    def combine(
        self,
        news_score: float = 0.0,
        news_confidence: float = 0.5,
        social_score: float = 0.0,
        social_confidence: float = 0.5,
        search_score: float = 0.0,
        search_confidence: float = 0.5,
        technical_score: float = 0.0,
    ) -> Dict:
        """Combine sentiment scores from multiple sources.

        Args:
            news_score: News sentiment score [-1, 1]
            news_confidence: Confidence in news score [0, 1]
            social_score: Social media sentiment score [-1, 1]
            social_confidence: Confidence in social score [0, 1]
            search_score: Search trend sentiment score [-1, 1]
            search_confidence: Confidence in search score [0, 1]
            technical_score: Technical/market-based sentiment proxy

        Returns:
            Dict with combined sentiment and breakdown
        """
        scores = {
            "news": (news_score, news_confidence),
            "social": (social_score, social_confidence),
            "search": (search_score, search_confidence),
            "technical": (technical_score, 0.8),  # Technical has fixed confidence
        }

        weighted_sum = 0.0
        weight_sum = 0.0
        breakdown = {}

        for source, (score, confidence) in scores.items():
            effective_weight = self.source_weights.get(source, 0.1) * confidence
            weighted_sum += score * effective_weight
            weight_sum += effective_weight
            breakdown[source] = {
                "score": score,
                "confidence": confidence,
                "effective_weight": effective_weight,
            }

        # Normalize
        combined_score = weighted_sum / weight_sum if weight_sum > 0 else 0.0

        # Overall confidence (weighted average of confidences)
        overall_confidence = sum(
            c * self.source_weights.get(s, 0.1) for s, (_, c) in scores.items()
        ) / sum(self.source_weights.get(s, 0.1) for s in scores)

        # Determine sentiment label
        if combined_score > 0.3:
            label = "strong_bullish"
        elif combined_score > 0.1:
            label = "bullish"
        elif combined_score > -0.1:
            label = "neutral"
        elif combined_score > -0.3:
            label = "bearish"
        else:
            label = "strong_bearish"

        # Consensus score: how much sources agree
        valid_scores = [s for s, c in scores.values() if c > 0.1]
        if len(valid_scores) > 1:
            consensus = 1.0 - min(np.std(valid_scores), 1.0)
        else:
            consensus = 0.5

        return {
            "combined_score": round(combined_score, 4),
            "confidence": round(overall_confidence, 4),
            "label": label,
            "consensus": round(consensus, 4),
            "breakdown": breakdown,
            "active_sources": sum(1 for _, c in scores.values() if c > 0.1),
        }
