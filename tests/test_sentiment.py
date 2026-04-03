"""Unit tests for enhanced sentiment analysis module."""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.enhanced_sentiment import (
    NewsSentimentAnalyzer,
    SocialMediaAnalyzer,
    SearchTrendAnalyzer,
    CombinedSentiment,
)


class TestNewsSentimentAnalyzer(unittest.TestCase):
    """Test news sentiment analysis."""

    def setUp(self):
        self.analyzer = NewsSentimentAnalyzer()

    def test_positive_chinese_text(self):
        """Test positive Chinese sentiment."""
        text = "该股业绩大增，利好消息频出，机构看好并上调评级"
        score = self.analyzer.analyze_text(text)
        self.assertGreater(score, 0)

    def test_negative_chinese_text(self):
        """Test negative Chinese sentiment."""
        text = "该公司业绩下滑，面临亏损风险，机构下调评级并警告"
        score = self.analyzer.analyze_text(text)
        self.assertLess(score, 0)

    def test_english_positive_text(self):
        """Test positive English sentiment."""
        text = "Strong earnings growth, analysts upgrade and recommend buy"
        score = self.analyzer.analyze_text(text)
        self.assertGreater(score, 0)

    def test_english_negative_text(self):
        """Test negative English sentiment."""
        text = "Company faces crisis, analysts downgrade to sell amid losses"
        score = self.analyzer.analyze_text(text)
        self.assertLess(score, 0)

    def test_neutral_text(self):
        """Test neutral sentiment."""
        text = "公司发布年度报告，董事会将于下周召开"
        score = self.analyzer.analyze_text(text)
        # Neutral text should be close to 0
        self.assertGreaterEqual(score, -0.3)
        self.assertLessEqual(score, 0.3)

    def test_empty_text(self):
        """Test empty text returns 0."""
        self.assertEqual(self.analyzer.analyze_text(""), 0.0)
        self.assertEqual(self.analyzer.analyze_text(None), 0.0)

    def test_negation_handling(self):
        """Test negation flips sentiment."""
        positive_text = "业绩增长"
        negated_text = "业绩不增长"

        pos_score = self.analyzer.analyze_text(positive_text)
        neg_score = self.analyzer.analyze_text(negated_text)

        # Negated should be lower or opposite
        self.assertLess(neg_score, pos_score)

    def test_analyze_news_batch(self):
        """Test batch news analysis."""
        news_items = [
            {"title": "利好消息", "content": "业绩大增", "date": "2024-01-15"},
            {"title": "利空警告", "content": "业绩下滑", "date": "2024-01-16"},
            {"title": "中性公告", "content": "董事会召开", "date": "2024-01-17"},
        ]

        result = self.analyzer.analyze_news_batch(news_items, source="caixin")
        self.assertFalse(result.empty)
        self.assertEqual(len(result), 3)
        self.assertIn(
            "sentiment_score"
            if "sentiment_score" in result.columns
            else "raw_sentiment",
            result.columns,
        )


class TestSocialMediaAnalyzer(unittest.TestCase):
    """Test social media sentiment analysis."""

    def setUp(self):
        self.analyzer = SocialMediaAnalyzer()

    def test_weibo_positive(self):
        """Test positive Weibo sentiment."""
        posts = ["冲啊！加仓满仓干", "yyds，起飞吃肉", "涨停板，龙头主线"]
        result = self.analyzer.analyze_weibo_sentiment("000001", posts)

        self.assertEqual(result["platform"], "weibo")
        self.assertGreater(result["post_count"], 0)
        self.assertGreater(result["sentiment_score"], 0)

    def test_weibo_negative(self):
        """Test negative Weibo sentiment."""
        posts = ["割肉跑路，清仓被套", "韭菜凉凉，关灯吃面", "爆仓跌停，血亏"]
        result = self.analyzer.analyze_weibo_sentiment("000001", posts)

        self.assertLess(result["sentiment_score"], 0)

    def test_empty_posts(self):
        """Test empty posts return default."""
        result = self.analyzer.analyze_weibo_sentiment("000001", [])
        self.assertEqual(result["post_count"], 0)
        self.assertEqual(result["sentiment_score"], 0.0)

    def test_reddit_sentiment(self):
        """Test Reddit sentiment analysis."""
        posts = [
            f"$AAPL is bullish, strong earnings beat!",
            f"Buying more $AAPL, upgrade to outperform",
            f"$AAPL crash coming, downgrade to sell",
        ]
        result = self.analyzer.analyze_reddit_sentiment("AAPL", posts)

        self.assertEqual(result["platform"], "reddit")
        self.assertGreater(result["post_count"], 0)

    def test_multiple_platforms(self):
        """Test multi-platform analysis."""
        result = self.analyzer.analyze_multiple_platforms(
            stock_code="000001.SZ",
            weibo_posts=["涨停利好", "加仓冲"],
            reddit_posts=["$AAPL bullish"],
        )

        self.assertFalse(result.empty)
        self.assertIn("weibo", result["platform"].values)


class TestSearchTrendAnalyzer(unittest.TestCase):
    """Test search trend analysis."""

    def setUp(self):
        self.analyzer = SearchTrendAnalyzer()

    def test_rising_trend(self):
        """Test detection of rising search trend."""
        trend_data = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=14),
                "search_volume": [
                    10,
                    12,
                    15,
                    20,
                    25,
                    30,
                    35,
                    40,
                    50,
                    60,
                    70,
                    80,
                    90,
                    100,
                ],
            }
        )

        result = self.analyzer.analyze_search_trend(["测试股票"], trend_data)

        self.assertGreater(result["trend_direction"], 0)
        self.assertGreater(result["trend_strength"], 0)
        self.assertFalse(result["is_volume_spike"])

    def test_declining_trend(self):
        """Test detection of declining search trend."""
        trend_data = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=14),
                "search_volume": [
                    100,
                    90,
                    80,
                    70,
                    60,
                    50,
                    40,
                    35,
                    30,
                    25,
                    20,
                    15,
                    12,
                    10,
                ],
            }
        )

        result = self.analyzer.analyze_search_trend(["测试股票"], trend_data)

        self.assertLess(result["trend_direction"], 0)

    def test_spike_detection(self):
        """Test volume spike detection."""
        trend_data = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=14),
                "search_volume": [
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    100,
                ],
            }
        )

        result = self.analyzer.analyze_search_trend(["测试股票"], trend_data)

        self.assertTrue(result["is_volume_spike"])
        self.assertGreater(result["spike_ratio"], 2.0)

    def test_empty_data(self):
        """Test empty data returns defaults."""
        result = self.analyzer.analyze_search_trend([], pd.DataFrame())
        self.assertEqual(result["trend_direction"], 0.0)


class TestCombinedSentiment(unittest.TestCase):
    """Test combined sentiment scoring."""

    def setUp(self):
        self.combiner = CombinedSentiment()

    def test_strong_bullish(self):
        """Test strong bullish combination."""
        result = self.combiner.combine(
            news_score=0.7,
            news_confidence=0.8,
            social_score=0.6,
            social_confidence=0.7,
            technical_score=0.5,
        )

        self.assertGreater(result["combined_score"], 0.3)
        self.assertEqual(result["label"], "strong_bullish")

    def test_strong_bearish(self):
        """Test strong bearish combination."""
        result = self.combiner.combine(
            news_score=-0.7,
            news_confidence=0.8,
            social_score=-0.6,
            social_confidence=0.7,
            technical_score=-0.5,
        )

        self.assertLess(result["combined_score"], -0.3)
        self.assertEqual(result["label"], "strong_bearish")

    def test_neutral(self):
        """Test neutral combination."""
        result = self.combiner.combine(
            news_score=0.05,
            news_confidence=0.5,
            social_score=-0.05,
            social_confidence=0.5,
            technical_score=0.0,
        )

        self.assertGreater(result["combined_score"], -0.1)
        self.assertLess(result["combined_score"], 0.1)
        self.assertEqual(result["label"], "neutral")

    def test_confidence_weighting(self):
        """Test that low confidence reduces impact."""
        # High score but low confidence
        result_low_conf = self.combiner.combine(
            news_score=0.9,
            news_confidence=0.1,
        )

        # High score with high confidence
        result_high_conf = self.combiner.combine(
            news_score=0.9,
            news_confidence=0.9,
        )

        self.assertLess(
            abs(result_low_conf["combined_score"]),
            abs(result_high_conf["combined_score"]),
        )

    def test_consensus_score(self):
        """Test consensus calculation."""
        # All sources agree
        result_agree = self.combiner.combine(
            news_score=0.5,
            news_confidence=0.8,
            social_score=0.5,
            social_confidence=0.8,
            technical_score=0.5,
        )

        # Sources disagree
        result_disagree = self.combiner.combine(
            news_score=0.8,
            news_confidence=0.8,
            social_score=-0.8,
            social_confidence=0.8,
            technical_score=0.0,
        )

        self.assertGreater(result_agree["consensus"], result_disagree["consensus"])

    def test_breakdown(self):
        """Test result breakdown."""
        result = self.combiner.combine(
            news_score=0.5,
            news_confidence=0.7,
            social_score=-0.3,
            social_confidence=0.6,
        )

        self.assertIn("news", result["breakdown"])
        self.assertIn("social", result["breakdown"])
        self.assertEqual(result["breakdown"]["news"]["score"], 0.5)
        self.assertEqual(result["breakdown"]["social"]["score"], -0.3)


if __name__ == "__main__":
    unittest.main()
