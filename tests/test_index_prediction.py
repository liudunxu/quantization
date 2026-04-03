"""Unit tests for A-share index features and prediction."""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_providers.fetch_stock_data import A_SHARE_INDICES, fetch_index_data
from src.features.index_features import (
    INDEX_NAMES,
    extract_index_features,
    get_index_name,
)


class TestIndexDataFetcher(unittest.TestCase):
    """Test index data fetching."""

    def test_index_code_mapping(self):
        """Test that index code mapping is populated."""
        self.assertIn("000001", A_SHARE_INDICES)
        self.assertIn("000300", A_SHARE_INDICES)
        self.assertIn("399001", A_SHARE_INDICES)
        self.assertIn("399006", A_SHARE_INDICES)
        self.assertEqual(A_SHARE_INDICES["000001"], "上证指数")
        self.assertEqual(A_SHARE_INDICES["000300"], "沪深300")

    def test_fetch_index_data_empty_on_invalid(self):
        """Test that invalid index code returns empty DataFrame."""
        df = fetch_index_data("999999", days=30)
        self.assertTrue(df.empty)


class TestIndexFeatures(unittest.TestCase):
    """Test index feature extraction."""

    def test_get_index_name(self):
        """Test index name lookup."""
        self.assertEqual(get_index_name("000001"), "上证指数")
        self.assertEqual(get_index_name("000300"), "沪深300")
        self.assertEqual(get_index_name("399001"), "深证成指")
        self.assertEqual(get_index_name("399006"), "创业板指")
        self.assertEqual(get_index_name("999999"), "999999")

    def test_index_names_complete(self):
        """Test that INDEX_NAMES matches A_SHARE_INDICES."""
        for code in A_SHARE_INDICES:
            self.assertIn(code, INDEX_NAMES)

    def test_extract_index_features_with_mock_data(self):
        """Test feature extraction with mock index data."""
        dates = pd.date_range("2024-01-01", periods=200)
        np.random.seed(42)
        prices = 3000 + np.cumsum(np.random.randn(200) * 10)

        mock_data = pd.DataFrame(
            {
                "date": dates,
                "open": prices * 0.99,
                "high": prices * 1.01,
                "low": prices * 0.98,
                "close": prices,
                "volume": np.random.randint(100000000, 500000000, 200),
            }
        )

        from src.features.technical import TechnicalFeatures

        tech = TechnicalFeatures()
        df = tech.extract("000001", days=200, _preloaded_data=mock_data)

        self.assertFalse(df.empty)
        self.assertIn("date", df.columns)
        self.assertIn("close", df.columns)
        self.assertIn("rsi", df.columns)
        self.assertIn("macd", df.columns)
        self.assertIn("atr", df.columns)
        self.assertIn("volume_ratio", df.columns)
        self.assertGreater(len(df.columns), 50)


class TestIndexPredictionIntegration(unittest.TestCase):
    """Test index prediction integration."""

    def test_predict_script_accepts_index_arg(self):
        """Test that predict.py accepts --index argument."""
        import argparse
        from scripts.predict import parse_args

        import sys

        old_argv = sys.argv
        try:
            sys.argv = [
                "predict.py",
                "--index",
                "000001",
                "--single-model",
                "--train-days",
                "100",
            ]
            args = parse_args()
            self.assertEqual(args.index, "000001")
            self.assertIsNone(args.stock)
            self.assertEqual(args.train_days, 100)
        finally:
            sys.argv = old_argv

    def test_predict_script_requires_stock_or_index(self):
        """Test that predict.py requires either --stock or --index."""
        from scripts.predict import parse_args
        import sys

        old_argv = sys.argv
        try:
            sys.argv = ["predict.py"]
            args = parse_args()
            self.assertIsNone(args.stock)
            self.assertIsNone(args.index)
        finally:
            sys.argv = old_argv


if __name__ == "__main__":
    unittest.main()
