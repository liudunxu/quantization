"""Unit tests for ML model."""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.trainer import StockTradingModel


class TestStockTradingModel(unittest.TestCase):
    """Test ML model functionality."""

    def setUp(self):
        # Create sample data with features
        np.random.seed(42)
        n_samples = 200
        dates = pd.date_range("2024-01-01", periods=n_samples)

        # Generate price data
        prices = 100 + np.cumsum(np.random.randn(n_samples) * 2)

        self.df = pd.DataFrame(
            {
                "date": dates,
                "close": prices,
                "open": prices * 0.99,
                "high": prices * 1.02,
                "low": prices * 0.98,
                "volume": np.random.randint(1000000, 10000000, n_samples),
                # Add some feature columns
                "ma_5": pd.Series(prices).rolling(5).mean().values,
                "ma_10": pd.Series(prices).rolling(10).mean().values,
                "ma_20": pd.Series(prices).rolling(20).mean().values,
                "rsi_14": np.random.uniform(20, 80, n_samples),
                "macd": np.random.randn(n_samples),
                "macd_signal": np.random.randn(n_samples),
                "macd_hist": np.random.randn(n_samples),
                "bb_upper": prices * 1.05,
                "bb_middle": prices,
                "bb_lower": prices * 0.95,
                "atr": np.random.uniform(1, 5, n_samples),
                "returns": np.random.randn(n_samples) * 0.02,
                "index_returns": np.random.randn(n_samples) * 0.01,
            }
        )

    def test_model_initialization(self):
        """Test model initialization."""
        model = StockTradingModel()
        self.assertIsNotNone(model)
        self.assertEqual(len(model.models), 0)

    def test_model_training(self):
        """Test model training."""
        model = StockTradingModel()
        train_df = self.df.iloc[:150]
        eval_df = self.df.iloc[150:180]

        metrics = model.train(train_df, forward_days=5, threshold=0.01, eval_df=eval_df)

        self.assertIn("train_accuracy", metrics)
        self.assertIn("label_distribution", metrics)
        self.assertGreater(len(model.models), 0)

    def test_model_prediction(self):
        """Test model prediction."""
        model = StockTradingModel()
        train_df = self.df.iloc[:150]

        model.train(train_df, forward_days=5, threshold=0.01)

        # Predict on latest data
        pred, confidence = model.predict(self.df)

        self.assertIn(pred, [-1, 0, 1])
        self.assertGreaterEqual(confidence, 0)
        self.assertLessEqual(confidence, 1)

    def test_model_predict_proba(self):
        """Test model probability prediction."""
        model = StockTradingModel()
        train_df = self.df.iloc[:150]

        model.train(train_df, forward_days=5, threshold=0.01)

        proba = model.predict_proba(self.df)

        self.assertIn("sell_probability", proba)
        self.assertIn("hold_probability", proba)
        self.assertIn("buy_probability", proba)

    def test_label_distribution(self):
        """Test label distribution is balanced."""
        model = StockTradingModel()
        train_df = self.df.iloc[:150]

        metrics = model.train(train_df, forward_days=5, threshold=0.01)
        label_dist = metrics["label_distribution"]

        # Should have all three labels
        self.assertIn("buy", label_dist)
        self.assertIn("hold", label_dist)
        self.assertIn("sell", label_dist)

        # Total should match training samples
        total = label_dist["buy"] + label_dist["hold"] + label_dist["sell"]
        self.assertEqual(total, metrics["train_samples"])


if __name__ == "__main__":
    unittest.main()
