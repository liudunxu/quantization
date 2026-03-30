"""Unit tests for feature engineering."""

import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTechnicalFeatures(unittest.TestCase):
    """Test technical feature extraction."""

    def setUp(self):
        # Create sample OHLCV data
        dates = pd.date_range("2024-01-01", periods=100)
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100) * 2)

        self.df = pd.DataFrame(
            {
                "date": dates,
                "close": prices,
                "open": prices * 0.99,
                "high": prices * 1.02,
                "low": prices * 0.98,
                "volume": np.random.randint(1000000, 10000000, 100),
            }
        )

    def test_ma_calculation(self):
        """Test moving average calculation."""
        df = self.df.copy()
        for period in [5, 10, 20, 60]:
            df[f"ma_{period}"] = df["close"].rolling(window=period).mean()

        self.assertIn("ma_5", df.columns)
        self.assertIn("ma_10", df.columns)
        self.assertIn("ma_20", df.columns)
        self.assertIn("ma_60", df.columns)

    def test_rsi_calculation(self):
        """Test RSI calculation."""
        df = self.df.copy()
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # RSI should be between 0 and 100
        rsi_values = rsi.dropna()
        self.assertTrue((rsi_values >= 0).all())
        self.assertTrue((rsi_values <= 100).all())

    def test_macd_calculation(self):
        """Test MACD calculation."""
        df = self.df.copy()
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal

        self.assertEqual(len(macd), len(df))
        self.assertEqual(len(signal), len(df))
        self.assertEqual(len(hist), len(df))

    def test_bollinger_calculation(self):
        """Test Bollinger Bands calculation."""
        df = self.df.copy()
        period = 20
        std = 2
        df["bb_middle"] = df["close"].rolling(window=period).mean()
        df["bb_std"] = df["close"].rolling(window=period).std()
        df["bb_upper"] = df["bb_middle"] + (df["bb_std"] * std)
        df["bb_lower"] = df["bb_middle"] - (df["bb_std"] * std)

        self.assertIn("bb_upper", df.columns)
        self.assertIn("bb_middle", df.columns)
        self.assertIn("bb_lower", df.columns)

        # Upper should be greater than lower
        valid_idx = ~(df["bb_upper"].isna() | df["bb_lower"].isna())
        self.assertTrue(
            (df.loc[valid_idx, "bb_upper"] >= df.loc[valid_idx, "bb_lower"]).all()
        )

    def test_atr_calculation(self):
        """Test ATR calculation."""
        df = self.df.copy()
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()

        # ATR should be positive
        atr_values = atr.dropna()
        self.assertTrue((atr_values > 0).all())


class TestFeatureTypes(unittest.TestCase):
    """Test feature type identification."""

    def test_feature_type_technical(self):
        """Test technical feature type."""
        # Just verify we can import and check feature types
        from src.features.technical import TechnicalFeatures

        self.assertEqual(TechnicalFeatures(None).feature_type, "technical")

    def test_feature_type_fundamental(self):
        """Test fundamental feature type."""
        from src.features.fundamental import FundamentalFeatures

        self.assertEqual(FundamentalFeatures(None).feature_type, "fundamental")

    def test_feature_type_market(self):
        """Test market feature type."""
        from src.features.market import MarketFeatures

        self.assertEqual(MarketFeatures(None).feature_type, "market")


if __name__ == "__main__":
    unittest.main()
