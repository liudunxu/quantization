"""Unit tests for the stock trading system."""

import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.cache import FeatureCache
from src.utils.stock_info import StockInfoResolver, StockInfo
from src.backtest.engine import (
    BacktestEngine,
    BuyAndHoldStrategy,
    HighSellLowBuyStrategy,
    Strategy
)


class TestCache(unittest.TestCase):
    """Test cache functionality."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.cache = FeatureCache(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cache_set_and_get(self):
        """Test basic cache set and get."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        self.cache.set('TEST.SZ', 'technical', df)

        result = self.cache.get('TEST.SZ', 'technical')
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)

    def test_cache_delete(self):
        """Test cache deletion."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        self.cache.set('TEST.SZ', 'technical', df)

        deleted = self.cache.delete('TEST.SZ')
        self.assertIn('technical', deleted)

        result = self.cache.get('TEST.SZ', 'technical')
        self.assertIsNone(result)

    def test_cache_clear_all(self):
        """Test clearing all cache."""
        self.cache.set('TEST1.SZ', 'technical', pd.DataFrame({'a': [1]}))
        self.cache.set('TEST2.SZ', 'technical', pd.DataFrame({'a': [2]}))

        self.cache.clear_all()
        self.assertEqual(len(self.cache.manifest), 0)

    def test_get_cache_info(self):
        """Test getting cache info for a stock."""
        self.cache.set('TEST.SZ', 'technical', pd.DataFrame({'a': [1]}))
        self.cache.set('TEST.SZ', 'fundamental', pd.DataFrame({'b': [2]}))

        info = self.cache.get_cache_info('TEST.SZ')
        self.assertEqual(info['stock_code'], 'TEST.SZ')
        self.assertEqual(info['count'], 2)
        self.assertIn('technical', info['cached_types'])


class TestStockInfoResolver(unittest.TestCase):
    """Test stock info resolution."""

    def test_resolve_a_share_sz(self):
        """Test resolving A-share SZ."""
        info = StockInfoResolver.resolve('000001.SZ')
        self.assertEqual(info.market, 'a_share')
        self.assertEqual(info.exchange, 'SZ')
        self.assertEqual(info.symbol, '000001')

    def test_resolve_a_share_sh(self):
        """Test resolving A-share SH."""
        info = StockInfoResolver.resolve('600000.SH')
        self.assertEqual(info.market, 'a_share')
        self.assertEqual(info.exchange, 'SH')
        self.assertEqual(info.symbol, '600000')

    def test_resolve_hk_share(self):
        """Test resolving HK share."""
        info = StockInfoResolver.resolve('0700.HK')
        self.assertEqual(info.market, 'hk_share')
        self.assertEqual(info.exchange, 'HK')
        self.assertEqual(info.symbol, '0700')

    def test_resolve_us_share(self):
        """Test resolving US share."""
        info = StockInfoResolver.resolve('AAPL')
        self.assertEqual(info.market, 'us_share')
        self.assertEqual(info.symbol, 'AAPL')

    def test_get_index_code(self):
        """Test getting index code."""
        a_share = StockInfoResolver.resolve('000001.SZ')
        self.assertEqual(StockInfoResolver.get_index_code(a_share), '000001.SH')

        hk_share = StockInfoResolver.resolve('0700.HK')
        self.assertEqual(StockInfoResolver.get_index_code(hk_share), 'HSI.HK')

        us_share = StockInfoResolver.resolve('AAPL')
        self.assertEqual(StockInfoResolver.get_index_code(us_share), '^GSPC')


class TestStrategies(unittest.TestCase):
    """Test trading strategies."""

    def setUp(self):
        self.engine = BacktestEngine(initial_cash=100000)

        # Create sample data
        dates = pd.date_range('2024-01-01', periods=100)
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100) * 2)

        self.df = pd.DataFrame({
            'date': dates,
            'close': prices,
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'volume': np.random.randint(1000000, 10000000, 100)
        })

    def test_buy_and_hold(self):
        """Test buy and hold strategy."""
        strategy = BuyAndHoldStrategy()
        result = self.engine.run(self.df, strategy)

        self.assertEqual(result.total_trades, 1)  # One buy
        self.assertTrue(result.total_return > 0)  # Should make some return

    def test_high_sell_low_buy(self):
        """Test high sell low buy strategy."""
        strategy = HighSellLowBuyStrategy(lookback=10, threshold=0.1)
        result = self.engine.run(self.df, strategy)

        # Should have multiple trades
        self.assertGreater(result.total_trades, 1)

    def test_compare_strategies(self):
        """Test comparing multiple strategies."""
        strategies = [
            BuyAndHoldStrategy(),
            HighSellLowBuyStrategy()
        ]

        results_df = self.engine.compare_strategies(self.df, strategies)

        self.assertEqual(len(results_df), 2)
        self.assertIn('Strategy', results_df.columns)
        self.assertIn('Total Return', results_df.columns)


if __name__ == '__main__':
    unittest.main()
