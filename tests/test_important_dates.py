"""Unit tests for important dates manager and performance monitoring."""

import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import time
from datetime import datetime

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.important_dates import ImportantDatesManager
from src.utils.perf_monitor import (
    Timer,
    timing_decorator,
    PerformanceTracker,
    get_tracker,
    perf_timer,
)


class TestImportantDatesManager(unittest.TestCase):
    """Test important dates management."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = str(Path(self.temp_dir) / "test_important_dates.db")
        self.manager = ImportantDatesManager(self.db_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_add_and_get_date(self):
        """Test adding and retrieving important dates."""
        self.manager.add_date(
            date="2024-01-15",
            market="a_share",
            event_type="policy",
            description="Test policy event",
            impact_level="high",
            source="test",
        )

        dates = self.manager.get_dates(market="a_share")
        self.assertFalse(dates.empty)
        self.assertEqual(len(dates), 1)
        self.assertEqual(dates.iloc[0]["event_type"], "policy")

    def test_get_dates_as_list(self):
        """Test getting dates as list."""
        self.manager.add_date("2024-01-15", "a_share", "policy")
        self.manager.add_date("2024-02-20", "a_share", "crisis")

        dates = self.manager.get_dates_as_list(market="a_share")
        self.assertEqual(len(dates), 2)
        self.assertIn("2024-01-15", dates)
        self.assertIn("2024-02-20", dates)

    def test_delete_date(self):
        """Test deleting important dates."""
        self.manager.add_date("2024-01-15", "a_share", "policy")
        self.manager.delete_date("2024-01-15", "a_share", "policy")

        dates = self.manager.get_dates(market="a_share")
        self.assertTrue(dates.empty)

    def test_fetch_from_web(self):
        """Test fetching predefined events."""
        count = self.manager.fetch_from_web(market="a_share", years=2)
        self.assertGreater(count, 0)

        dates = self.manager.get_dates(market="a_share")
        self.assertFalse(dates.empty)

    def test_search_web_events(self):
        """Test web event search functionality."""
        count = self.manager.search_web_events(
            market="us",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )
        # Should find FOMC meetings, NFP dates, earnings seasons
        self.assertGreater(count, 0)

    def test_detect_high_volatility_dates(self):
        """Test detecting high volatility dates from price data."""
        dates = pd.date_range("2024-01-01", periods=100)
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100) * 2)

        # Inject an extreme day
        prices[50] = prices[49] * 1.10  # 10% jump

        df = pd.DataFrame(
            {
                "date": dates,
                "close": prices,
                "open": prices * 0.99,
                "high": prices * 1.02,
                "low": prices * 0.98,
            }
        )

        volatile_dates = self.manager.detect_high_volatility_dates(df, market="a_share")
        self.assertIsInstance(volatile_dates, list)

    def test_get_or_detect_dates(self):
        """Test get or detect with auto-detection."""
        dates = pd.date_range("2024-01-01", periods=50)
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(50) * 2)

        df = pd.DataFrame(
            {
                "date": dates,
                "close": prices,
                "open": prices * 0.99,
                "high": prices * 1.02,
                "low": prices * 0.98,
            }
        )

        result = self.manager.get_or_detect_dates(
            df, market="a_share", auto_detect=True
        )
        self.assertIsInstance(result, list)

    def test_date_filtering(self):
        """Test date range filtering."""
        self.manager.add_date("2024-01-15", "a_share", "policy")
        self.manager.add_date("2024-06-15", "a_share", "crisis")
        self.manager.add_date("2025-01-15", "a_share", "policy")

        dates = self.manager.get_dates(
            market="a_share",
            start_date="2024-01-01",
            end_date="2024-12-31",
        )
        self.assertEqual(len(dates), 2)

    def test_event_type_filtering(self):
        """Test filtering by event type."""
        self.manager.add_date("2024-01-15", "a_share", "policy")
        self.manager.add_date("2024-02-20", "a_share", "crisis")

        dates = self.manager.get_dates(market="a_share", event_type="policy")
        self.assertEqual(len(dates), 1)
        self.assertEqual(dates.iloc[0]["event_type"], "policy")


class TestPerformanceMonitor(unittest.TestCase):
    """Test performance monitoring utilities."""

    def test_timer_context_manager(self):
        """Test Timer context manager."""
        with Timer("test_operation") as timer:
            time.sleep(0.05)

        self.assertGreater(timer.elapsed, 0.04)
        self.assertLess(timer.elapsed, 0.2)

    def test_timing_decorator(self):
        """Test timing decorator."""

        @timing_decorator("test_func")
        def slow_function():
            time.sleep(0.05)
            return 42

        result = slow_function()
        self.assertEqual(result, 42)

    def test_perf_timer_context(self):
        """Test perf_timer context manager."""
        with perf_timer("test_block") as timer:
            time.sleep(0.05)

        self.assertGreater(timer.elapsed, 0.04)

    def test_performance_tracker_record(self):
        """Test recording performance metrics."""
        tracker = PerformanceTracker()
        tracker.record("operation_a", 1.0)
        tracker.record("operation_a", 2.0)
        tracker.record("operation_b", 3.0)

        summary = tracker.get_summary()
        self.assertEqual(summary["operation_a"]["count"], 2)
        self.assertAlmostEqual(summary["operation_a"]["avg"], 1.5)
        self.assertEqual(summary["operation_a"]["min"], 1.0)
        self.assertEqual(summary["operation_a"]["max"], 2.0)
        self.assertEqual(summary["operation_b"]["count"], 1)

    def test_performance_tracker_reset(self):
        """Test resetting performance tracker."""
        tracker = PerformanceTracker()
        tracker.record("op", 1.0)
        tracker.reset()

        summary = tracker.get_summary()
        self.assertEqual(len(summary), 0)

    def test_global_tracker(self):
        """Test global tracker singleton."""
        tracker = get_tracker()
        self.assertIsInstance(tracker, PerformanceTracker)

        tracker2 = get_tracker()
        self.assertIs(tracker, tracker2)


if __name__ == "__main__":
    unittest.main()
