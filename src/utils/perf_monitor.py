"""Performance monitoring utilities for timing and memory tracking."""

import logging
import time
import functools
from typing import Optional, Callable, Any, Dict
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class Timer:
    """Context manager for timing code blocks.

    Usage:
        with Timer("my_operation"):
            do_something()
    """

    def __init__(
        self, operation: str, logger_instance: Optional[logging.Logger] = None
    ):
        self.operation = operation
        self.logger = logger_instance or logger
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.elapsed = time.perf_counter() - self.start
        self.logger.info(f"[PERF] {self.operation} completed in {self.elapsed:.3f}s")


def timing_decorator(operation: Optional[str] = None) -> Callable:
    """Decorator to time function execution.

    Usage:
        @timing_decorator("train_model")
        def train_model():
            ...
    """

    def decorator(func: Callable) -> Callable:
        op_name = operation or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.info(f"[PERF] {op_name} completed in {elapsed:.3f}s")
            return result

        return wrapper

    return decorator


@contextmanager
def perf_timer(operation: str):
    """Context manager generator for timing code blocks.

    Usage:
        with perf_timer("data_loading") as timer:
            load_data()
        print(f"Elapsed: {timer.elapsed:.3f}s")
    """
    start = time.perf_counter()
    timer = Timer(operation)
    timer.start = start
    yield timer
    timer.elapsed = time.perf_counter() - start
    logger.info(f"[PERF] {operation} completed in {timer.elapsed:.3f}s")


class PerformanceTracker:
    """Track performance metrics across multiple operations."""

    def __init__(self):
        self._metrics: Dict[str, Dict[str, float]] = {}

    def record(self, operation: str, elapsed: float) -> None:
        """Record a performance measurement.

        Args:
            operation: Operation name
            elapsed: Elapsed time in seconds
        """
        if operation not in self._metrics:
            self._metrics[operation] = {
                "count": 0,
                "total": 0.0,
                "min": float("inf"),
                "max": 0.0,
            }

        metrics = self._metrics[operation]
        metrics["count"] += 1
        metrics["total"] += elapsed
        metrics["min"] = min(metrics["min"], elapsed)
        metrics["max"] = max(metrics["max"], elapsed)

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get performance summary with averages.

        Returns:
            Dictionary with operation metrics including average
        """
        summary = {}
        for op, metrics in self._metrics.items():
            summary[op] = {
                **metrics,
                "avg": metrics["total"] / metrics["count"]
                if metrics["count"] > 0
                else 0.0,
            }
        return summary

    def log_summary(self, logger_instance: Optional[logging.Logger] = None) -> None:
        """Log performance summary.

        Args:
            logger_instance: Optional logger to use
        """
        log = logger_instance or logger
        summary = self.get_summary()
        log.info("[PERF SUMMARY] Performance metrics:")
        for op, metrics in summary.items():
            log.info(
                f"  {op}: avg={metrics['avg']:.3f}s, "
                f"min={metrics['min']:.3f}s, max={metrics['max']:.3f}s, "
                f"count={metrics['count']}"
            )

    def reset(self) -> None:
        """Reset all metrics."""
        self._metrics.clear()


# Global performance tracker instance
_tracker = PerformanceTracker()


def get_tracker() -> PerformanceTracker:
    """Get global performance tracker instance."""
    return _tracker
