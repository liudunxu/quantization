"""Notification module - unified notification to console, file, and webhooks."""

from src.notification.notifier import Notifier, get_notifier

__all__ = ["Notifier", "get_notifier"]

