"""Unified notification system.

Supports console output, file logging, and webhook notifications.
Designed for mid-term traders who need alerts when signals trigger.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_LOG_DIR = "logs/notifications"


class Notifier:
    """Multi-channel notification system.

    Channels:
    - console: Print to stdout
    - file: Write to daily log file
    - webhook: POST JSON to HTTP endpoint (e.g., ntfy.sh, Bark, DingTalk)
    """

    def __init__(
        self,
        console: bool = True,
        file: bool = True,
        webhook_url: Optional[str] = None,
        log_dir: str = DEFAULT_LOG_DIR,
    ):
        self.console = console
        self.file = file
        self.webhook_url = webhook_url or os.environ.get("NOTIFY_WEBHOOK_URL")
        self.log_dir = Path(log_dir)
        if self.file:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def send(
        self,
        title: str,
        message: str,
        level: str = "info",
        stock_code: str = "",
        data: Optional[Dict] = None,
    ):
        """Send notification to all enabled channels.

        Args:
            title: Short title (e.g., "BUY Signal: 0700.HK")
            message: Detailed message
            level: info, warning, danger, success
            stock_code: Optional stock code for categorization
            data: Optional structured data dict
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        icon = {"info": "ℹ️", "warning": "⚠️", "danger": "🔴", "success": "✅"}.get(level, "ℹ️")

        if self.console:
            self._console_send(icon, title, message, timestamp, level)

        if self.file:
            self._file_send(title, message, timestamp, level, stock_code, data)

        if self.webhook_url:
            self._webhook_send(title, message, level, stock_code, data)

    def send_scan_result(
        self,
        scan_type: str,
        results: List[Dict],
        limit: int = 20,
    ):
        """Send scan results summary."""
        if not results:
            self.send(
                title=f"扫描完成: {scan_type}",
                message="未发现符合条件的股票",
                level="info",
            )
            return

        lines = [f"发现 {len(results)} 只符合条件的股票 (显示前 {min(limit, len(results))} 只):\n"]
        for i, r in enumerate(results[:limit], 1):
            code = r.get("stock_code", "?")
            name = r.get("stock_name", code)
            direction = r.get("direction", "NEUTRAL")
            confidence = r.get("confidence", 0)
            price = r.get("current_price", 0)
            arrow = {"UP": "↑", "DOWN": "↓", "NEUTRAL": "→"}.get(direction, "?")
            lines.append(f"  {i:2d}. {code:<12} {name:<12} {arrow} {direction} ({confidence:.0%}) ¥{price:.2f}")

        self.send(
            title=f"扫描结果: {scan_type} - {len(results)} 只",
            message="\n".join(lines),
            level="success" if len(results) > 0 else "info",
        )

    def send_trade_signal(
        self,
        stock_code: str,
        stock_name: str,
        direction: str,
        confidence: float,
        price: float,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
        bullish_factors: Optional[List[str]] = None,
        bearish_factors: Optional[List[str]] = None,
    ):
        """Send a trade signal notification."""
        arrow = {"UP": "🟢 买入", "DOWN": "🔴 卖出", "NEUTRAL": "⚪ 观望"}.get(direction, "⚪")
        lines = [
            f"{arrow} {direction} {confidence:.0%}",
            f"  股票: {stock_code} {stock_name}",
            f"  价格: {price:.2f}",
        ]
        if stop_loss > 0:
            lines.append(f"  止损: {stop_loss:.2f} ({(stop_loss/price-1)*100:+.1f}%)")
        if take_profit > 0:
            lines.append(f"  止盈: {take_profit:.2f} ({(take_profit/price-1)*100:+.1f}%)")

        level = "success" if direction == "UP" else ("danger" if direction == "DOWN" else "info")

        self.send(
            title=f"{direction} Signal: {stock_code} {stock_name}",
            message="\n".join(lines),
            level=level,
            stock_code=stock_code,
        )

    def send_risk_alert(
        self,
        stock_code: str,
        stock_name: str,
        alert_type: str,
        current_price: float,
        stop_loss: float,
        detail: str = "",
    ):
        """Send a risk alert (stop-loss hit, take-profit hit, etc.)."""
        lines = [
            f"⚠️ {alert_type}",
            f"  股票: {stock_code} {stock_name}",
            f"  当前价格: {current_price:.2f}",
            f"  止损价位: {stop_loss:.2f}",
        ]
        if detail:
            lines.append(f"  详情: {detail}")

        self.send(
            title=f"Risk Alert: {stock_code} - {alert_type}",
            message="\n".join(lines),
            level="danger",
            stock_code=stock_code,
        )

    def _console_send(self, icon: str, title: str, message: str, timestamp: str, level: str):
        color = {"info": "\033[36m", "warning": "\033[33m", "danger": "\033[31m", "success": "\033[32m"}.get(level, "")
        reset = "\033[0m"
        print(f"{color}{icon} [{timestamp}] {title}{reset}")
        for line in message.split("\n"):
            print(f"  {line}")

    def _file_send(self, title: str, message: str, timestamp: str, level: str, stock_code: str, data: Optional[Dict]):
        date_str = datetime.now().strftime("%Y-%m-%d")
        log_file = self.log_dir / f"{date_str}.log"
        entry = {
            "timestamp": timestamp,
            "level": level,
            "title": title,
            "message": message,
            "stock_code": stock_code,
        }
        if data:
            entry["data"] = data
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write notification log: {e}")

    def _webhook_send(self, title: str, message: str, level: str, stock_code: str, data: Optional[Dict]):
        payload = {
            "title": title,
            "message": message,
            "level": level,
            "stock_code": stock_code,
            "timestamp": datetime.now().isoformat(),
        }
        if data:
            payload["data"] = data
        try:
            resp = requests.post(
                self.webhook_url,
                json=payload,
                timeout=5,
            )
            if resp.status_code >= 300:
                logger.warning(f"Webhook returned {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            logger.warning(f"Webhook notification failed: {e}")


_notifier: Optional[Notifier] = None


def get_notifier(
    console: bool = True,
    file: bool = True,
    webhook_url: Optional[str] = None,
    log_dir: str = DEFAULT_LOG_DIR,
) -> Notifier:
    """Get singleton Notifier instance."""
    global _notifier
    if _notifier is None:
        _notifier = Notifier(console=console, file=file, webhook_url=webhook_url, log_dir=log_dir)
    return _notifier

