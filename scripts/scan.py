#!/usr/bin/env python
"""股票扫描器 - 根据预测信号和技术指标筛选符合条件的股票

扫描 watchlist (stocks.txt) 中的股票，按照预测置信度、方向、
技术信号强度等条件进行筛选，输出可操作的交易信号列表。

适合中线交易风格：每日盘前/盘后运行一次。

Usage:
    python scripts/scan.py                            # 默认扫描全部 (fast_mode)
    python scripts/scan.py --zone cn                 # 只扫描 A 股
    python scripts/scan.py --zone hk                 # 只扫描港股
    python scripts/scan.py --zone us                 # 只扫描美股
    python scripts/scan.py --direction up            # 只看看多信号
    python scripts/scan.py --min-confidence 0.70     # 最低置信度 70%
    python scripts/scan.py --top 10                  # 只显示前 10 只
    python scripts/scan.py --output json              # JSON 格式输出
    python scripts/scan.py --watchlist my_stocks.txt  # 自定义 watchlist
"""

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.notification.notifier import Notifier
from src.utils.stock_info import STOCK_NAMES, ZONE_SUFFIX

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_WATCHLIST = "stocks.txt"


def load_watchlist(path: str = DEFAULT_WATCHLIST) -> List[str]:
    """Load stock codes from watchlist file."""
    watchlist_path = project_root / path
    if not watchlist_path.exists():
        logger.warning(f"Watchlist not found: {watchlist_path}")
        return []
    codes = []
    with open(watchlist_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                codes.append(line.upper())
    return codes


def filter_by_zone(codes: List[str], zone: Optional[str] = None) -> List[str]:
    """Filter stock codes by market zone."""
    if zone is None:
        return codes
    suffixes = ZONE_SUFFIX.get(zone, [])
    filtered = []
    for code in codes:
        if zone == "us":
            if "." not in code:
                filtered.append(code)
        else:
            if any(code.endswith(s) for s in suffixes):
                filtered.append(code)
    return filtered


def scan_stocks(
    codes: List[str],
    fast_mode: bool = True,
    min_confidence: float = 0.65,
    direction: Optional[str] = None,
) -> List[Dict]:
    """Run prediction on each stock and collect results.

    Args:
        codes: List of stock codes to scan
        fast_mode: Use fast mode (skip training, use cached models)
        min_confidence: Minimum confidence threshold
        direction: Filter by direction ('up', 'down', or None for all)

    Returns:
        List of scan result dicts sorted by confidence descending
    """
    from scripts.predict import run_prediction

    results = []
    total = len(codes)

    for i, code in enumerate(codes, 1):
        name = STOCK_NAMES.get(code, code)
        logger.info(f"Scanning [{i}/{total}] {code} {name}")

        try:
            pred = run_prediction(
                code=code,
                fast_mode=fast_mode,
                skip_training=True,
                skip_eval=True,
                skip_realtime=True,
            )

            if "error" in pred:
                logger.warning(f"  {code}: {pred['error']}")
                continue

            confidence = pred.get("confidence", 0)
            pred_direction = pred.get("direction", "NEUTRAL")

            if confidence < min_confidence:
                continue

            if direction and pred_direction.upper() != direction.upper():
                continue

            result = {
                "stock_code": code,
                "stock_name": name,
                "market": pred.get("market", ""),
                "direction": pred_direction,
                "confidence": confidence,
                "current_price": pred.get("current_price", 0),
                "prediction_date": pred.get("prediction_date", ""),
                "target_date": pred.get("target_date", ""),
                "ml_prob_up": pred.get("ml_prob_up", 0.5),
                "technical_signal": pred.get("technical_signal", "NEUTRAL"),
                "momentum_signal": pred.get("momentum_signal", "NEUTRAL"),
                "bullish_factors": pred.get("bullish_factors", []),
                "bearish_factors": pred.get("bearish_factors", []),
            }
            results.append(result)

        except Exception as e:
            logger.warning(f"  {code} failed: {e}")
            continue

    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results


def format_scan_report(results: List[Dict], scan_date: str = "") -> str:
    """Format scan results as a readable report."""
    if not results:
        return f"扫描日期: {scan_date}\n未发现符合条件的股票"

    lines = [
        f"扫描日期: {scan_date or date.today().isoformat()}",
        f"扫描结果: {len(results)} 只股票符合条件",
        "",
        "=" * 90,
        f"{'#':>3}  {'代码':<13} {'名称':<12} {'方向':<6} {'置信度':>7} {'价格':>9}  {'技术':<6} {'动量':<6}",
        "-" * 90,
    ]

    for i, r in enumerate(results, 1):
        direction = r["direction"]
        arrow = {"UP": "↑", "DOWN": "↓", "NEUTRAL": "→"}.get(direction, "?")
        lines.append(
            f"{i:3d}  {r['stock_code']:<13} {r['stock_name']:<10} "
            f" {arrow} {direction:<4} {r['confidence']:>6.1%} "
            f"{r['current_price']:>9.2f}  "
            f"{r.get('technical_signal', '?'):<6} "
            f"{r.get('momentum_signal', '?'):<6}"
        )

    lines.append("=" * 90)

    up_count = sum(1 for r in results if r["direction"] == "UP")
    down_count = sum(1 for r in results if r["direction"] == "DOWN")
    neutral_count = len(results) - up_count - down_count
    lines.append(f"\n汇总: ↑看多 {up_count}  ↓看空 {down_count}  →中性 {neutral_count}")

    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="股票扫描器 - 扫描 watchlist 中的股票信号",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/scan.py                         # 默认扫描全部 (fast_mode)
  python scripts/scan.py --zone cn               # 只扫描 A 股
  python scripts/scan.py --direction up           # 只看看多信号
  python scripts/scan.py --min-confidence 0.70   # 最低置信度 70%
  python scripts/scan.py --top 10                # 只显示前 10 只
  python scripts/scan.py --output json           # JSON 格式输出
        """,
    )
    parser.add_argument("--zone", type=str, choices=["cn", "hk", "us"], help="只扫描指定市场")
    parser.add_argument("--direction", type=str, choices=["up", "down"], help="只看指定方向信号")
    parser.add_argument("--min-confidence", type=float, default=0.65, help="最低置信度阈值 (默认: 0.65)")
    parser.add_argument("--top", type=int, default=20, help="最多显示几只股票 (默认: 20)")
    parser.add_argument("--output", choices=["text", "json"], default="text", help="输出格式 (默认: text)")
    parser.add_argument("--watchlist", type=str, default=DEFAULT_WATCHLIST, help="Watchlist 文件路径")
    parser.add_argument("--slow", action="store_true", help="慢速模式 (重新训练模型)")
    parser.add_argument("--notify", action="store_true", help="发送通知")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    codes = load_watchlist(args.watchlist)
    if not codes:
        print("Error: No stock codes found in watchlist")
        sys.exit(1)

    if args.zone:
        codes = filter_by_zone(codes, args.zone)
        print(f"\n  Zone filter: {args.zone} -> {len(codes)} stocks")

    print()
    print("=" * 60)
    print(f"  STOCK SCANNER - {date.today().isoformat()}")
    print(f"  Scanning {len(codes)} stocks...")
    print("=" * 60)

    results = scan_stocks(
        codes=codes,
        fast_mode=not args.slow,
        min_confidence=args.min_confidence,
        direction=args.direction,
    )

    results = results[:args.top]

    if args.output == "json":
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        report = format_scan_report(results)
        print(report)

    if args.notify and results:
        notifier = Notifier(console=True, file=True)
        notifier.send_scan_result(
            scan_type=f"{'_' + args.zone if args.zone else ''} scan",
            results=results,
            limit=args.top,
        )

    return results


if __name__ == "__main__":
    main()

