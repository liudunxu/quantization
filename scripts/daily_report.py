#!/usr/bin/env python
"""每日报告 - 盘前个股关注列表 + 持仓状态 + 风险检查

整合扫描结果和持仓信息，生成每日操作建议。
适合中线交易：每晚或盘前运行一次。

Usage:
    python scripts/daily_report.py                   # 完整盘前报告
    python scripts/daily_report.py --zone cn         # 只看 A 股
    python scripts/daily_report.py --quick            # 快速模式 (只看持仓 + 风险)
    python scripts/daily_report.py --output json      # JSON 格式输出
    python scripts/daily_report.py --notify          # 发送通知
"""

import argparse
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.notification.notifier import Notifier
from src.risk.manager import RiskManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def section_header(title: str, width: int = 70) -> str:
    return f"\n{'=' * width}\n  {title}\n{'=' * width}"


def format_watchlist_from_scan(scan_results: List[Dict]) -> str:
    """Format scan results into an actionable watchlist."""
    if not scan_results:
        return "  无符合条件的信号"

    up_signals = [r for r in scan_results if r["direction"] == "UP"]
    down_signals = [r for r in scan_results if r["direction"] == "DOWN"]

    lines = []

    if up_signals:
        lines.append("\n  📈 看多信号 (建议关注/加仓):")
        lines.append(f"  {'#':>3}  {'代码':<13} {'名称':<12} {'置信度':>7} {'当前价':>9}")
        lines.append("  " + "-" * 50)
        for i, r in enumerate(up_signals, 1):
            lines.append(
                f"  {i:3d}  {r['stock_code']:<13} {r['stock_name']:<10} "
                f"{r['confidence']:>6.1%} {r['current_price']:>9.2f}"
            )

    if down_signals:
        lines.append("\n  📉 看空信号 (建议减仓/止损):")
        lines.append(f"  {'#':>3}  {'代码':<13} {'名称':<12} {'置信度':>7} {'当前价':>9}")
        lines.append("  " + "-" * 50)
        for i, r in enumerate(down_signals, 1):
            lines.append(
                f"  {i:3d}  {r['stock_code']:<13} {r['stock_name']:<10} "
                f"{r['confidence']:>6.1%} {r['current_price']:>9.2f}"
            )

    return "\n".join(lines)


def format_portfolio_status(rm: RiskManager, total_capital: float) -> str:
    """Format current portfolio status for the report."""
    positions = rm.get_all_positions()
    summary = rm.portfolio_summary(total_capital)

    lines = [
        f"  持仓数: {summary['positions_count']}  "
        f"投入: ¥{summary['total_invested']:,.2f}  "
        f"市值: ¥{summary['total_market_value']:,.2f}  "
        f"浮动P&L: ¥{summary['total_pnl']:>+,.2f} ({summary['total_pnl_pct']})  "
        f"现金: ¥{summary['cash_available']:,.2f}",
    ]

    if not positions:
        lines.append("  当前无持仓")
        return "\n".join(lines)

    lines.append("")
    lines.append(
        f"  {'代码':<13} {'名称':<10} {'股数':>5} {'成本':>8} {'现价':>8} "
        f"{'P&L':>10} {'止损':>8} {'止盈':>8} {'持仓天数':>6}"
    )
    lines.append("  " + "-" * 85)

    for code, pos in positions.items():
        pnl = pos.profit_loss
        holding_days = (date.today() - date.fromisoformat(pos.entry_date)).days if pos.entry_date else 0
        sl_str = f"{pos.stop_loss:.2f}" if pos.stop_loss > 0 else "-"
        tp_str = f"{pos.take_profit:.2f}" if pos.take_profit > 0 else "-"
        lines.append(
            f"  {code:<13} {pos.stock_name:<10} {pos.shares:>5} "
            f"{pos.entry_price:>8.2f} {pos.current_price:>8.2f} "
            f"{pnl:>+10.2f} {sl_str:>8} {tp_str:>8} {holding_days:>5}天"
        )

    return "\n".join(lines)


def format_risk_checks(rm: RiskManager) -> str:
    """Check all positions for risk alerts."""
    positions = rm.get_all_positions()
    if not positions:
        return "  无持仓需要检查"

    alerts = []
    warnings = []

    for code, pos in positions.items():
        if pos.current_price <= 0:
            continue

        if pos.stop_loss > 0:
            gap_pct = (pos.current_price - pos.stop_loss) / pos.stop_loss
            if gap_pct < 0.03:
                alerts.append(
                    f"  🔴 {code} {pos.stock_name}: 价格接近止损! "
                    f"当前={pos.current_price:.2f} 止损={pos.stop_loss:.2f} "
                    f"距离止损仅 {gap_pct:.1%}"
                )
            elif gap_pct < 0.07:
                warnings.append(
                    f"  🟡 {code} {pos.stock_name}: 距止损 {gap_pct:.1%} "
                    f"当前={pos.current_price:.2f} 止损={pos.stop_loss:.2f}"
                )

        if pos.take_profit > 0:
            gap_pct = (pos.take_profit - pos.current_price) / pos.current_price
            if gap_pct < 0.03:
                alerts.append(
                    f"  🟢 {code} {pos.stock_name}: 接近止盈! "
                    f"当前={pos.current_price:.2f} 止盈={pos.take_profit:.2f} "
                    f"距止盈 {gap_pct:.1%}"
                )

        holding_days = 0
        if pos.entry_date:
            holding_days = (date.today() - date.fromisoformat(pos.entry_date)).days
        if holding_days > 20:
            warnings.append(
                f"  ⏰ {code} {pos.stock_name}: 已持有 {holding_days} 天，"
                f"P&L={pos.profit_loss:+.2f} ({pos.profit_loss_pct:+.1%})"
            )

    lines = []
    if alerts:
        lines.append("\n  ⚠️  风险警报:")
        lines.extend(alerts)
    if warnings:
        lines.append("\n  📋 注意事项:")
        lines.extend(warnings)
    if not alerts and not warnings:
        lines.append("  ✅ 所有持仓风险正常")

    return "\n".join(lines)


def generate_report(
    rm: RiskManager,
    total_capital: float,
    scan_results: Optional[List[Dict]] = None,
    quick: bool = False,
) -> str:
    """Generate the full daily report."""
    today = date.today().isoformat()
    weekday_cn = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][date.today().weekday()]

    sections = []
    sections.append(f"\n  📊 每日盘前报告 - {today} {weekday_cn}")
    sections.append(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Section 1: Portfolio Status
    sections.append(section_header("一、持仓总览"))
    sections.append(format_portfolio_status(rm, total_capital))

    # Section 2: Risk Checks
    sections.append(section_header("二、风险检查"))
    sections.append(format_risk_checks(rm))

    if not quick and scan_results is not None:
        # Section 3: Scan Results
        sections.append(section_header("三、扫描信号"))
        sections.append(format_watchlist_from_scan(scan_results))

        # Section 4: Action Items
        sections.append(section_header("四、操作建议"))

        positions = rm.get_all_positions()
        action_items = []

        for code, pos in positions.items():
            matching = [r for r in scan_results if r["stock_code"] == code]
            if matching:
                r = matching[0]
                if r["direction"] == "DOWN" and pos.profit_loss_pct < 0:
                    action_items.append(f"  🔻 {code} {pos.stock_name}: 看空信号+亏损，建议止损")
                elif r["direction"] == "UP" and pos.profit_loss_pct > 0.05:
                    action_items.append(f"  🟢 {code} {pos.stock_name}: 看多信号+盈利，可考虑加仓或移动止损")

        for r in scan_results[:5]:
            code = r["stock_code"]
            if code not in positions and r["confidence"] >= 0.70:
                action_items.append(f"  🆕 {code} {r['stock_name']}: 新信号 置信度={r['confidence']:.0%}")

        if action_items:
            sections.extend(action_items)
        else:
            sections.append("  今日无特别操作建议")

    sections.append("")
    return "\n".join(sections)


def parse_args():
    parser = argparse.ArgumentParser(
        description="每日报告 - 盘前个股关注列表 + 持仓状态 + 风险检查",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  daily_report.py                   # 完整盘前报告
  daily_report.py --zone cn         # 只看 A 股
  daily_report.py --quick           # 快速模式
  daily_report.py --output json     # JSON 格式输出
  daily_report.py --notify          # 发送通知
        """,
    )
    parser.add_argument("--zone", type=str, choices=["cn", "hk", "us"], help="只看指定市场")
    parser.add_argument("--quick", action="store_true", help="快速模式 (只看持仓+风险，不扫描)")
    parser.add_argument("--output", choices=["text", "json"], default="text", help="输出格式")
    parser.add_argument("--capital", type=float, default=1_000_000, help="总资金")
    parser.add_argument("--db", type=str, default="cache/portfolio.db", help="持仓数据库路径")
    parser.add_argument("--notify", action="store_true", help="发送通知")
    parser.add_argument("--min-confidence", type=float, default=0.65, help="扫描最低置信度")
    parser.add_argument("--top", type=int, default=15, help="扫描结果最多显示几只")
    return parser.parse_args()


def main():
    args = parse_args()
    rm = RiskManager(db_path=args.db)

    scan_results = None
    if not args.quick:
        logger.info("Running stock scan (fast mode)...")
        try:
            from scripts.scan import filter_by_zone, load_watchlist, scan_stocks
            codes = load_watchlist()
            if args.zone:
                codes = filter_by_zone(codes, args.zone)
            scan_results = scan_stocks(
                codes=codes,
                fast_mode=True,
                min_confidence=args.min_confidence,
            )
            scan_results = scan_results[:args.top]
        except Exception as e:
            logger.warning(f"Scan failed: {e}")
            scan_results = []

    report = generate_report(rm, args.capital, scan_results, quick=args.quick)

    if args.output == "json":
        data = {
            "date": date.today().isoformat(),
            "portfolio": rm.portfolio_summary(args.capital),
            "risk_alerts": format_risk_checks(rm),
            "scan_results": scan_results or [],
        }
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print(report)

    if args.notify:
        notifier = Notifier(console=True, file=True)
        summary = rm.portfolio_summary(args.capital)
        positions_count = summary["positions_count"]
        pnl = summary["total_pnl"]
        notifier.send(
            title=f"盘前报告 {date.today().isoformat()}",
            message=f"持仓{positions_count}只 浮动P&L: ¥{pnl:+,.2f}",
            level="info",
        )


if __name__ == "__main__":
    main()
