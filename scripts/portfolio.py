#!/usr/bin/env python
"""持仓管理脚本 - 跟踪持仓、计算P&L、管理止损止盈

支持中线交易风格：每日更新持仓状态、检查止损止盈触发、生成持仓报告。

Usage:
    python scripts/portfolio.py status                       # 查看持仓总览
    python scripts/portfolio.py buy 0700.HK 400 50          # 买入: 400股@50
    python scripts/portfolio.py buy 0700.HK 400 50 --sl 45 --tp 60  # 带止损止盈
    python scripts/portfolio.py sell 0700.HK 55             # 卖出: @55
    python scripts/portfolio.py update                       # 更新所有持仓当前价格
    python scripts/portfolio.py check                       # 检查止损止盈是否触发
    python scripts/portfolio.py list                        # 列出所有持仓
    python scripts/portfolio.py trades                      # 查看交易历史
    python scripts/portfolio.py summary                      # P&L 汇总
    python scripts/portfolio.py size 0700.HK 400 50 1000000  # 计算建议仓位
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.risk.manager import PositionSizer, RiskManager, StopLossCalculator
from src.utils.stock_info import STOCK_NAMES, StockInfoResolver

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def get_stock_name(code: str) -> str:
    return STOCK_NAMES.get(code.upper(), code)


def resolve_market(code: str) -> str:
    try:
        info = StockInfoResolver.resolve(code)
        return info.market.replace("_share", "")
    except ValueError:
        return "unknown"


def cmd_status(rm: RiskManager, total_capital: float):
    """Show portfolio status overview."""
    summary = rm.portfolio_summary(total_capital)
    positions = rm.get_all_positions()

    print()
    print("=" * 70)
    print(f"  PORTFOLIO STATUS - {summary['date']}")
    print("=" * 70)
    print(f"  Total Capital:    ¥{summary['total_capital']:>12,.2f}")
    print(f"  Invested:         ¥{summary['total_invested']:>12,.2f}")
    print(f"  Market Value:     ¥{summary['total_market_value']:>12,.2f}")
    print(f"  P&L:              ¥{summary['total_pnl']:>12,.2f}  ({summary['total_pnl_pct']})")
    print(f"  Cash Available:   ¥{summary['cash_available']:>12,.2f}")
    print(f"  Positions:        {summary['positions_count']}")
    print()

    if not positions:
        print("  No open positions.")
        return

    print(f"  {'Code':<13} {'Name':<10} {'Shares':>6} {'Entry':>9} {'Price':>9} {'P&L':>10} {'SL':>9} {'TP':>9} {'Date':<12}")
    print("  " + "-" * 95)
    for code, pos in positions.items():
        pnl = pos.profit_loss
        pnl_pct = f"{pos.profit_loss_pct:+.1%}"
        sl_str = f"{pos.stop_loss:.2f}" if pos.stop_loss > 0 else "-"
        tp_str = f"{pos.take_profit:.2f}" if pos.take_profit > 0 else "-"
        print(
            f"  {code:<13} {pos.stock_name:<10} {pos.shares:>6} "
            f"{pos.entry_price:>9.2f} {pos.current_price:>9.2f} "
            f"{pnl:>+9.2f}({pnl_pct:>5}) "
            f"{sl_str:>9} {tp_str:>9} {pos.entry_date}"
        )
    print()


def cmd_buy(
    rm: RiskManager,
    stock_code: str,
    shares: int,
    price: float,
    stop_loss: float = 0.0,
    take_profit: float = 0.0,
    notes: str = "",
):
    """Open a new position."""
    can_open, reason = rm.can_open_position(stock_code, total_capital=1_000_000)
    if not can_open:
        print(f"  Cannot open position: {reason}")
        return

    pos = rm.open_position(
        stock_code=stock_code.upper(),
        entry_price=price,
        shares=shares,
        stop_loss=stop_loss,
        take_profit=take_profit,
        stock_name=get_stock_name(stock_code),
        market=resolve_market(stock_code),
        notes=notes,
    )
    print(f"  Opened: {pos.stock_code} {pos.stock_name} {pos.shares}@{pos.entry_price:.2f}")
    if stop_loss > 0:
        print(f"  Stop-loss: {stop_loss:.2f} ({(stop_loss/price-1)*100:+.1f}%)")
    if take_profit > 0:
        print(f"  Take-profit: {take_profit:.2f} ({(take_profit/price-1)*100:+.1f}%)")


def cmd_sell(rm: RiskManager, stock_code: str, price: float, reason: str = ""):
    """Close a position."""
    trade = rm.close_position(stock_code.upper(), exit_price=price, reason=reason)
    if trade is None:
        print(f"  No position found for {stock_code}")
        return
    print(f"  Closed: {trade.stock_code} {trade.stock_name} {trade.shares}@{price:.2f}")


def cmd_list(rm: RiskManager):
    """List all positions."""
    positions = rm.get_all_positions()
    if not positions:
        print("\n  No open positions.\n")
        return

    print()
    print(f"  Open Positions ({len(positions)}):")
    print("-" * 60)
    for code, pos in positions.items():
        print(f"  {code:<13} {pos.stock_name:<10} {pos.shares} shares @ {pos.entry_price:.2f}")
        print(f"    Current: {pos.current_price:.2f}  SL: {pos.stop_loss:.2f}  TP: {pos.take_profit:.2f}")
        print(f"    P&L: {pos.profit_loss:+.2f} ({pos.profit_loss_pct:+.1%})  Since: {pos.entry_date}")
    print()


def cmd_trades(rm: RiskManager, stock_code: Optional[str] = None):
    """Show trade history."""
    trades = rm.get_trades(stock_code)
    if not trades:
        print("\n  No trade history.\n")
        return

    print()
    print(f"  Trade History ({len(trades)} trades):")
    print(f"  {'Date':<12} {'Dir':<5} {'Code':<13} {'Shares':>6} {'Price':>9} {'Amount':>12}")
    print("  " + "-" * 65)
    for t in trades:
        print(
            f"  {t.date:<12} {t.direction:<5} {t.stock_code:<13} "
            f"{t.shares:>6} {t.price:>9.2f} {t.amount:>12,.2f}"
        )
    print()


def cmd_check(rm: RiskManager, prices: Optional[Dict[str, float]] = None):
    """Check if any stop-loss or take-profit is triggered."""
    positions = rm.get_all_positions()
    if not positions:
        print("\n  No positions to check.\n")
        return

    alerts = []
    for code, pos in positions.items():
        current_price = prices.get(code, pos.current_price) if prices else pos.current_price
        if current_price <= 0:
            current_price = pos.entry_price

        sl_hit = rm.check_stop_loss_hit(code, current_price)
        tp_hit = rm.check_take_profit_hit(code, current_price)

        if sl_hit:
            alerts.append(("STOP-LOSS", code, pos.stock_name, current_price, sl_hit))
        if tp_hit:
            alerts.append(("TAKE-PROFIT", code, pos.stock_name, current_price, tp_hit))

    if alerts:
        print("\n  ⚠️  RISK ALERTS:")
        print("  " + "-" * 60)
        for alert_type, code, name, price, reason in alerts:
            print(f"  {alert_type}: {code} {name} @ {price:.2f} - {reason}")
        print()
    else:
        print("\n  ✓ No stop-loss or take-profit triggered.\n")


def cmd_size(
    stock_code: str,
    shares: Optional[int],
    entry_price: float,
    total_capital: float,
    stop_loss: float = 0.0,
    method: str = "risk",
):
    """Calculate position size recommendations."""
    sizer = PositionSizer()
    stop_calc = StopLossCalculator()

    if stop_loss <= 0:
        stop_loss = entry_price * 0.93

    print(f"\n  Position Size Calculator for {stock_code} ({get_stock_name(stock_code)})")
    print(f"  Entry Price: {entry_price:.2f}  Stop-Loss: {stop_loss:.2f}  Capital: {total_capital:,.0f}")
    print()

    risk_shares, risk_value = sizer.size_by_risk(total_capital, entry_price, stop_loss)
    print(f"  Risk-based sizing:       {risk_shares:>6} shares  (¥{risk_value:>12,.2f})")

    pct_shares, pct_value = sizer.size_by_portfolio_pct(total_capital, price=entry_price)
    print(f"  Portfolio % sizing (20%): {pct_shares:>6} shares  (¥{pct_value:>12,.2f})")

    atr_shares, atr_value = sizer.size_by_atr(
        total_capital, entry_price, atr=entry_price * 0.02,
    )
    print(f"  ATR-based sizing:        {atr_shares:>6} shares  (¥{atr_value:>12,.2f})")

    sl, tp_fixed = stop_calc.fixed_pct_stop(entry_price)
    sl_atr, tp_atr = stop_calc.atr_stop(entry_price, atr=entry_price * 0.02)
    print()
    print("  Suggested Stop-Loss / Take-Profit:")
    print(f"    Fixed (7%/15%):  SL={sl:.2f}  TP={tp_fixed:.2f}")
    print(f"    ATR (2x):        SL={sl_atr:.2f}  TP={tp_atr:.2f}")
    print()


def cmd_summary(rm: RiskManager, total_capital: float):
    """Show P&L summary."""
    stats = rm.trade_statistics()
    portfolio = rm.portfolio_summary(total_capital)

    print()
    print("=" * 60)
    print("  P&L SUMMARY")
    print("=" * 60)
    print(f"  Total Trades:      {stats.get('total_trades', 0)}")
    print(f"  Total Bought:      ¥{stats.get('total_bought', 0):>12,.2f}")
    print(f"  Total Sold:        ¥{stats.get('total_sold', 0):>12,.2f}")
    print(f"  Net P&L:           ¥{stats.get('net_pnl', 0):>+12,.2f}")
    print()
    print(f"  Open Positions:    {portfolio['positions_count']}")
    print(f"  Market Value:      ¥{portfolio['total_market_value']:>12,.2f}")
    print(f"  Unrealized P&L:   ¥{portfolio['total_pnl']:>+12,.2f} ({portfolio['total_pnl_pct']})")
    print(f"  Cash Available:    ¥{portfolio['cash_available']:>12,.2f}")
    print()


def cmd_update(rm: RiskManager):
    """Update current prices for all positions from prediction system."""
    positions = rm.get_all_positions()
    if not positions:
        print("\n  No positions to update.\n")
        return

    print(f"\n  Updating prices for {len(positions)} positions...")

    from scripts.predict import PredictionService
    service = PredictionService()

    prices = {}
    for code in positions:
        try:
            price = service.data_pipeline.get_realtime_price(code)
            if price and price > 0:
                prices[code] = price
                print(f"  {code}: {price:.2f}")
            else:
                print(f"  {code}: price unavailable, keeping {positions[code].current_price:.2f}")
        except Exception as e:
            print(f"  {code}: error - {e}")

    if prices:
        rm.update_prices(prices)
        print(f"\n  Updated {len(prices)} prices.\n")
    else:
        print("\n  No prices updated.\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="持仓管理 - 跟踪持仓、计算P&L、管理止损止盈",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  portfolio.py status                      # 查看持仓总览
  portfolio.py buy 0700.HK 400 50          # 买入 400股@50
  portfolio.py buy 0700.HK 400 50 --sl 45 --tp 60
  portfolio.py sell 0700.HK 55            # 卖出 @55
  portfolio.py update                      # 更新当前价格
  portfolio.py check                       # 检查止损止盈
  portfolio.py size 0700.HK - 50 1000000   # 建议仓位
        """,
    )
    sub = parser.add_subparsers(dest="command", help="Commands")

    sub.add_parser("status", help="查看持仓总览")
    sub.add_parser("list", help="列出所有持仓")
    sub.add_parser("trades", help="查看交易历史")
    sub.add_parser("update", help="更新所有持仓当前价格")
    sub.add_parser("check", help="检查止损止盈触发")
    sub.add_parser("summary", help="P&L 汇总")

    buy_p = sub.add_parser("buy", help="买入开仓")
    buy_p.add_argument("stock_code", type=str, help="股票代码")
    buy_p.add_argument("shares", type=int, help="股数")
    buy_p.add_argument("price", type=float, help="买入价格")
    buy_p.add_argument("--sl", type=float, default=0, help="止损价")
    buy_p.add_argument("--tp", type=float, default=0, help="止盈价")
    buy_p.add_argument("--notes", type=str, default="", help="备注")

    sell_p = sub.add_parser("sell", help="卖出平仓")
    sell_p.add_argument("stock_code", type=str, help="股票代码")
    sell_p.add_argument("price", type=float, help="卖出价格")
    sell_p.add_argument("--reason", type=str, default="", help="卖出原因")

    size_p = sub.add_parser("size", help="计算建议仓位")
    size_p.add_argument("stock_code", type=str, help="股票代码")
    size_p.add_argument("shares", type=str, help="股数 (用 - 跳过)")
    size_p.add_argument("price", type=float, help="买入价格")
    size_p.add_argument("capital", type=float, help="总资金")
    size_p.add_argument("--sl", type=float, default=0, help="止损价")
    size_p.add_argument("--method", choices=["risk", "pct", "atr"], default="risk", help="仓位计算方法")

    parser.add_argument("--capital", type=float, default=1_000_000, help="总资金 (默认: 1,000,000)")
    parser.add_argument("--db", type=str, default="cache/portfolio.db", help="数据库路径")
    return parser.parse_args()


def main():
    args = parse_args()

    rm = RiskManager(db_path=args.db)

    if args.command == "buy":
        cmd_buy(rm, args.stock_code, args.shares, args.price, args.sl, args.tp, args.notes)

    elif args.command == "sell":
        cmd_sell(rm, args.stock_code, args.price, args.reason)

    elif args.command == "status":
        cmd_status(rm, args.capital)

    elif args.command == "list":
        cmd_list(rm)

    elif args.command == "trades":
        stock = getattr(args, "stock_code", None)
        cmd_trades(rm, stock)

    elif args.command == "check":
        cmd_check(rm)

    elif args.command == "update":
        cmd_update(rm)

    elif args.command == "summary":
        cmd_summary(rm, args.capital)

    elif args.command == "size":
        shares = None if args.shares == "-" else int(args.shares)
        cmd_size(args.stock_code, shares, args.price, args.capital, args.sl, args.method)

    else:
        parser = argparse.ArgumentParser(
            description="持仓管理 - 跟踪持仓、计算P&L、管理止损止盈"
        )
        parser.print_help()


if __name__ == "__main__":
    main()

