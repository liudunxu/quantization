"""Risk management - position sizing, stop-loss, and risk controls.

For mid-term (swing) trading with at most 1 round-trip per stock per day.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "cache/portfolio.db"


@dataclass
class Position:
    """A single position in the portfolio."""
    stock_code: str
    stock_name: str = ""
    market: str = ""
    direction: str = "long"
    entry_price: float = 0.0
    current_price: float = 0.0
    shares: int = 0
    entry_date: str = ""
    stop_loss: float = 0.0
    take_profit: float = 0.0
    cost_basis: float = 0.0
    notes: str = ""

    @property
    def market_value(self) -> float:
        return self.current_price * self.shares

    @property
    def profit_loss(self) -> float:
        if self.entry_price <= 0 or self.shares <= 0:
            return 0.0
        return (self.current_price - self.entry_price) * self.shares

    @property
    def profit_loss_pct(self) -> float:
        if self.entry_price <= 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price


@dataclass
class TradeRecord:
    """A completed trade record."""
    stock_code: str
    stock_name: str = ""
    direction: str = "buy"
    price: float = 0.0
    shares: int = 0
    amount: float = 0.0
    date: str = ""
    stop_loss: float = 0.0
    take_profit: float = 0.0
    notes: str = ""


class PositionSizer:
    """Calculate position sizes based on risk parameters.

    Supports multiple sizing methods suitable for mid-term trading:
    - Fixed percentage of portfolio
    - Risk-based (ATR stop-loss distance)
    - Volatility-adjusted (inverse volatility weighting)
    """

    def __init__(
        self,
        max_position_pct: float = 0.20,
        max_portfolio_risk_pct: float = 0.02,
        max_positions: int = 10,
    ):
        self.max_position_pct = max_position_pct
        self.max_portfolio_risk_pct = max_portfolio_risk_pct
        self.max_positions = max_positions

    def size_by_portfolio_pct(
        self,
        total_capital: float,
        position_pct: Optional[float] = None,
        price: float = 0.0,
    ) -> Tuple[int, float]:
        """Size position as a percentage of total portfolio.

        Returns (shares, position_value).
        """
        pct = position_pct or self.max_position_pct
        position_value = total_capital * pct
        if price <= 0:
            return 0, 0.0
        shares = int(position_value / price)
        return shares, shares * price

    def size_by_risk(
        self,
        total_capital: float,
        entry_price: float,
        stop_loss_price: float,
        min_shares: int = 100,
    ) -> Tuple[int, float]:
        """Size position based on risk per trade.

        Risk = (entry - stop_loss) / entry, position = risk_budget / risk_per_share.
        For mid-term, risk_budget = total_capital * max_portfolio_risk_pct.

        Returns (shares, position_value).
        """
        if entry_price <= 0 or stop_loss_price <= 0 or stop_loss_price >= entry_price:
            return 0, 0.0

        risk_per_share = entry_price - stop_loss_price
        if risk_per_share <= 0:
            return 0, 0.0

        risk_budget = total_capital * self.max_portfolio_risk_pct
        shares = int(risk_budget / risk_per_share)
        shares = max(shares, 0)

        max_shares = int(total_capital * self.max_position_pct / entry_price)
        shares = min(shares, max_shares)

        if shares < min_shares:
            return 0, 0.0

        return shares, shares * entry_price

    def size_by_atr(
        self,
        total_capital: float,
        entry_price: float,
        atr: float,
        atr_multiplier: float = 2.0,
        min_shares: int = 100,
    ) -> Tuple[int, float]:
        """Size position using ATR-based stop-loss distance.

        Stop-loss = entry - atr * multiplier.
        Returns (shares, position_value).
        """
        if entry_price <= 0 or atr <= 0:
            return 0, 0.0
        stop_loss = entry_price - atr * atr_multiplier
        if stop_loss <= 0:
            stop_loss = entry_price * 0.92
        return self.size_by_risk(total_capital, entry_price, stop_loss, min_shares)


class StopLossCalculator:
    """Calculate stop-loss and take-profit levels.

    Supports multiple methods suitable for mid-term holding periods.
    """

    def fixed_pct_stop(
        self,
        entry_price: float,
        stop_pct: float = 0.07,
        take_profit_pct: float = 0.15,
    ) -> Tuple[float, float]:
        """Fixed percentage stop-loss and take-profit.

        Default: 7% stop-loss, 15% take-profit (R:R ≈ 1:2.1).
        """
        return entry_price * (1 - stop_pct), entry_price * (1 + take_profit_pct)

    def atr_stop(
        self,
        entry_price: float,
        atr: float,
        multiplier: float = 2.0,
        take_profit_ratio: float = 3.0,
    ) -> Tuple[float, float]:
        """ATR-based stop-loss with proportional take-profit.

        Stop = entry - atr * multiplier
        TP = entry + atr * multiplier * take_profit_ratio
        """
        stop = entry_price - atr * multiplier
        tp = entry_price + atr * multiplier * take_profit_ratio
        return max(stop, 0), tp

    def support_stop(
        self,
        entry_price: float,
        support_price: float,
        buffer_pct: float = 0.015,
        take_profit_pct: float = 0.15,
    ) -> Tuple[float, float]:
        """Support-level stop-loss with buffer.

        Stop = support * (1 - buffer)
        """
        stop = support_price * (1 - buffer_pct)
        tp = entry_price * (1 + take_profit_pct)
        return stop, tp

    def trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        highest_since_entry: float,
        trail_pct: float = 0.05,
    ) -> float:
        """Trailing stop from highest point.

        Returns the current stop-loss level.
        """
        trail_from = max(highest_since_entry, current_price)
        return trail_from * (1 - trail_pct)

    def ma_stop(
        self,
        entry_price: float,
        ma_price: float,
        buffer_pct: float = 0.01,
        take_profit_pct: float = 0.15,
    ) -> Tuple[float, float]:
        """MA-based stop-loss (e.g., MA20 or MA60 for mid-term).

        Stop = MA * (1 - buffer)
        """
        stop = ma_price * (1 - buffer_pct)
        tp = entry_price * (1 + take_profit_pct)
        return stop, tp


class RiskManager:
    """Portfolio-level risk manager with position tracking.

    Manages positions via a simple JSON file, calculates portfolio risk,
    and enforces risk limits suitable for mid-term trading.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.sizer = PositionSizer()
        self.stop_calc = StopLossCalculator()
        self._positions: Dict[str, Position] = {}
        self._trades: List[TradeRecord] = []
        self._load()

    def _load(self):
        if self.db_path.exists():
            try:
                with open(self.db_path, "r") as f:
                    data = json.load(f)
                for code, pos_data in data.get("positions", {}).items():
                    self._positions[code] = Position(**pos_data)
                for t_data in data.get("trades", []):
                    self._trades.append(TradeRecord(**t_data))
            except Exception as e:
                logger.warning(f"Failed to load portfolio: {e}")

    def _save(self):
        try:
            data = {
                "positions": {code: asdict(pos) for code, pos in self._positions.items()},
                "trades": [asdict(t) for t in self._trades],
            }
            with open(self.db_path, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save portfolio: {e}")

    # ---- Position Management ----

    def open_position(
        self,
        stock_code: str,
        entry_price: float,
        shares: int,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
        stock_name: str = "",
        market: str = "",
        notes: str = "",
    ) -> Position:
        """Open a new position. Closes existing position for same stock first."""
        if stock_code in self._positions:
            self.close_position(stock_code, entry_price, "replace")

        pos = Position(
            stock_code=stock_code,
            stock_name=stock_name,
            market=market,
            entry_price=entry_price,
            current_price=entry_price,
            shares=shares,
            entry_date=date.today().isoformat(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            cost_basis=entry_price * shares,
            notes=notes,
        )
        self._positions[stock_code] = pos
        self._trades.append(TradeRecord(
            stock_code=stock_code,
            stock_name=stock_name,
            direction="buy",
            price=entry_price,
            shares=shares,
            amount=entry_price * shares,
            date=date.today().isoformat(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            notes=notes,
        ))
        self._save()
        logger.info(f"Opened position: {stock_code} {shares}@{entry_price:.2f}")
        return pos

    def close_position(
        self,
        stock_code: str,
        exit_price: float,
        reason: str = "",
    ) -> Optional[TradeRecord]:
        """Close a position and record the trade."""
        if stock_code not in self._positions:
            logger.warning(f"No position for {stock_code}")
            return None

        pos = self._positions[stock_code]
        trade = TradeRecord(
            stock_code=stock_code,
            stock_name=pos.stock_name,
            direction="sell",
            price=exit_price,
            shares=pos.shares,
            amount=exit_price * pos.shares,
            date=date.today().isoformat(),
            stop_loss=pos.stop_loss,
            take_profit=pos.take_profit,
            notes=f"{reason}".strip(),
        )
        del self._positions[stock_code]
        self._trades.append(trade)
        self._save()
        pnl = (exit_price - pos.entry_price) * pos.shares
        pnl_pct = (exit_price / pos.entry_price - 1) * 100 if pos.entry_price > 0 else 0
        logger.info(
            f"Closed position: {stock_code} {pos.shares}@{exit_price:.2f} "
            f"P&L: {pnl:+.2f} ({pnl_pct:+.1f}%) reason={reason}"
        )
        return trade

    def update_price(self, stock_code: str, current_price: float):
        """Update current price for a position."""
        if stock_code in self._positions:
            self._positions[stock_code].current_price = current_price

    def update_prices(self, prices: Dict[str, float]):
        """Batch update current prices."""
        for code, price in prices.items():
            self.update_price(code, price)
        self._save()

    def get_position(self, stock_code: str) -> Optional[Position]:
        return self._positions.get(stock_code)

    def get_all_positions(self) -> Dict[str, Position]:
        return dict(self._positions)

    # ---- Risk Checks ----

    def can_open_position(self, stock_code: str, total_capital: float) -> Tuple[bool, str]:
        """Check if a new position can be opened (risk limits)."""
        if len(self._positions) >= self.sizer.max_positions:
            return False, f"Max positions ({self.sizer.max_positions}) reached"

        total_risk = 0.0
        for pos in self._positions.values():
            if pos.entry_price > 0 and pos.stop_loss > 0:
                risk_per_share = pos.entry_price - pos.stop_loss
                total_risk += risk_per_share * pos.shares

        max_risk = total_capital * self.sizer.max_portfolio_risk_pct
        if total_risk >= max_risk:
            return False, f"Portfolio risk ({total_risk:.0f}) exceeds limit ({max_risk:.0f})"

        if stock_code in self._positions:
            return False, f"Already holding {stock_code}"

        return True, "OK"

    def check_stop_loss_hit(self, stock_code: str, current_price: float) -> Optional[str]:
        """Check if stop-loss is triggered. Returns reason or None."""
        pos = self._positions.get(stock_code)
        if pos is None or pos.stop_loss <= 0:
            return None
        if current_price <= pos.stop_loss:
            return f"Stop-loss hit: {current_price:.2f} <= {pos.stop_loss:.2f}"
        return None

    def check_take_profit_hit(self, stock_code: str, current_price: float) -> Optional[str]:
        """Check if take-profit is triggered. Returns reason or None."""
        pos = self._positions.get(stock_code)
        if pos is None or pos.take_profit <= 0:
            return None
        if current_price >= pos.take_profit:
            return f"Take-profit hit: {current_price:.2f} >= {pos.take_profit:.2f}"
        return None

    # ---- Portfolio Metrics ----

    def total_market_value(self) -> float:
        return sum(p.market_value for p in self._positions.values())

    def total_profit_loss(self) -> float:
        return sum(p.profit_loss for p in self._positions.values())

    def total_cost_basis(self) -> float:
        return sum(p.cost_basis for p in self._positions.values())

    def portfolio_summary(self, total_capital: float) -> dict:
        """Get a full portfolio summary."""
        positions = []
        for code, pos in self._positions.items():
            pnl = pos.profit_loss
            pnl_pct = pos.profit_loss_pct
            positions.append({
                "code": code,
                "name": pos.stock_name,
                "shares": pos.shares,
                "entry": pos.entry_price,
                "current": pos.current_price,
                "stop_loss": pos.stop_loss,
                "take_profit": pos.take_profit,
                "pnl": round(pnl, 2),
                "pnl_pct": f"{pnl_pct:+.1%}",
                "entry_date": pos.entry_date,
                "weight": f"{pos.market_value / total_capital:.1%}" if total_capital > 0 else "0%",
            })

        total_pnl = self.total_profit_loss()
        total_cost = self.total_cost_basis()
        invested = total_cost if total_cost > 0 else 1
        return {
            "date": date.today().isoformat(),
            "total_capital": total_capital,
            "positions_count": len(self._positions),
            "total_invested": round(total_cost, 2),
            "total_market_value": round(self.total_market_value(), 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": f"{total_pnl / invested:+.1%}" if invested > 0 else "0%",
            "cash_available": round(total_capital - total_cost, 2),
            "positions": positions,
        }

    # ---- Trade History ----

    def get_trades(self, stock_code: Optional[str] = None) -> List[TradeRecord]:
        if stock_code:
            return [t for t in self._trades if t.stock_code == stock_code]
        return list(self._trades)

    def trade_statistics(self) -> dict:
        """Calculate trading statistics from history."""
        sells = [t for t in self._trades if t.direction == "sell"]
        if not sells:
            return {"total_trades": 0}

        buy_trades = [t for t in self._trades if t.direction == "buy"]
        total_bought = sum(t.amount for t in buy_trades)
        total_sold = sum(t.amount for t in sells)

        return {
            "total_trades": len(sells),
            "total_bought": round(total_bought, 2),
            "total_sold": round(total_sold, 2),
            "net_pnl": round(total_sold - total_bought, 2),
        }


_singleton: Optional[RiskManager] = None


def get_risk_manager(db_path: str = DEFAULT_DB_PATH) -> RiskManager:
    """Get singleton RiskManager instance."""
    global _singleton
    if _singleton is None:
        _singleton = RiskManager(db_path)
    return _singleton
