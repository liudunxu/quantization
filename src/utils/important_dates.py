"""Important dates manager - tracks dates with extreme market impact."""

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd

logger = logging.getLogger(__name__)


class ImportantDatesManager:
    """Manager for important dates that significantly impact stock prices."""

    def __init__(self, db_path: str = "cache/important_dates.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS important_dates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    market TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    description TEXT,
                    impact_level TEXT DEFAULT 'high',
                    source TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, market, event_type)
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_date_market 
                ON important_dates(date, market)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_market 
                ON important_dates(market)
            """)

    def add_date(
        self,
        date: str,
        market: str,
        event_type: str,
        description: str = "",
        impact_level: str = "high",
        source: str = "manual",
    ) -> bool:
        """Add an important date.

        Args:
            date: Date string in YYYY-MM-DD format
            market: Market identifier (a_share, hk, us, global)
            event_type: Type of event (rate_decision, policy, crisis, etc.)
            description: Event description
            impact_level: Impact level (high, medium, low)
            source: Data source

        Returns:
            True if added successfully
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO important_dates 
                    (date, market, event_type, description, impact_level, source)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (date, market, event_type, description, impact_level, source),
                )
            return True
        except Exception as e:
            logger.error(f"Failed to add important date: {e}")
            return False

    def get_dates(
        self,
        market: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get important dates with optional filters.

        Args:
            market: Filter by market
            start_date: Start date filter
            end_date: End date filter
            event_type: Filter by event type

        Returns:
            DataFrame with important dates
        """
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM important_dates WHERE 1=1"
            params = []

            if market:
                query += " AND market = ?"
                params.append(market)

            if start_date:
                query += " AND date >= ?"
                params.append(start_date)

            if end_date:
                query += " AND date <= ?"
                params.append(end_date)

            if event_type:
                query += " AND event_type = ?"
                params.append(event_type)

            query += " ORDER BY date DESC"

            return pd.read_sql_query(query, conn, params=params)

    def get_dates_as_list(
        self,
        market: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[str]:
        """Get important dates as a list of date strings.

        Args:
            market: Filter by market
            start_date: Start date filter
            end_date: End date filter

        Returns:
            List of date strings in YYYY-MM-DD format
        """
        df = self.get_dates(market, start_date, end_date)
        if df.empty:
            return []
        return df["date"].unique().tolist()

    def delete_date(self, date: str, market: str, event_type: str) -> bool:
        """Delete an important date.

        Args:
            date: Date string
            market: Market identifier
            event_type: Event type

        Returns:
            True if deleted successfully
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    DELETE FROM important_dates 
                    WHERE date = ? AND market = ? AND event_type = ?
                """,
                    (date, market, event_type),
                )
            return True
        except Exception as e:
            logger.error(f"Failed to delete important date: {e}")
            return False

    def fetch_from_web(self, market: str = "global", years: int = 3) -> int:
        """Fetch important dates from web sources.

        Args:
            market: Market identifier
            years: Number of years to look back

        Returns:
            Number of dates added
        """
        count = 0
        start_year = datetime.now().year - years

        # Predefined important dates for different markets
        important_events = self._get_predefined_events(market, start_year)

        for event in important_events:
            if self.add_date(
                date=event["date"],
                market=event["market"],
                event_type=event["event_type"],
                description=event["description"],
                impact_level=event.get("impact_level", "high"),
                source=event.get("source", "predefined"),
            ):
                count += 1

        logger.info(f"Added {count} important dates for {market}")
        return count

    def _get_predefined_events(self, market: str, start_year: int) -> List[Dict]:
        """Get predefined important events.

        Args:
            market: Market identifier
            start_year: Start year

        Returns:
            List of event dictionaries
        """
        events = []

        # Global events affecting all markets
        global_events = [
            # COVID-19 pandemic
            {
                "date": "2020-03-09",
                "market": "global",
                "event_type": "crisis",
                "description": "COVID-19全球股市暴跌开始",
            },
            {
                "date": "2020-03-12",
                "market": "global",
                "event_type": "crisis",
                "description": "黑色星期四 - 全球股市熔断",
            },
            {
                "date": "2020-03-16",
                "market": "global",
                "event_type": "crisis",
                "description": "美股再次熔断",
            },
            {
                "date": "2020-03-23",
                "market": "global",
                "event_type": "policy",
                "description": "美联储无限量QE",
            },
            # 2022 events
            {
                "date": "2022-02-24",
                "market": "global",
                "event_type": "crisis",
                "description": "俄乌冲突爆发",
            },
            {
                "date": "2022-03-16",
                "market": "global",
                "event_type": "rate_decision",
                "description": "美联储首次加息25基点",
            },
            {
                "date": "2022-06-15",
                "market": "global",
                "event_type": "rate_decision",
                "description": "美联储加息75基点",
            },
            {
                "date": "2022-09-21",
                "market": "global",
                "event_type": "rate_decision",
                "description": "美联储连续第三次加息75基点",
            },
            # 2023 events
            {
                "date": "2023-03-10",
                "market": "global",
                "event_type": "crisis",
                "description": "硅谷银行倒闭",
            },
            {
                "date": "2023-03-15",
                "market": "global",
                "event_type": "crisis",
                "description": "瑞信危机",
            },
            {
                "date": "2023-05-03",
                "market": "global",
                "event_type": "rate_decision",
                "description": "美联储加息25基点至5-5.25%",
            },
            # 2024 events
            {
                "date": "2024-01-24",
                "market": "global",
                "event_type": "policy",
                "description": "中国央行降准50基点",
            },
            {
                "date": "2024-03-20",
                "market": "global",
                "event_type": "rate_decision",
                "description": "日本央行结束负利率",
            },
        ]

        # A-share specific events
        a_share_events = [
            {
                "date": "2020-07-06",
                "market": "a_share",
                "event_type": "policy",
                "description": "A股大涨 - 牛市信号",
            },
            {
                "date": "2021-02-18",
                "market": "a_share",
                "event_type": "crisis",
                "description": "抱团股崩盘开始",
            },
            {
                "date": "2022-04-27",
                "market": "a_share",
                "event_type": "crisis",
                "description": "A股触底反弹",
            },
            {
                "date": "2022-10-31",
                "market": "a_share",
                "event_type": "crisis",
                "description": "A股再次触底",
            },
            {
                "date": "2023-01-30",
                "market": "a_share",
                "event_type": "policy",
                "description": "春节后A股高开",
            },
            {
                "date": "2023-08-28",
                "market": "a_share",
                "event_type": "policy",
                "description": "印花税减半利好",
            },
            {
                "date": "2024-02-05",
                "market": "a_share",
                "event_type": "crisis",
                "description": "A股大跌 - 量化危机",
            },
            {
                "date": "2024-02-06",
                "market": "a_share",
                "event_type": "policy",
                "description": "国家队救市",
            },
        ]

        # HK specific events
        hk_events = [
            {
                "date": "2020-03-23",
                "market": "hk",
                "event_type": "crisis",
                "description": "港股跌至低点",
            },
            {
                "date": "2021-02-18",
                "market": "hk",
                "event_type": "crisis",
                "description": "恒生科技指数见顶",
            },
            {
                "date": "2021-07-27",
                "market": "hk",
                "event_type": "crisis",
                "description": "港股科技股暴跌",
            },
            {
                "date": "2022-03-15",
                "market": "hk",
                "event_type": "crisis",
                "description": "港股史诗级暴跌",
            },
            {
                "date": "2022-10-31",
                "market": "hk",
                "event_type": "crisis",
                "description": "港股跌至低点",
            },
            {
                "date": "2023-01-27",
                "market": "hk",
                "event_type": "policy",
                "description": "港股春节后大涨",
            },
        ]

        # US specific events
        us_events = [
            {
                "date": "2020-03-23",
                "market": "us",
                "event_type": "crisis",
                "description": "美股见底",
            },
            {
                "date": "2021-01-27",
                "market": "us",
                "event_type": "crisis",
                "description": "GameStop散户大战",
            },
            {
                "date": "2022-01-03",
                "market": "us",
                "event_type": "crisis",
                "description": "美股开始大跌",
            },
            {
                "date": "2022-10-13",
                "market": "us",
                "event_type": "crisis",
                "description": "CPI数据引发暴跌",
            },
            {
                "date": "2023-01-06",
                "market": "us",
                "event_type": "policy",
                "description": "非农数据超预期",
            },
            {
                "date": "2024-01-19",
                "market": "us",
                "event_type": "policy",
                "description": "科技股财报季",
            },
        ]

        # Filter by year
        def filter_by_year(events_list, start_year):
            return [e for e in events_list if int(e["date"][:4]) >= start_year]

        # Select events based on market
        if market == "global":
            events.extend(filter_by_year(global_events, start_year))
        elif market == "a_share":
            events.extend(filter_by_year(global_events, start_year))
            events.extend(filter_by_year(a_share_events, start_year))
        elif market == "hk":
            events.extend(filter_by_year(global_events, start_year))
            events.extend(filter_by_year(hk_events, start_year))
        elif market == "us":
            events.extend(filter_by_year(global_events, start_year))
            events.extend(filter_by_year(us_events, start_year))
        else:
            # Return all
            events.extend(filter_by_year(global_events, start_year))
            events.extend(filter_by_year(a_share_events, start_year))
            events.extend(filter_by_year(hk_events, start_year))
            events.extend(filter_by_year(us_events, start_year))

        return events

    def search_web_events(self, market: str, start_date: str, end_date: str) -> int:
        """Search for important events from web (placeholder for future implementation).

        Args:
            market: Market identifier
            start_date: Start date
            end_date: End date

        Returns:
            Number of events found
        """
        # TODO: Implement web scraping for economic calendar
        # Possible sources:
        # - Investing.com economic calendar
        # - Yahoo Finance earnings calendar
        # - Central bank meeting schedules

        logger.info(f"Web search not yet implemented for {market}")
        return 0

    def detect_high_volatility_dates(
        self,
        df: pd.DataFrame,
        market: str = "a_share",
        threshold_std: float = 2.0,
        min_change_pct: float = 0.03,
    ) -> List[str]:
        """Detect dates with extremely high volatility from price data.

        Args:
            df: DataFrame with columns: date, open, high, low, close
            market: Market identifier
            threshold_std: Number of standard deviations to consider as extreme
            min_change_pct: Minimum percentage change to consider (default 3%)

        Returns:
            List of date strings in YYYY-MM-DD format
        """
        if df.empty or "date" not in df.columns:
            return []

        # Make a copy and select only needed columns
        required_cols = ["date", "open", "high", "low", "close"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            return []

        data = df[required_cols].copy()
        data["date"] = pd.to_datetime(data["date"])
        data = data.sort_values("date").reset_index(drop=True)

        # Calculate daily metrics
        # 1. Daily return (close to close)
        data["daily_return"] = data["close"].pct_change()

        # 2. Intraday volatility (high-low range / open)
        data["intraday_range"] = (data["high"] - data["low"]) / data["open"]

        # 3. Gap (open vs previous close)
        data["prev_close"] = data["close"].shift(1)
        data["gap"] = abs(data["open"] - data["prev_close"]) / data["prev_close"]

        # Remove first two rows with NaN (from pct_change and shift)
        data = data.iloc[2:].reset_index(drop=True)

        if len(data) < 10:
            logger.warning("Not enough data to detect high volatility dates")
            return []

        # Calculate thresholds
        return_mean = data["daily_return"].abs().mean()
        return_std = data["daily_return"].abs().std()
        range_mean = data["intraday_range"].mean()
        range_std = data["intraday_range"].std()

        # Handle edge case where std is 0
        if return_std == 0:
            return_std = 0.01
        if range_std == 0:
            range_std = 0.01

        # Identify extreme dates
        extreme_dates = []

        for idx, row in data.iterrows():
            date_str = row["date"].strftime("%Y-%m-%d")
            is_extreme = False
            reasons = []

            # Check daily return
            abs_return = abs(row["daily_return"])
            if (
                abs_return > min_change_pct
                and abs_return > return_mean + threshold_std * return_std
            ):
                is_extreme = True
                direction = "大涨" if row["daily_return"] > 0 else "大跌"
                reasons.append(f"日涨跌幅{row['daily_return']:.2%}{direction}")

            # Check intraday range
            if (
                row["intraday_range"] > min_change_pct
                and row["intraday_range"] > range_mean + threshold_std * range_std
            ):
                is_extreme = True
                reasons.append(f"日内振幅{row['intraday_range']:.2%}")

            # Check gap
            if row["gap"] > min_change_pct and row["gap"] > 0.02:  # 2% gap
                is_extreme = True
                gap_dir = "高开" if row["open"] > row["prev_close"] else "低开"
                reasons.append(f"跳空{gap_dir}{row['gap']:.2%}")

            if is_extreme:
                extreme_dates.append(
                    {
                        "date": date_str,
                        "reasons": reasons,
                        "return": row["daily_return"],
                        "range": row["intraday_range"],
                        "gap": row["gap"],
                    }
                )

        # Add to database
        count = 0
        for event in extreme_dates:
            description = "极端波动: " + ", ".join(event["reasons"])
            if self.add_date(
                date=event["date"],
                market=market,
                event_type="high_volatility",
                description=description,
                impact_level="high",
                source="auto_detected",
            ):
                count += 1

        logger.info(f"Detected {count} high volatility dates for {market}")
        return [e["date"] for e in extreme_dates]

    def get_or_detect_dates(
        self,
        df: pd.DataFrame,
        market: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        auto_detect: bool = True,
    ) -> List[str]:
        """Get important dates from database, with optional auto-detection.

        Args:
            df: Stock price DataFrame for auto-detection
            market: Market identifier
            start_date: Start date filter
            end_date: End date filter
            auto_detect: Whether to auto-detect if no dates found

        Returns:
            List of date strings to exclude
        """
        # First try to get from database
        dates = self.get_dates_as_list(market, start_date, end_date)

        if dates:
            logger.info(f"Found {len(dates)} important dates in database for {market}")
            return dates

        # If no dates found and auto_detect is enabled, detect from data
        if auto_detect and not df.empty:
            logger.info(
                f"No dates in database, auto-detecting high volatility dates for {market}"
            )
            dates = self.detect_high_volatility_dates(df, market)
            return dates

        return []


# Singleton instance
_important_dates_manager = None


def get_important_dates_manager() -> ImportantDatesManager:
    """Get singleton instance of ImportantDatesManager."""
    global _important_dates_manager
    if _important_dates_manager is None:
        _important_dates_manager = ImportantDatesManager()
    return _important_dates_manager
