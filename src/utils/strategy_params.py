"""Strategy parameters management with SQLite storage.

Supports parameter lookup priority: stock_code > market > default
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Default strategy parameters by market
DEFAULT_MARKET_PARAMS = {
    "a_share": {
        # A股市场：政策影响大、风格切换快、波动性高
        "highsell_lookback": 8,
        "highsell_threshold": 0.06,
        "ml_confidence_threshold": 0.30,
        "rolling_train_window": 60,
        "rolling_retrain_interval": 8,
        "bear_market_threshold": -0.012,
    },
    "hk": {
        # 港股市场：受A股和美股双重影响
        "highsell_lookback": 10,
        "highsell_threshold": 0.07,
        "ml_confidence_threshold": 0.32,
        "rolling_train_window": 80,
        "rolling_retrain_interval": 10,
        "bear_market_threshold": -0.008,
    },
    "us": {
        # 美股市场：趋势性强、波动相对平稳
        "highsell_lookback": 12,
        "highsell_threshold": 0.09,
        "ml_confidence_threshold": 0.35,
        "rolling_train_window": 100,
        "rolling_retrain_interval": 12,
        "bear_market_threshold": -0.005,
    },
    "default": {
        "highsell_lookback": 10,
        "highsell_threshold": 0.08,
        "ml_confidence_threshold": 0.35,
        "rolling_train_window": 90,
        "rolling_retrain_interval": 10,
        "bear_market_threshold": -0.008,
    },
}


# Default rule-based strategy parameters
DEFAULT_RULE_PARAMS = {
    # MA Golden Cross Strategy
    "ma_golden_cross": {
        "fast_ma": 5,
        "slow_ma": 10,
        "volume_ratio": 1.0,
    },
    # Bull Trend Strategy
    "bull_trend": {
        "ma5_period": 5,
        "ma10_period": 10,
        "ma20_period": 20,
        "pullback_threshold": 0.05,
    },
    # Shrink Pullback Strategy
    "shrink_pullback": {
        "lookback": 5,
        "ma_period": 5,
        "volume_shrink": 0.8,
    },
    # Bottom Volume Strategy
    "bottom_volume": {
        "drop_threshold": 0.10,
        "volume_multiplier": 2.0,
    },
    # Box Oscillation Strategy
    "box_oscillation": {
        "lookback": 40,
        "box_touch_min": 2,
        "support_margin": 0.03,
        "resistance_margin": 0.03,
        "box_width_min": 0.05,
    },
    # Emotion Cycle Strategy
    "emotion_cycle": {
        "volume_shrink": 0.5,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
    },
    # Volume Breakout Strategy
    "volume_breakout": {
        "lookback": 15,
        "volume_multiplier": 1.5,
    },
    # One Yang Three Yin Strategy
    "one_yang_three_yin": {
        "body_threshold": 0.02,
    },
    # MACD Divergence Strategy
    "macd_divergence": {
        "lookback": 15,
    },
    # High Sell Low Buy Strategy
    "highsell_lowbuy": {
        "lookback": 20,
        "threshold": 0.15,
    },
}


class StrategyParamManager:
    """Strategy parameter manager with SQLite storage.

    Supports parameter lookup priority: stock_code > market > default
    """

    def __init__(self, db_path: str = "cache/strategy_params.db"):
        """Initialize StrategyParamStore."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._init_default_params()

    def _init_db(self) -> None:
        """Initialize SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            # Strategy parameters table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS strategy_params (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stock_code TEXT,
                    market TEXT,
                    strategy_name TEXT NOT NULL,
                    params TEXT NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(stock_code, market, strategy_name)
                )
            """)

            # Market parameters table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS market_params (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stock_code TEXT,
                    market TEXT NOT NULL,
                    params TEXT NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(stock_code, market)
                )
            """)

            # Create indexes
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_strategy_lookup
                ON strategy_params(stock_code, market, strategy_name)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_market_lookup
                ON market_params(stock_code, market)
            """)

            conn.commit()

    def _init_default_params(self) -> None:
        """Initialize default parameters if not exists."""
        with sqlite3.connect(self.db_path) as conn:
            # Check if default params exist
            cursor = conn.execute(
                "SELECT COUNT(*) FROM market_params WHERE market = 'default'"
            )
            if cursor.fetchone()[0] == 0:
                # Insert default market params
                for market, params in DEFAULT_MARKET_PARAMS.items():
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO market_params (market, params, description)
                        VALUES (?, ?, ?)
                        """,
                        (
                            market,
                            json.dumps(params),
                            f"Default {market} market parameters",
                        ),
                    )

                # Insert default rule strategy params
                for strategy, params in DEFAULT_RULE_PARAMS.items():
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO strategy_params (market, strategy_name, params, description)
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            "default",
                            strategy,
                            json.dumps(params),
                            f"Default {strategy} parameters",
                        ),
                    )

                conn.commit()
                logger.info("Initialized default strategy parameters")

    def get_market_params(
        self,
        market: str,
        stock_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get market parameters with priority: stock_code > market > default.

        Args:
            market: Market type ('a_share', 'hk', 'us')
            stock_code: Stock code (optional, for stock-specific params)

        Returns:
            Dictionary of market parameters
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Priority 1: Stock-specific market params
            if stock_code:
                cursor = conn.execute(
                    "SELECT params FROM market_params WHERE stock_code = ? AND market = ?",
                    (stock_code, market),
                )
                row = cursor.fetchone()
                if row:
                    logger.debug(f"Found stock-specific market params for {stock_code}")
                    return json.loads(row["params"])

            # Priority 2: Market-level params
            cursor = conn.execute(
                "SELECT params FROM market_params WHERE stock_code IS NULL AND market = ?",
                (market,),
            )
            row = cursor.fetchone()
            if row:
                logger.debug(f"Found market params for {market}")
                return json.loads(row["params"])

            # Priority 3: Default params
            cursor = conn.execute(
                "SELECT params FROM market_params WHERE stock_code IS NULL AND market = 'default'",
            )
            row = cursor.fetchone()
            if row:
                logger.debug("Using default market params")
                return json.loads(row["params"])

            # Fallback to hardcoded defaults
            return DEFAULT_MARKET_PARAMS.get("default", {})

    def get_strategy_params(
        self,
        strategy_name: str,
        market: str = "default",
        stock_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get strategy parameters with priority: stock_code > market > default.

        Args:
            strategy_name: Strategy name (e.g., 'ma_golden_cross', 'box_oscillation')
            market: Market type ('a_share', 'hk', 'us', 'default')
            stock_code: Stock code (optional, for stock-specific params)

        Returns:
            Dictionary of strategy parameters
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Priority 1: Stock-specific + market params
            if stock_code:
                cursor = conn.execute(
                    """
                    SELECT params FROM strategy_params
                    WHERE stock_code = ? AND market = ? AND strategy_name = ?
                    """,
                    (stock_code, market, strategy_name),
                )
                row = cursor.fetchone()
                if row:
                    logger.debug(
                        f"Found stock-specific params for {stock_code}/{strategy_name}"
                    )
                    return json.loads(row["params"])

            # Priority 2: Market-level params
            cursor = conn.execute(
                """
                SELECT params FROM strategy_params
                WHERE stock_code IS NULL AND market = ? AND strategy_name = ?
                """,
                (market, strategy_name),
            )
            row = cursor.fetchone()
            if row:
                logger.debug(f"Found market params for {market}/{strategy_name}")
                return json.loads(row["params"])

            # Priority 3: Default params
            cursor = conn.execute(
                """
                SELECT params FROM strategy_params
                WHERE stock_code IS NULL AND market = 'default' AND strategy_name = ?
                """,
                (strategy_name,),
            )
            row = cursor.fetchone()
            if row:
                logger.debug(f"Using default params for {strategy_name}")
                return json.loads(row["params"])

            # Fallback to hardcoded defaults
            return DEFAULT_RULE_PARAMS.get(strategy_name, {})

    def set_market_params(
        self,
        market: str,
        params: Dict[str, Any],
        stock_code: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        """Set market parameters.

        Args:
            market: Market type
            params: Parameters dictionary
            stock_code: Stock code (optional, for stock-specific params)
            description: Description of the parameters
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO market_params (stock_code, market, params, description, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (stock_code, market, json.dumps(params), description),
            )
            conn.commit()
            logger.info(f"Set market params for {stock_code or 'default'}/{market}")

    def set_strategy_params(
        self,
        strategy_name: str,
        params: Dict[str, Any],
        market: str = "default",
        stock_code: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        """Set strategy parameters.

        Args:
            strategy_name: Strategy name
            params: Parameters dictionary
            market: Market type
            stock_code: Stock code (optional, for stock-specific params)
            description: Description of the parameters
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO strategy_params
                (stock_code, market, strategy_name, params, description, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (stock_code, market, strategy_name, json.dumps(params), description),
            )
            conn.commit()
            logger.info(
                f"Set strategy params for {stock_code or 'default'}/{market}/{strategy_name}"
            )

    def delete_market_params(
        self,
        market: str,
        stock_code: Optional[str] = None,
    ) -> None:
        """Delete market parameters."""
        with sqlite3.connect(self.db_path) as conn:
            if stock_code:
                conn.execute(
                    "DELETE FROM market_params WHERE stock_code = ? AND market = ?",
                    (stock_code, market),
                )
            else:
                conn.execute(
                    "DELETE FROM market_params WHERE stock_code IS NULL AND market = ?",
                    (market,),
                )
            conn.commit()
            logger.info(f"Deleted market params for {stock_code or 'default'}/{market}")

    def delete_strategy_params(
        self,
        strategy_name: str,
        market: str = "default",
        stock_code: Optional[str] = None,
    ) -> None:
        """Delete strategy parameters."""
        with sqlite3.connect(self.db_path) as conn:
            if stock_code:
                conn.execute(
                    """
                    DELETE FROM strategy_params
                    WHERE stock_code = ? AND market = ? AND strategy_name = ?
                    """,
                    (stock_code, market, strategy_name),
                )
            else:
                conn.execute(
                    """
                    DELETE FROM strategy_params
                    WHERE stock_code IS NULL AND market = ? AND strategy_name = ?
                    """,
                    (market, strategy_name),
                )
            conn.commit()
            logger.info(
                f"Deleted strategy params for {stock_code or 'default'}/{market}/{strategy_name}"
            )

    def list_params(
        self,
        stock_code: Optional[str] = None,
        market: Optional[str] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """List all parameters, optionally filtered by stock_code or market.

        Returns:
            Dictionary with 'market_params' and 'strategy_params' lists
        """
        result = {"market_params": [], "strategy_params": []}

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Build query for market params
            market_query = "SELECT * FROM market_params WHERE 1=1"
            market_params = []
            if stock_code:
                market_query += " AND stock_code = ?"
                market_params.append(stock_code)
            if market:
                market_query += " AND market = ?"
                market_params.append(market)

            cursor = conn.execute(market_query, market_params)
            for row in cursor.fetchall():
                result["market_params"].append(
                    {
                        "stock_code": row["stock_code"],
                        "market": row["market"],
                        "params": json.loads(row["params"]),
                        "description": row["description"],
                        "updated_at": row["updated_at"],
                    }
                )

            # Build query for strategy params
            strategy_query = "SELECT * FROM strategy_params WHERE 1=1"
            strategy_params = []
            if stock_code:
                strategy_query += " AND stock_code = ?"
                strategy_params.append(stock_code)
            if market:
                strategy_query += " AND market = ?"
                strategy_params.append(market)

            cursor = conn.execute(strategy_query, strategy_params)
            for row in cursor.fetchall():
                result["strategy_params"].append(
                    {
                        "stock_code": row["stock_code"],
                        "market": row["market"],
                        "strategy_name": row["strategy_name"],
                        "params": json.loads(row["params"]),
                        "description": row["description"],
                        "updated_at": row["updated_at"],
                    }
                )

        return result

    def get_all_strategy_params(
        self,
        strategy_name: str,
        market: str = "default",
        stock_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get all parameters for a strategy, merged with defaults.

        This ensures all required parameters are present even if not explicitly set.

        Args:
            strategy_name: Strategy name
            market: Market type
            stock_code: Stock code (optional)

        Returns:
            Complete dictionary of strategy parameters
        """
        # Start with hardcoded defaults
        defaults = DEFAULT_RULE_PARAMS.get(strategy_name, {})

        # Get custom params from database
        custom_params = self.get_strategy_params(strategy_name, market, stock_code)

        # Merge: custom params override defaults
        merged = {**defaults, **custom_params}

        return merged


# Singleton instance
_manager: Optional[StrategyParamManager] = None


def get_param_manager(
    db_path: str = "cache/strategy_params.db",
) -> StrategyParamManager:
    """Get or create the singleton StrategyParamManager instance."""
    global _manager
    if _manager is None:
        _manager = StrategyParamManager(db_path)
    return _manager
