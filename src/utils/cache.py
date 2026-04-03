"""Feature caching system with SQLite storage."""

import sqlite3
import json
import hashlib
from pathlib import Path
from typing import Any, Optional, Dict, List
import pandas as pd
from datetime import datetime
import io


class SQLiteFeatureCache:
    """SQLite-based feature cache with fine-grained control."""

    # Default cache expiration in hours (24 hours = 1 day)
    DEFAULT_TTL_HOURS = 24

    def __init__(self, cache_dir: str = "cache", ttl_hours: Optional[int] = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "feature_cache.db"
        self.ttl_hours = ttl_hours or self.DEFAULT_TTL_HOURS
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stock_code TEXT NOT NULL,
                    feature_type TEXT NOT NULL,
                    cache_key TEXT NOT NULL,
                    params TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    row_count INTEGER,
                    column_count INTEGER,
                    size_bytes INTEGER,
                    UNIQUE(stock_code, feature_type, cache_key)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stock_code TEXT NOT NULL,
                    feature_type TEXT NOT NULL,
                    cache_key TEXT NOT NULL,
                    data BLOB NOT NULL,
                    FOREIGN KEY (stock_code, feature_type, cache_key)
                        REFERENCES cache_metadata(stock_code, feature_type, cache_key)
                        ON DELETE CASCADE
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_stock_code
                ON cache_metadata(stock_code)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_feature_type
                ON cache_metadata(feature_type)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_key
                ON cache_metadata(cache_key)
            """)

            conn.commit()

    def _compute_key(
        self, stock_code: str, feature_type: str, params: Optional[Dict] = None
    ) -> str:
        """Compute cache key."""
        key_str = f"{stock_code}_{feature_type}"
        if params:
            key_str += f"_{hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()}"
        return key_str

    def _serialize_df(self, df: pd.DataFrame) -> bytes:
        """Serialize DataFrame to bytes."""
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=True)
        return buffer.getvalue()

    def _deserialize_df(self, data: bytes) -> pd.DataFrame:
        """Deserialize bytes to DataFrame."""
        buffer = io.BytesIO(data)
        return pd.read_parquet(buffer)

    def get_latest_date(
        self, stock_code: str, feature_type: str, params: Optional[Dict] = None
    ) -> Optional[pd.Timestamp]:
        """Get the latest date in cached data."""
        df = self.get(stock_code, feature_type, params)
        if df is None or df.empty or "date" not in df.columns:
            return None
        df["date"] = pd.to_datetime(df["date"])
        return df["date"].max()

    def merge_and_update(
        self,
        stock_code: str,
        feature_type: str,
        new_df: pd.DataFrame,
        params: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """Merge new data with cached data, deduplicate, and update cache.

        Args:
            stock_code: Stock code
            feature_type: Feature type
            new_df: New data to merge
            params: Optional params

        Returns:
            Merged and deduplicated DataFrame
        """
        cached_df = self.get(stock_code, feature_type, params)

        if cached_df is None or cached_df.empty:
            # No cached data, just cache the new data
            self.set(stock_code, feature_type, new_df, params)
            return new_df

        # Merge new data with cached data
        combined = pd.concat([cached_df, new_df], ignore_index=True)

        # Deduplicate by date (keep latest)
        if "date" in combined.columns:
            combined["date"] = pd.to_datetime(combined["date"])
            combined = combined.drop_duplicates(subset=["date"], keep="last")
            combined = combined.sort_values("date").reset_index(drop=True)

        # Update cache
        self.set(stock_code, feature_type, combined, params)

        return combined

    def get_latest_date(
        self, stock_code: str, feature_type: str, params: Optional[Dict] = None
    ) -> Optional[pd.Timestamp]:
        """Get the latest date in cached data."""
        df = self.get(stock_code, feature_type, params)
        if df is None or df.empty or "date" not in df.columns:
            return None
        df["date"] = pd.to_datetime(df["date"])
        return df["date"].max()

    def merge_and_update(
        self,
        stock_code: str,
        feature_type: str,
        new_df: pd.DataFrame,
        params: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """Merge new data with cached data, deduplicate, and update cache.

        Args:
            stock_code: Stock code
            feature_type: Feature type
            new_df: New data to merge
            params: Optional params

        Returns:
            Merged and deduplicated DataFrame
        """
        cached_df = self.get(stock_code, feature_type, params)

        if cached_df is None or cached_df.empty:
            # No cached data, just cache the new data
            self.set(stock_code, feature_type, new_df, params)
            return new_df

        # Merge new data with cached data
        combined = pd.concat([cached_df, new_df], ignore_index=True)

        # Deduplicate by date (keep latest)
        if "date" in combined.columns:
            combined["date"] = pd.to_datetime(combined["date"])
            combined = combined.drop_duplicates(subset=["date"], keep="last")
            combined = combined.sort_values("date").reset_index(drop=True)

        # Update cache
        self.set(stock_code, feature_type, combined, params)

        return combined

    def get(
        self, stock_code: str, feature_type: str, params: Optional[Dict] = None
    ) -> Optional[pd.DataFrame]:
        """Get cached features for a stock.

        Returns None if cache is expired (older than ttl_hours).
        """
        cache_key = self._compute_key(stock_code, feature_type, params)

        with sqlite3.connect(self.db_path) as conn:
            # Check if cache is expired
            cursor = conn.execute(
                "SELECT updated_at FROM cache_metadata WHERE stock_code = ? AND feature_type = ? AND cache_key = ?",
                (stock_code, feature_type, cache_key),
            )
            meta_row = cursor.fetchone()

            if meta_row is not None:
                updated_at = pd.to_datetime(meta_row[0])
                age_hours = (pd.Timestamp.now() - updated_at).total_seconds() / 3600
                if age_hours > self.ttl_hours:
                    # Cache expired, delete it
                    conn.execute(
                        "DELETE FROM cache_data WHERE stock_code = ? AND feature_type = ? AND cache_key = ?",
                        (stock_code, feature_type, cache_key),
                    )
                    conn.execute(
                        "DELETE FROM cache_metadata WHERE stock_code = ? AND feature_type = ? AND cache_key = ?",
                        (stock_code, feature_type, cache_key),
                    )
                    conn.commit()
                    return None

            # Get cached data
            cursor = conn.execute(
                "SELECT data FROM cache_data WHERE stock_code = ? AND feature_type = ? AND cache_key = ?",
                (stock_code, feature_type, cache_key),
            )
            row = cursor.fetchone()

            if row is None:
                return None

            try:
                return self._deserialize_df(row[0])
            except Exception:
                return None

    def set(
        self,
        stock_code: str,
        feature_type: str,
        df: pd.DataFrame,
        params: Optional[Dict] = None,
    ) -> None:
        """Cache features for a stock."""
        cache_key = self._compute_key(stock_code, feature_type, params)
        data = self._serialize_df(df)
        params_json = json.dumps(params) if params else None

        with sqlite3.connect(self.db_path) as conn:
            # Delete existing cache if any
            conn.execute(
                "DELETE FROM cache_data WHERE stock_code = ? AND feature_type = ? AND cache_key = ?",
                (stock_code, feature_type, cache_key),
            )
            conn.execute(
                "DELETE FROM cache_metadata WHERE stock_code = ? AND feature_type = ? AND cache_key = ?",
                (stock_code, feature_type, cache_key),
            )

            # Insert new cache
            conn.execute(
                """INSERT INTO cache_metadata
                   (stock_code, feature_type, cache_key, params, row_count, column_count, size_bytes)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    stock_code,
                    feature_type,
                    cache_key,
                    params_json,
                    len(df),
                    len(df.columns),
                    len(data),
                ),
            )
            conn.execute(
                "INSERT INTO cache_data (stock_code, feature_type, cache_key, data) VALUES (?, ?, ?, ?)",
                (stock_code, feature_type, cache_key, data),
            )
            conn.commit()

    def get_by_date_range(
        self,
        stock_code: str,
        feature_type: str,
        start_date: str,
        end_date: str,
        params: Optional[Dict] = None,
    ) -> Optional[pd.DataFrame]:
        """Get cached features for a stock within a date range."""
        df = self.get(stock_code, feature_type, params)
        if df is None:
            return None

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            mask = (df["date"] >= start_date) & (df["date"] <= end_date)
            return df[mask]
        return df

    def get_latest(
        self,
        stock_code: str,
        feature_type: str,
        n: int = 30,
        params: Optional[Dict] = None,
    ) -> Optional[pd.DataFrame]:
        """Get latest N rows of cached features."""
        df = self.get(stock_code, feature_type, params)
        if df is None:
            return None

        if "date" in df.columns:
            df = df.sort_values("date")
        return df.tail(n)

    def refresh(self, stock_code: str) -> List[str]:
        """Refresh cache for a stock."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT DISTINCT feature_type FROM cache_metadata WHERE stock_code = ?",
                (stock_code,),
            )
            feature_types = [row[0] for row in cursor.fetchall()]

            conn.execute("DELETE FROM cache_data WHERE stock_code = ?", (stock_code,))
            conn.execute(
                "DELETE FROM cache_metadata WHERE stock_code = ?", (stock_code,)
            )
            conn.commit()

        return feature_types

    def delete(self, stock_code: str) -> List[str]:
        """Delete all cached features for a stock."""
        return self.refresh(stock_code)

    def delete_by_type(self, stock_code: str, feature_type: str) -> bool:
        """Delete cached features for a specific type."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM cache_data WHERE stock_code = ? AND feature_type = ?",
                (stock_code, feature_type),
            )
            conn.execute(
                "DELETE FROM cache_metadata WHERE stock_code = ? AND feature_type = ?",
                (stock_code, feature_type),
            )
            conn.commit()
            return cursor.rowcount > 0

    def clear_all(self) -> None:
        """Clear all cached features."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache_data")
            conn.execute("DELETE FROM cache_metadata")
            conn.commit()

    def list_cached(self, stock_code: Optional[str] = None) -> List[Dict]:
        """List all cached items, optionally filtered by stock code."""
        with sqlite3.connect(self.db_path) as conn:
            if stock_code:
                cursor = conn.execute(
                    """SELECT stock_code, feature_type, cache_key, params,
                              created_at, updated_at, row_count, column_count, size_bytes
                       FROM cache_metadata
                       WHERE stock_code = ?
                       ORDER BY updated_at DESC""",
                    (stock_code,),
                )
            else:
                cursor = conn.execute(
                    """SELECT stock_code, feature_type, cache_key, params,
                              created_at, updated_at, row_count, column_count, size_bytes
                       FROM cache_metadata
                       ORDER BY updated_at DESC"""
                )

            return [
                {
                    "stock_code": row[0],
                    "feature_type": row[1],
                    "cache_key": row[2],
                    "params": json.loads(row[3]) if row[3] else None,
                    "created_at": row[4],
                    "updated_at": row[5],
                    "row_count": row[6],
                    "column_count": row[7],
                    "size_bytes": row[8],
                }
                for row in cursor.fetchall()
            ]

    def get_cache_info(self, stock_code: str) -> Dict:
        """Get cache info for a specific stock."""
        items = self.list_cached(stock_code)
        total_size = sum(item.get("size_bytes", 0) for item in items)
        return {
            "stock_code": stock_code,
            "cached_types": list(set(item["feature_type"] for item in items)),
            "count": len(items),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "items": items,
        }

    def get_all_stocks(self) -> List[str]:
        """Get list of all stocks in cache."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT DISTINCT stock_code FROM cache_metadata")
            return [row[0] for row in cursor.fetchall()]

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM cache_metadata")
            total_items = cursor.fetchone()[0]

            cursor = conn.execute("SELECT SUM(size_bytes) FROM cache_metadata")
            total_size = cursor.fetchone()[0] or 0

            cursor = conn.execute(
                "SELECT COUNT(DISTINCT stock_code) FROM cache_metadata"
            )
            total_stocks = cursor.fetchone()[0]

            cursor = conn.execute(
                "SELECT feature_type, COUNT(*) FROM cache_metadata GROUP BY feature_type"
            )
            by_type = dict(cursor.fetchall())

            return {
                "total_items": total_items,
                "total_stocks": total_stocks,
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "by_type": by_type,
                "db_path": str(self.db_path),
            }


# Backward compatibility: keep the old class name
FeatureCache = SQLiteFeatureCache


# Global cache instance
_cache: Optional[SQLiteFeatureCache] = None


def get_cache(cache_dir: str = "cache") -> SQLiteFeatureCache:
    """Get global cache instance."""
    global _cache
    if _cache is None:
        _cache = SQLiteFeatureCache(cache_dir)
    return _cache
