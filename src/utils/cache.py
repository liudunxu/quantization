"""Feature caching system with SQLite and Redis (Upstash) storage."""

import hashlib
import io
import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class SQLiteFeatureCache:
    """SQLite-based feature cache with fine-grained control."""

    # Default cache expiration in hours (24 hours = 1 day)
    DEFAULT_TTL_HOURS = 24

    def __init__(self, cache_dir: str = "cache", ttl_hours: Optional[int] = None):
        """Initialize FeatureCache."""
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


class RedisFeatureCache:
    """Redis-based feature cache using Upstash REST API.

    Uses Upstash Redis REST API (no redis-py dependency required).
    DataFrames are serialized to Parquet bytes, then base64-encoded for storage.
    """

    DEFAULT_TTL_SECONDS = 24 * 3600  # 24 hours

    def __init__(
        self,
        url: Optional[str] = None,
        token: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
    ):
        """Initialize Redis cache with Upstash REST API.

        Args:
            url: Upstash Redis REST URL (env: UPSTASH_REDIS_REST_URL)
            token: Upstash Redis REST token (env: UPSTASH_REDIS_REST_TOKEN)
            ttl_seconds: TTL in seconds, defaults to 24 hours
        """
        self.url = url or os.environ.get("UPSTASH_REDIS_REST_URL", "")
        self.token = token or os.environ.get("UPSTASH_REDIS_REST_TOKEN", "")
        self.ttl_seconds = ttl_seconds or self.DEFAULT_TTL_SECONDS
        self.key_prefix = "quarnt:cache:"
        self._available = bool(self.url and self.token)

        if not self._available:
            logger.debug("Upstash Redis cache disabled: missing URL or token")

    def _is_available(self) -> bool:
        """Check if Redis is available."""
        return self._available

    def _make_key(
        self, stock_code: str, feature_type: str, params: Optional[Dict] = None
    ) -> str:
        """Build Redis key."""
        raw_key = f"{stock_code}_{feature_type}"
        if params:
            raw_key += f"_{hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()}"
        return f"{self.key_prefix}{raw_key}"

    def _meta_key(self, base_key: str) -> str:
        return f"{base_key}:meta"

    def _request(self, method: str, *commands: list) -> Optional[Any]:
        """Send REST request to Upstash Redis.

        Args:
            method: HTTP method (GET/POST)
            *commands: Redis commands as lists, e.g. ["GET", "key"]

        Returns:
            Response result or None on failure
        """
        if not self._available:
            return None

        import urllib.error
        import urllib.request

        url = f"{self.url}/pipeline" if len(commands) > 1 else f"{self.url}"

        if len(commands) == 1:
            payload = json.dumps(commands[0]).encode()
        else:
            payload = json.dumps(list(commands)).encode()

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.token}",
            },
            method=method,
        )

        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                # Upstash returns {"result": [...]} for pipeline, or {"result": value} for single
                if isinstance(data, dict) and "result" in data:
                    return data["result"]
                return data
        except Exception as e:
            logger.warning("Upstash Redis request failed: %s", e)
            self._available = False
            return None

    def _serialize_df(self, df: pd.DataFrame) -> str:
        """Serialize DataFrame to base64-encoded Parquet string."""
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=True)
        import base64

        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _deserialize_df(self, data: str) -> pd.DataFrame:
        """Deserialize base64-encoded Parquet string to DataFrame."""
        import base64

        raw = base64.b64decode(data)
        buffer = io.BytesIO(raw)
        return pd.read_parquet(buffer)

    def get(
        self, stock_code: str, feature_type: str, params: Optional[Dict] = None
    ) -> Optional[pd.DataFrame]:
        """Get cached features from Redis."""
        if not self._is_available():
            return None

        key = self._make_key(stock_code, feature_type, params)
        meta_key = self._meta_key(key)

        try:
            result = self._request("POST", ["GET", key], ["GET", meta_key])
            if result is None or not isinstance(result, list) or len(result) < 2:
                return None

            data_b64 = result[0]
            meta_json = result[1]

            if data_b64 is None:
                return None

            df = self._deserialize_df(data_b64)

            if meta_json:
                meta = (
                    json.loads(meta_json) if isinstance(meta_json, str) else meta_json
                )
                stored_at = meta.get("stored_at", 0)
                if (time.time() - stored_at) > self.ttl_seconds:
                    self._request("POST", ["DEL", key], ["DEL", meta_key])
                    return None

            return df
        except Exception as e:
            logger.warning("Redis get failed: %s", e)
            return None

    def set(
        self,
        stock_code: str,
        feature_type: str,
        df: pd.DataFrame,
        params: Optional[Dict] = None,
    ) -> None:
        """Cache features in Redis."""
        if not self._is_available():
            return

        key = self._make_key(stock_code, feature_type, params)
        meta_key = self._meta_key(key)
        data_b64 = self._serialize_df(df)
        meta = json.dumps(
            {
                "stock_code": stock_code,
                "feature_type": feature_type,
                "params": params,
                "row_count": len(df),
                "column_count": len(df.columns),
                "stored_at": time.time(),
            }
        )

        try:
            self._request(
                "POST",
                ["SET", key, data_b64, "EX", self.ttl_seconds],
                ["SET", meta_key, meta, "EX", self.ttl_seconds],
            )
        except Exception as e:
            logger.warning("Redis set failed: %s", e)

    def merge_and_update(
        self,
        stock_code: str,
        feature_type: str,
        new_df: pd.DataFrame,
        params: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """Merge new data with cached data and update Redis."""
        cached_df = self.get(stock_code, feature_type, params)

        if cached_df is None or cached_df.empty:
            self.set(stock_code, feature_type, new_df, params)
            return new_df

        combined = pd.concat([cached_df, new_df], ignore_index=True)
        if "date" in combined.columns:
            combined["date"] = pd.to_datetime(combined["date"])
            combined = combined.drop_duplicates(subset=["date"], keep="last")
            combined = combined.sort_values("date").reset_index(drop=True)

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

    def get_by_date_range(
        self,
        stock_code: str,
        feature_type: str,
        start_date: str,
        end_date: str,
        params: Optional[Dict] = None,
    ) -> Optional[pd.DataFrame]:
        """Get cached features within a date range."""
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
        """Get latest N rows."""
        df = self.get(stock_code, feature_type, params)
        if df is None:
            return None
        if "date" in df.columns:
            df = df.sort_values("date")
        return df.tail(n)

    def refresh(self, stock_code: str) -> List[str]:
        """Delete all cached features for a stock. Returns deleted feature types."""
        if not self._is_available():
            return []

        feature_types = []
        try:
            keys = self._request("POST", ["KEYS", f"{self.key_prefix}{stock_code}_*"])
            if keys and isinstance(keys, list):
                for key in keys:
                    if key and isinstance(key, str) and ":meta" not in key:
                        feature_types.append(
                            key.split(":")[-1].split("_")[1]
                            if len(key.split(":")[-1].split("_")) > 1
                            else "unknown"
                        )
                if keys:
                    self._request("POST", *([["DEL", k] for k in keys]))
        except Exception as e:
            logger.warning("Redis refresh failed: %s", e)
        return list(set(feature_types))

    def delete(self, stock_code: str) -> List[str]:
        """Delete all cached features for a stock."""
        return self.refresh(stock_code)

    def delete_by_type(self, stock_code: str, feature_type: str) -> bool:
        """Delete cached features for a specific type."""
        if not self._is_available():
            return False
        key = self._make_key(stock_code, feature_type)
        meta_key = self._meta_key(key)
        try:
            self._request("POST", ["DEL", key], ["DEL", meta_key])
            return True
        except Exception as e:
            logger.warning("Redis delete_by_type failed: %s", e)
            return False

    def clear_all(self) -> None:
        """Clear all cached features."""
        if not self._is_available():
            return
        try:
            keys = self._request("POST", ["KEYS", f"{self.key_prefix}*"])
            if keys and isinstance(keys, list) and keys:
                self._request("POST", *([["DEL", k] for k in keys]))
        except Exception as e:
            logger.warning("Redis clear_all failed: %s", e)

    def list_cached(self, stock_code: Optional[str] = None) -> List[Dict]:
        """List all cached items."""
        if not self._is_available():
            return []

        results = []
        try:
            pattern = (
                f"{self.key_prefix}{stock_code}_*"
                if stock_code
                else f"{self.key_prefix}*"
            )
            keys = self._request("POST", ["KEYS", pattern])
            if keys and isinstance(keys, list):
                for key in keys:
                    if key and isinstance(key, str) and ":meta" not in key:
                        meta_key = self._meta_key(key)
                        meta_result = self._request("POST", ["GET", meta_key])
                        meta = {}
                        if meta_result:
                            try:
                                meta = (
                                    json.loads(meta_result)
                                    if isinstance(meta_result, str)
                                    else meta_result
                                )
                            except Exception:
                                pass
                        results.append(
                            {
                                "stock_code": meta.get("stock_code", "unknown"),
                                "feature_type": meta.get("feature_type", "unknown"),
                                "cache_key": key,
                                "params": meta.get("params"),
                                "created_at": None,
                                "updated_at": None,
                                "row_count": meta.get("row_count", 0),
                                "column_count": meta.get("column_count", 0),
                                "size_bytes": 0,
                            }
                        )
        except Exception as e:
            logger.warning("Redis list_cached failed: %s", e)
        return results

    def get_cache_info(self, stock_code: str) -> Dict:
        """Get cache info for a specific stock."""
        items = self.list_cached(stock_code)
        return {
            "stock_code": stock_code,
            "cached_types": list(set(item["feature_type"] for item in items)),
            "count": len(items),
            "total_size_bytes": 0,
            "total_size_mb": 0,
            "items": items,
        }

    def get_all_stocks(self) -> List[str]:
        """Get list of all stocks in cache."""
        if not self._is_available():
            return []
        stocks = set()
        try:
            keys = self._request("POST", ["KEYS", f"{self.key_prefix}*"])
            if keys and isinstance(keys, list):
                for key in keys:
                    if key and isinstance(key, str) and ":meta" not in key:
                        parts = key.replace(self.key_prefix, "").split("_")
                        if parts:
                            stocks.add(parts[0])
        except Exception as e:
            logger.warning("Redis get_all_stocks failed: %s", e)
        return list(stocks)

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        items = self.list_cached()
        return {
            "total_items": len(items),
            "total_stocks": len(set(item["stock_code"] for item in items)),
            "total_size_bytes": 0,
            "total_size_mb": 0,
            "by_type": {},
            "backend": "redis",
        }


class HybridCache:
    """Fail-safe hybrid cache: Redis (primary) + SQLite (fallback).

    Read strategy:
        1. Try Redis first
        2. If Redis miss or fails, fall back to SQLite
        3. If Redis read succeeded but missed, backfill from SQLite -> Redis

    Write strategy:
        1. Always write to Redis (fire-and-forget, non-blocking on failure)
        2. Always write to SQLite as backup
    """

    def __init__(
        self, redis_cache: RedisFeatureCache, sqlite_cache: SQLiteFeatureCache
    ):
        self.redis = redis_cache
        self.sqlite = sqlite_cache

    def get(
        self, stock_code: str, feature_type: str, params: Optional[Dict] = None
    ) -> Optional[pd.DataFrame]:
        """Get from Redis first, fallback to SQLite."""
        redis_hit = False
        try:
            df = self.redis.get(stock_code, feature_type, params)
            if df is not None:
                return df
            redis_hit = True
        except Exception as e:
            logger.debug("Redis get failed, falling back to SQLite: %s", e)

        # Fallback to SQLite
        df = self.sqlite.get(stock_code, feature_type, params)

        # Backfill: if Redis was reachable but missed, and we got SQLite data, write to Redis
        if redis_hit and df is not None:
            try:
                self.redis.set(stock_code, feature_type, df, params)
            except Exception:
                pass

        return df

    def set(
        self,
        stock_code: str,
        feature_type: str,
        df: pd.DataFrame,
        params: Optional[Dict] = None,
    ) -> None:
        """Write to both Redis and SQLite."""
        try:
            self.redis.set(stock_code, feature_type, df, params)
        except Exception as e:
            logger.debug("Redis set failed (SQLite will still be used): %s", e)

        try:
            self.sqlite.set(stock_code, feature_type, df, params)
        except Exception as e:
            logger.error("SQLite set failed: %s", e)

    def merge_and_update(
        self,
        stock_code: str,
        feature_type: str,
        new_df: pd.DataFrame,
        params: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """Merge and update both caches."""
        cached_df = self.get(stock_code, feature_type, params)

        if cached_df is None or cached_df.empty:
            self.set(stock_code, feature_type, new_df, params)
            return new_df

        combined = pd.concat([cached_df, new_df], ignore_index=True)
        if "date" in combined.columns:
            combined["date"] = pd.to_datetime(combined["date"])
            combined = combined.drop_duplicates(subset=["date"], keep="last")
            combined = combined.sort_values("date").reset_index(drop=True)

        self.set(stock_code, feature_type, combined, params)
        return combined

    def get_latest_date(
        self, stock_code: str, feature_type: str, params: Optional[Dict] = None
    ) -> Optional[pd.Timestamp]:
        """Get latest date from cache."""
        df = self.get(stock_code, feature_type, params)
        if df is None or df.empty or "date" not in df.columns:
            return None
        df["date"] = pd.to_datetime(df["date"])
        return df["date"].max()

    def get_by_date_range(
        self,
        stock_code: str,
        feature_type: str,
        start_date: str,
        end_date: str,
        params: Optional[Dict] = None,
    ) -> Optional[pd.DataFrame]:
        """Get cached features within a date range."""
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
        """Get latest N rows."""
        df = self.get(stock_code, feature_type, params)
        if df is None:
            return None
        if "date" in df.columns:
            df = df.sort_values("date")
        return df.tail(n)

    def refresh(self, stock_code: str) -> List[str]:
        """Refresh cache for a stock."""
        types = []
        try:
            types.extend(self.redis.refresh(stock_code))
        except Exception as e:
            logger.debug("Redis refresh failed: %s", e)
        try:
            types.extend(self.sqlite.refresh(stock_code))
        except Exception as e:
            logger.error("SQLite refresh failed: %s", e)
        return list(set(types))

    def delete(self, stock_code: str) -> List[str]:
        """Delete all cached features for a stock."""
        return self.refresh(stock_code)

    def delete_by_type(self, stock_code: str, feature_type: str) -> bool:
        """Delete cached features for a specific type."""
        success = False
        try:
            self.redis.delete_by_type(stock_code, feature_type)
            success = True
        except Exception as e:
            logger.debug("Redis delete_by_type failed: %s", e)
        try:
            self.sqlite.delete_by_type(stock_code, feature_type)
            success = True
        except Exception as e:
            logger.error("SQLite delete_by_type failed: %s", e)
        return success

    def clear_all(self) -> None:
        """Clear all cached features."""
        try:
            self.redis.clear_all()
        except Exception as e:
            logger.debug("Redis clear_all failed: %s", e)
        try:
            self.sqlite.clear_all()
        except Exception as e:
            logger.error("SQLite clear_all failed: %s", e)

    def list_cached(self, stock_code: Optional[str] = None) -> List[Dict]:
        """List all cached items (from SQLite, as the source of truth)."""
        return self.sqlite.list_cached(stock_code)

    def get_cache_info(self, stock_code: str) -> Dict:
        """Get cache info for a specific stock."""
        return self.sqlite.get_cache_info(stock_code)

    def get_all_stocks(self) -> List[str]:
        """Get list of all stocks in cache (union of both)."""
        stocks = set()
        try:
            stocks.update(self.redis.get_all_stocks())
        except Exception:
            pass
        try:
            stocks.update(self.sqlite.get_all_stocks())
        except Exception:
            pass
        return list(stocks)

    def get_stats(self) -> Dict:
        """Get cache statistics from both backends."""
        stats = {"backend": "hybrid"}
        try:
            stats["redis"] = self.redis.get_stats()
        except Exception as e:
            stats["redis"] = {"error": str(e)}
        try:
            stats["sqlite"] = self.sqlite.get_stats()
        except Exception as e:
            stats["sqlite"] = {"error": str(e)}
        return stats


# Global cache instance
_cache: Optional[Any] = None


def get_cache(cache_dir: str = "cache") -> Any:
    """Get global cache instance.

    Returns HybridCache (Redis + SQLite) if Upstash is configured,
    otherwise returns SQLiteFeatureCache for backward compatibility.
    """
    global _cache
    if _cache is None:
        redis_url = os.environ.get("UPSTASH_REDIS_REST_URL", "")
        redis_token = os.environ.get("UPSTASH_REDIS_REST_TOKEN", "")

        if redis_url and redis_token:
            redis_cache = RedisFeatureCache(url=redis_url, token=redis_token)
            sqlite_cache = SQLiteFeatureCache(cache_dir)
            _cache = HybridCache(redis_cache, sqlite_cache)
            logger.info("Using hybrid cache (Redis + SQLite)")
        else:
            _cache = SQLiteFeatureCache(cache_dir)
            logger.info(
                "Using SQLite cache (set UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN to enable Redis)"
            )
    return _cache
