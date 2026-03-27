"""Feature caching system with file-based storage."""

import os
import json
import hashlib
from pathlib import Path
from typing import Any, Optional, Dict, List
import pandas as pd
import joblib
from datetime import datetime


class FeatureCache:
    """File-based feature cache that never expires."""

    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_file = self.cache_dir / "manifest.json"
        self._load_manifest()

    def _load_manifest(self) -> None:
        """Load cache manifest."""
        if self.manifest_file.exists():
            with open(self.manifest_file, 'r') as f:
                self.manifest = json.load(f)
        else:
            self.manifest = {}

    def _save_manifest(self) -> None:
        """Save cache manifest."""
        with open(self.manifest_file, 'w') as f:
            json.dump(self.manifest, f, indent=2)

    def _get_cache_path(self, stock_code: str, feature_type: str) -> Path:
        """Get cache file path for a stock and feature type."""
        return self.cache_dir / f"{stock_code}_{feature_type}.parquet"

    def _compute_key(self, stock_code: str, feature_type: str, params: Optional[Dict] = None) -> str:
        """Compute cache key."""
        key_str = f"{stock_code}_{feature_type}"
        if params:
            key_str += f"_{hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()}"
        return key_str

    def get(self, stock_code: str, feature_type: str, params: Optional[Dict] = None) -> Optional[pd.DataFrame]:
        """Get cached features for a stock."""
        key = self._compute_key(stock_code, feature_type, params)

        if key not in self.manifest:
            return None

        cache_path = self._get_cache_path(stock_code, feature_type)
        if not cache_path.exists():
            return None

        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            return None

    def set(self, stock_code: str, feature_type: str, df: pd.DataFrame, params: Optional[Dict] = None) -> None:
        """Cache features for a stock."""
        key = self._compute_key(stock_code, feature_type, params)
        cache_path = self._get_cache_path(stock_code, feature_type)

        df.to_parquet(cache_path)

        self.manifest[key] = {
            "stock_code": stock_code,
            "feature_type": feature_type,
            "params": params,
            "cached_at": datetime.now().isoformat(),
            "path": str(cache_path)
        }
        self._save_manifest()

    def refresh(self, stock_code: str) -> List[str]:
        """Refresh cache for a stock (returns list of feature types that were cached)."""
        # This is a placeholder - actual refresh happens in feature engineering
        refreshed = []
        for feature_type in ['fundamental', 'technical', 'market', 'industry']:
            cache_path = self._get_cache_path(stock_code, feature_type)
            if cache_path.exists():
                cache_path.unlink()
                refreshed.append(feature_type)

        # Remove from manifest
        keys_to_remove = [k for k, v in self.manifest.items() if v.get('stock_code') == stock_code]
        for key in keys_to_remove:
            del self.manifest[key]
        self._save_manifest()

        return refreshed

    def delete(self, stock_code: str) -> List[str]:
        """Delete all cached features for a stock."""
        deleted = []
        for feature_type in ['fundamental', 'technical', 'market', 'industry']:
            cache_path = self._get_cache_path(stock_code, feature_type)
            if cache_path.exists():
                cache_path.unlink()
                deleted.append(feature_type)

        # Remove from manifest
        keys_to_remove = [k for k, v in self.manifest.items() if v.get('stock_code') == stock_code]
        for key in keys_to_remove:
            del self.manifest[key]
        self._save_manifest()

        return deleted

    def clear_all(self) -> None:
        """Clear all cached features."""
        for cache_file in self.cache_dir.glob("*.parquet"):
            cache_file.unlink()

        self.manifest = {}
        self._save_manifest()

    def list_cached(self) -> List[Dict]:
        """List all cached items."""
        return list(self.manifest.values())

    def get_cache_info(self, stock_code: str) -> Dict:
        """Get cache info for a specific stock."""
        items = [v for k, v in self.manifest.items() if v.get('stock_code') == stock_code]
        return {
            "stock_code": stock_code,
            "cached_types": [item['feature_type'] for item in items],
            "count": len(items),
            "items": items
        }


# Global cache instance
_cache: Optional[FeatureCache] = None


def get_cache(cache_dir: str = "cache") -> FeatureCache:
    """Get global cache instance."""
    global _cache
    if _cache is None:
        _cache = FeatureCache(cache_dir)
    return _cache
