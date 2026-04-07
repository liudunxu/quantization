"""Configuration loader."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Config:
    """Configuration manager."""

    _instance: Optional["Config"] = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        """Create singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize Config singleton."""
        if not self._config:
            self.load()

    def load(self, config_path: Optional[str] = None) -> None:
        """Load configuration from YAML file."""
        if config_path is None:
            # Find config relative to project root
            config_path = (
                Path(__file__).parent.parent.parent / "configs" / "config.yaml"
            )

        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self._config = yaml.safe_load(f)
        else:
            self._config = self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "stock": {
                "a_share": {"sz_prefix": "SZ", "sh_prefix": "SH"},
                "hk_prefix": "HK",
                "us_prefix": "",
            },
            "data": {
                "backtest_days": 30,
                "train_days": 365,
                "cache_dir": "cache",
                "data_dir": "data",
            },
            "features": {
                "fundamental": {"enabled": True, "lookback_days": 30},
                "technical": {
                    "enabled": True,
                    "ma_periods": [5, 10, 20, 60],
                    "rsi_period": 14,
                    "macd_fast": 12,
                    "macd_slow": 26,
                    "macd_signal": 9,
                    "bollinger_period": 20,
                    "bollinger_std": 2,
                },
                "market": {"enabled": True},
            },
            "model": {
                "catboost": {
                    "iterations": 500,
                    "depth": 6,
                    "learning_rate": 0.03,
                    "l2_leaf_reg": 3,
                    "random_seed": 42,
                }
            },
            "backtest": {
                "initial_cash": 100000,
                "commission": 0.001,
                "slippage": 0.001,
            },
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key."""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    def set(self, key: str, value: Any) -> None:
        """Set configuration value by dot-notation key."""
        keys = key.split(".")
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        return self._config.get(section, {})


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global config instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config
