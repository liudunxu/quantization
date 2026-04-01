"""LightGBM model for stock trading decisions."""

from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.warning("LightGBM not installed. Install with: pip install lightgbm")

from .base import BaseModel

# Feature categories (same as CatBoost)
FEATURE_CATEGORIES = {
    "universal": [
        "returns",
        "momentum_",
        "rsi",
        "macd",
        "volume_ratio",
        "ma_",
        "volatility",
        "atr_",
        "bb_",
        "stoch_",
        "cci",
        "adx",
        "mfi",
        "turnover",
        "drawdown",
    ],
    "a_share": [
        "net_flow",
        "institutional_ratio",
        "main_net_flow",
        "super_large",
        "large_net_flow",
        "medium_net_flow",
        "small_net_flow",
    ],
    "hk": ["hsi", "hang_seng", "hk_"],
    "us": ["sp_", "nasdaq", "dow_", "^gspc", "qqq"],
    "market": ["index_", "market_", "alpha", "beta", "corr", "sector_"],
    "fundamental": [
        "pe_",
        "pb_",
        "roe_",
        "revenue_",
        "debt_",
        "profit_",
        "growth_",
        "dividend",
    ],
}


class LightGBMModel(BaseModel):
    """LightGBM model for stock trading decisions."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        if not HAS_LIGHTGBM:
            raise ImportError(
                "LightGBM not installed. Install with: pip install lightgbm"
            )

        self.config = config or {}
        self.n_estimators = self.config.get("n_estimators", 200)
        self.max_depth = self.config.get("max_depth", 6)
        self.learning_rate = self.config.get("learning_rate", 0.05)
        self.num_leaves = self.config.get("num_leaves", 31)
        self.random_state = self.config.get("random_state", 42)
        self.max_features = 60

        self.model = None
        self.models: List = []  # Ensemble

    @property
    def model_name(self) -> str:
        return "LightGBM"

    def _prepare_data(
        self, df: pd.DataFrame, forward_days: int = 5, threshold: float = 0.01
    ) -> tuple:
        """Prepare features and labels for training."""
        exclude_cols = [
            "date",
            "stock_code",
            "close",
            "open",
            "high",
            "low",
            "volume",
        ]
        feature_cols = [
            c
            for c in df.columns
            if c not in exclude_cols
            and df[c].dtype in [np.float64, np.int64, np.float32]
        ]

        if self.selected_features:
            feature_cols = [c for c in feature_cols if c in self.selected_features]

        if len(feature_cols) > self.max_features:
            feature_cols = feature_cols[: self.max_features]

        self.feature_names = feature_cols

        X = df[feature_cols].copy()
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)

        # Create labels
        future_returns = df["close"].pct_change(forward_days).shift(-forward_days)
        labels = np.where(
            future_returns > threshold, 1, np.where(future_returns < -threshold, -1, 0)
        )

        valid_idx = ~np.isnan(future_returns)
        X = X[valid_idx]
        labels = labels[valid_idx]

        return X, labels

    def train(
        self,
        df: pd.DataFrame,
        forward_days: int = 5,
        threshold: float = 0.01,
        eval_df: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Train LightGBM model."""
        X, labels = self._prepare_data(df, forward_days, threshold)

        if len(X) < 20:
            raise ValueError(f"Insufficient data: {len(X)} samples")

        # Shift labels to 0, 1, 2 for LightGBM
        labels_shifted = labels + 1

        # Create ensemble
        n_models = kwargs.get("n_estimators_ensemble", 3)
        self.models = []

        for i in range(n_models):
            model = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                random_state=self.random_state + i,
                class_weight="balanced",
                n_jobs=-1,
                verbose=-1,
            )
            model.fit(X, labels_shifted)
            self.models.append(model)

        self.model = self.models[0]  # Primary model
        self.is_trained = True

        # Training metrics
        train_pred = self.model.predict(X)
        train_accuracy = np.mean(train_pred == labels_shifted)

        return {
            "train_accuracy": train_accuracy,
            "feature_count": len(self.feature_names),
            "label_distribution": {
                "buy": int(np.sum(labels == 1)),
                "sell": int(np.sum(labels == -1)),
                "hold": int(np.sum(labels == 0)),
            },
        }

    def predict(self, df: pd.DataFrame) -> tuple:
        """Predict trading action using ensemble voting."""
        if not self.is_trained or not self.models:
            return "HOLD", 0.5

        X = self._get_features(df)
        if X.empty:
            return "HOLD", 0.5

        # Ensemble voting
        predictions = []
        for model in self.models:
            pred = model.predict(X.iloc[[-1]])[0]
            predictions.append(pred)

        # Majority voting
        from collections import Counter

        vote_counts = Counter(predictions)
        final_pred = vote_counts.most_common(1)[0][0]

        # Calculate confidence
        confidence = vote_counts[final_pred] / len(predictions)

        # Convert back to action
        action_map = {2: "BUY", 1: "HOLD", 0: "SELL"}
        action = action_map.get(final_pred, "HOLD")

        return action, confidence

    def predict_proba(self, df: pd.DataFrame) -> dict:
        """Predict trading action probabilities."""
        if not self.is_trained or not self.models:
            return {
                "buy_probability": 0.33,
                "hold_probability": 0.34,
                "sell_probability": 0.33,
            }

        X = self._get_features(df)
        if X.empty:
            return {
                "buy_probability": 0.33,
                "hold_probability": 0.34,
                "sell_probability": 0.33,
            }

        # Average probabilities across ensemble
        all_probas = []
        for model in self.models:
            proba = model.predict_proba(X.iloc[[-1]])[0]
            all_probas.append(proba)

        avg_proba = np.mean(all_probas, axis=0)

        # Map to buy/hold/sell (LightGBM classes: 0=SELL, 1=HOLD, 2=BUY)
        return {
            "sell_probability": float(avg_proba[0]),
            "hold_probability": float(avg_proba[1]),
            "buy_probability": float(avg_proba[2]),
        }

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        if not self.is_trained or not self.model:
            return pd.DataFrame(columns=["feature", "importance"])

        importance = self.model.feature_importances_
        df = pd.DataFrame(
            {"feature": self.feature_names, "importance": importance}
        ).sort_values("importance", ascending=False)

        return df

    def _get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get features for prediction."""
        if not self.feature_names:
            return pd.DataFrame()

        available_features = [f for f in self.feature_names if f in df.columns]
        if not available_features:
            return pd.DataFrame()

        X = df[available_features].copy()
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)

        return X
