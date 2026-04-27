"""XGBoost model for stock trading decisions."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not installed. Install with: pip install xgboost")

from .base import BaseModel


class XGBoostModel(BaseModel):
    """XGBoost model for stock trading decisions."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize XGBoostModel."""
        super().__init__(config)
        if not HAS_XGBOOST:
            raise ImportError(
                "XGBoost not installed. Install with: pip install xgboost"
            )

        self.config = config or {}
        self.n_estimators = self.config.get("n_estimators", 200)
        self.max_depth = self.config.get("max_depth", 6)
        self.learning_rate = self.config.get("learning_rate", 0.05)
        self.random_state = self.config.get("random_state", 42)
        self.max_features = 60

        self.model = None
        self.models: List = []  # Ensemble

    @property
    def model_name(self) -> str:
        """Return model name."""
        return "XGBoost"

    def _create_composite_labels(
        self,
        df: pd.DataFrame,
        forward_days: int = 5,
        threshold: float = 0.02,
        trend_weight: float = 0.3,
        momentum_weight: float = 0.3,
        market_weight: float = 0.2,
    ) -> pd.Series:
        """Create composite labels matching CatBoost's method for consistency."""
        future_returns = df["close"].shift(-forward_days) / df["close"] - 1

        # 1. Future return score (required condition)
        return_score = pd.Series(0.0, index=df.index)
        return_score[future_returns > threshold] = 1.0
        return_score[future_returns < -threshold] = -1.0

        # 2. Trend score (MA arrangement)
        trend_score = pd.Series(0.0, index=df.index)
        if "ma_bullish_arrange" in df.columns and "ma_bearish_arrange" in df.columns:
            trend_score[df["ma_bullish_arrange"] == 1] = 1.0
            trend_score[df["ma_bearish_arrange"] == 1] = -1.0

        # 3. Momentum score (RSI, MACD)
        momentum_score = pd.Series(0.0, index=df.index)
        if "rsi" in df.columns:
            momentum_score[df["rsi"] > 60] += 0.5
            momentum_score[df["rsi"] < 40] -= 0.5
        if "macd_hist" in df.columns:
            momentum_score[df["macd_hist"] > 0] += 0.5
            momentum_score[df["macd_hist"] < 0] -= 0.5
        momentum_score = momentum_score.clip(-1.0, 1.0)

        # 4. Market score
        market_score = pd.Series(0.0, index=df.index)
        if "index_returns" in df.columns:
            market_score[df["index_returns"] > 0.01] = 0.5
            market_score[df["index_returns"] < -0.01] = -0.5

        # Combine into composite signal
        return_weight = 1.0 - trend_weight - momentum_weight - market_weight
        composite = (
            return_score * return_weight
            + trend_score * trend_weight
            + momentum_score * momentum_weight
            + market_score * market_weight
        )

        # Create final labels
        labels = pd.Series(0, index=df.index)
        buy_threshold = 0.05
        sell_threshold = -0.05

        buy_condition = (composite >= buy_threshold) & (future_returns > threshold * 0.2)
        sell_condition = (composite <= sell_threshold) & (future_returns < -threshold * 0.2)

        labels[buy_condition] = 1
        labels[sell_condition] = -1

        return labels

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

        # Create labels using composite method
        labels = self._create_composite_labels(df, forward_days, threshold)

        valid_idx = ~labels.isna()
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
        """Train XGBoost model."""
        X, labels = self._prepare_data(df, forward_days, threshold)

        if len(X) < 20:
            raise ValueError(f"Insufficient data: {len(X)} samples")

        # Shift labels to 0, 1, 2 for XGBoost
        labels_shifted = labels + 1

        # Create ensemble
        n_models = kwargs.get("n_estimators_ensemble", 3)
        self.models = []

        for i in range(n_models):
            model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state + i,
                use_label_encoder=False,
                eval_metric="mlogloss",
                n_jobs=-1,
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
        """Predict trading action using soft voting (probability averaging)."""
        if not self.is_trained or not self.models:
            return "HOLD", 0.5

        X = self._get_features(df)
        if X.empty:
            return "HOLD", 0.5

        # Soft voting: average probabilities across all models
        all_probas = []
        for model in self.models:
            proba = model.predict_proba(X.iloc[[-1]])[0]
            all_probas.append(proba)

        # Average probabilities
        avg_proba = np.mean(all_probas, axis=0)

        # Get class with highest average probability
        pred_class = int(np.argmax(avg_proba))
        confidence = float(avg_proba[pred_class])

        # Convert back to action (classes: 0=SELL, 1=HOLD, 2=BUY)
        action_map = {2: "BUY", 1: "HOLD", 0: "SELL"}
        action = action_map.get(pred_class, "HOLD")

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

        # Map to buy/hold/sell (XGBoost classes: 0=SELL, 1=HOLD, 2=BUY)
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
