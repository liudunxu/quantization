"""Multi-model ensemble for stock trading decisions.

Combines CatBoost, LightGBM, and XGBoost predictions using weighted voting.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from .base import BaseModel
from .trainer import StockTradingModel


class MultiModelEnsemble(BaseModel):
    """Multi-model ensemble combining CatBoost, LightGBM, and XGBoost."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize MultiModelEnsemble."""
        super().__init__(config)
        self.config = config or {}
        self.models: Dict[str, BaseModel] = {}
        self.model_weights: Dict[str, float] = self.config.get(
            "model_weights",
            {"catboost": 0.4, "lightgbm": 0.3, "xgboost": 0.3},
        )
        self.model_accuracies: Dict[str, float] = {}  # Track model accuracies
        self.available_models = self._check_available_models()

    @property
    def model_name(self) -> str:
        """Return model name."""
        return "MultiModelEnsemble"

    def _check_available_models(self) -> Dict[str, bool]:
        """Check which models are available."""
        available = {"catboost": True}  # CatBoost is always available

        try:
            import lightgbm  # noqa: F401

            available["lightgbm"] = True
        except ImportError:
            available["lightgbm"] = False
            logger.debug("LightGBM not available")

        try:
            import xgboost  # noqa: F401

            available["xgboost"] = True
        except ImportError:
            available["xgboost"] = False
            logger.debug("XGBoost not available")

        return available

    def train(
        self,
        df: pd.DataFrame,
        forward_days: int = 5,
        threshold: float = 0.01,
        eval_df: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Train all available models with adaptive weight adjustment."""
        results = {}

        # Train CatBoost
        try:
            logger.info("Training CatBoost model...")
            catboost_model = StockTradingModel(self.config.get("catboost", {}))
            catboost_result = catboost_model.train(
                df, forward_days, threshold, eval_df, **kwargs
            )
            self.models["catboost"] = catboost_model
            self.model_accuracies["catboost"] = catboost_result["train_accuracy"]
            results["catboost"] = catboost_result
            logger.info(f"CatBoost accuracy: {catboost_result['train_accuracy']:.2%}")
        except Exception as e:
            logger.error(f"CatBoost training failed: {e}")

        # Train LightGBM
        if self.available_models.get("lightgbm"):
            try:
                logger.info("Training LightGBM model...")
                from .lgbm_model import LightGBMModel

                lgbm_model = LightGBMModel(self.config.get("lightgbm", {}))
                lgbm_result = lgbm_model.train(
                    df, forward_days, threshold, eval_df, **kwargs
                )
                self.models["lightgbm"] = lgbm_model
                self.model_accuracies["lightgbm"] = lgbm_result["train_accuracy"]
                results["lightgbm"] = lgbm_result
                logger.info(f"LightGBM accuracy: {lgbm_result['train_accuracy']:.2%}")
            except Exception as e:
                logger.error(f"LightGBM training failed: {e}")

        # Train XGBoost
        if self.available_models.get("xgboost"):
            try:
                logger.info("Training XGBoost model...")
                from .xgboost_model import XGBoostModel

                xgb_model = XGBoostModel(self.config.get("xgboost", {}))
                xgb_result = xgb_model.train(
                    df, forward_days, threshold, eval_df, **kwargs
                )
                self.models["xgboost"] = xgb_model
                self.model_accuracies["xgboost"] = xgb_result["train_accuracy"]
                results["xgboost"] = xgb_result
                logger.info(f"XGBoost accuracy: {xgb_result['train_accuracy']:.2%}")
            except Exception as e:
                logger.error(f"XGBoost training failed: {e}")

        self.is_trained = len(self.models) > 0

        # Update model weights based on training accuracy (adaptive weighting)
        self._update_weights_based_on_accuracy()

        # Calculate average accuracy
        avg_accuracy = np.mean([r["train_accuracy"] for r in results.values()])

        return {
            "train_accuracy": avg_accuracy,
            "models_trained": list(self.models.keys()),
            "model_results": results,
            "model_weights": self.model_weights,
        }

    def _update_weights_based_on_accuracy(self):
        """Update model weights based on training accuracy (adaptive weighting)."""
        if not self.model_accuracies:
            return

        # Calculate weights proportional to accuracy
        total_accuracy = sum(self.model_accuracies.values())
        if total_accuracy > 0:
            # Normalize weights to sum to 1.0
            for name, accuracy in self.model_accuracies.items():
                self.model_weights[name] = accuracy / total_accuracy
            
            # Ensure we don't deviate too much from default weights
            # Blend with default weights (70% adaptive, 30% default)
            default_weights = {"catboost": 0.4, "lightgbm": 0.3, "xgboost": 0.3}
            for name in self.model_weights:
                if name in default_weights:
                    self.model_weights[name] = (
                        0.7 * self.model_weights[name] + 
                        0.3 * default_weights[name]
                    )
            
            # Renormalize to ensure sum is 1.0
            total_weight = sum(self.model_weights.values())
            if total_weight > 0:
                for name in self.model_weights:
                    self.model_weights[name] /= total_weight
            
            logger.info(f"Updated model weights: {self.model_weights}")

    def predict(self, df: pd.DataFrame) -> tuple:
        """Predict using weighted voting across all models."""
        if not self.is_trained:
            return "HOLD", 0.5

        predictions = {}
        confidences = {}

        for name, model in self.models.items():
            try:
                action, confidence = model.predict(df)
                # Normalize action to string format
                if isinstance(action, int):
                    # Convert integer action to string (CatBoost format: -1, 0, 1)
                    action_map = {1: "BUY", 0: "HOLD", -1: "SELL"}
                    action = action_map.get(action, "HOLD")
                predictions[name] = action
                confidences[name] = confidence
            except Exception as e:
                logger.warning(f"Prediction failed for {name}: {e}")
                predictions[name] = "HOLD"
                confidences[name] = 0.5

        if not predictions:
            return "HOLD", 0.5

        # Weighted voting
        action_scores = {"BUY": 0.0, "HOLD": 0.0, "SELL": 0.0}

        for name, action in predictions.items():
            weight = self.model_weights.get(name, 0.3)
            confidence = confidences.get(name, 0.5)
            action_scores[action] += weight * confidence

        # Get final action
        final_action = max(action_scores, key=action_scores.get)

        # Calculate confidence
        total_score = sum(action_scores.values())
        if total_score > 0:
            final_confidence = action_scores[final_action] / total_score
        else:
            final_confidence = 0.5

        return final_action, final_confidence

    def predict_proba(self, df: pd.DataFrame) -> dict:
        """Predict probabilities using weighted average across all models."""
        if not self.is_trained:
            return {
                "buy_probability": 0.33,
                "hold_probability": 0.34,
                "sell_probability": 0.33,
            }

        all_probas = []
        weights = []

        for name, model in self.models.items():
            try:
                proba = model.predict_proba(df)
                all_probas.append(proba)
                weights.append(self.model_weights.get(name, 0.3))
            except Exception as e:
                logger.warning(f"Probability prediction failed for {name}: {e}")

        if not all_probas:
            return {
                "buy_probability": 0.33,
                "hold_probability": 0.34,
                "sell_probability": 0.33,
            }

        # Weighted average
        total_weight = sum(weights)
        avg_buy = (
            sum(p["buy_probability"] * w for p, w in zip(all_probas, weights))
            / total_weight
        )
        avg_hold = (
            sum(p["hold_probability"] * w for p, w in zip(all_probas, weights))
            / total_weight
        )
        avg_sell = (
            sum(p["sell_probability"] * w for p, w in zip(all_probas, weights))
            / total_weight
        )

        return {
            "buy_probability": avg_buy,
            "hold_probability": avg_hold,
            "sell_probability": avg_sell,
        }

    def get_feature_importance(self) -> pd.DataFrame:
        """Get combined feature importance from all models."""
        if not self.is_trained:
            return pd.DataFrame(columns=["feature", "importance"])

        all_importance = []

        for name, model in self.models.items():
            try:
                imp_df = model.get_feature_importance()
                if not imp_df.empty:
                    imp_df["model"] = name
                    imp_df["weight"] = self.model_weights.get(name, 0.3)
                    all_importance.append(imp_df)
            except Exception as e:
                logger.warning(f"Feature importance failed for {name}: {e}")

        if not all_importance:
            return pd.DataFrame(columns=["feature", "importance"])

        # Combine and weight
        combined = pd.concat(all_importance)
        combined["weighted_importance"] = combined["importance"] * combined["weight"]

        # Aggregate by feature
        result = (
            combined.groupby("feature")["weighted_importance"]
            .sum()
            .reset_index()
            .sort_values("weighted_importance", ascending=False)
        )
        result.columns = ["feature", "importance"]

        return result

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about trained models."""
        return {
            "available_models": self.available_models,
            "trained_models": list(self.models.keys()),
            "model_weights": self.model_weights,
            "is_trained": self.is_trained,
        }
