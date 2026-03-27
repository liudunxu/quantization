"""CatBoost model for stock trading decisions."""

from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import joblib
from ..utils.config import get_config


class StockTradingModel:
    """CatBoost model for stock trading decisions."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config().get_section('model').get('catboost', {})
        self.iterations = self.config.get('iterations', 500)
        self.depth = self.config.get('depth', 6)
        self.learning_rate = self.config.get('learning_rate', 0.03)
        self.l2_leaf_reg = self.config.get('l2_leaf_reg', 3)
        self.random_seed = self.config.get('random_seed', 42)
        self.model: Optional[CatBoostClassifier] = None
        self.feature_names: Optional[List[str]] = None

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for modeling."""
        # Drop non-feature columns
        drop_cols = ['date', 'stock_code', 'sector', 'industry']
        drop_cols = [c for c in drop_cols if c in df.columns]

        feature_df = df.drop(columns=drop_cols, errors='ignore')

        # Remove any remaining object or string columns
        for col in feature_df.columns:
            if feature_df[col].dtype == 'object' or feature_df[col].dtype == 'str':
                feature_df = feature_df.drop(columns=[col])

        # Fill NaN with median
        for col in feature_df.columns:
            if feature_df[col].isna().any():
                feature_df[col] = feature_df[col].fillna(feature_df[col].median())

        return feature_df

    def _create_labels(self, df: pd.DataFrame, forward_days: int = 5, threshold: float = 0.02) -> pd.Series:
        """Create labels for training.

        Label meanings:
        - 1: Buy (price will go up more than threshold)
        - 0: Hold
        - -1: Sell (price will go down more than threshold)
        """
        future_returns = df['close'].shift(-forward_days) / df['close'] - 1

        labels = pd.Series(0, index=df.index)
        labels[future_returns > threshold] = 1
        labels[future_returns < -threshold] = -1

        return labels

    def train(
        self,
        df: pd.DataFrame,
        forward_days: int = 5,
        threshold: float = 0.02,
        eval_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Train the model."""
        if df.empty or len(df) < 50:
            raise ValueError("Insufficient data for training")

        # Prepare features
        X = self._prepare_features(df)
        self.feature_names = list(X.columns)

        # Create labels
        labels = self._create_labels(df, forward_days, threshold)

        # Remove rows where forward returns couldn't be calculated (last forward_days rows)
        valid_idx = ~labels.isna()
        X = X[valid_idx]
        labels = labels[valid_idx]

        if len(X) < 30:
            raise ValueError("Insufficient valid samples for training")

        # Convert labels: -1->0, 0->1, 1->2 for CatBoost
        labels = (labels + 1).astype(int)

        # Train model
        self.model = CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            random_seed=self.random_seed,
            verbose=False,
            loss_function='MultiClass'
        )

        train_data = X
        train_labels = labels  # Already converted to 0,1,2

        if eval_df is not None and not eval_df.empty:
            eval_X = self._prepare_features(eval_df)
            eval_labels = self._create_labels(eval_df, forward_days, threshold)
            eval_valid_idx = ~eval_labels.isna()
            eval_X_valid = eval_X[eval_valid_idx]
            eval_labels_valid = (eval_labels[eval_valid_idx] + 1).astype(int)

            # Only use eval_set if we have valid samples
            if len(eval_X_valid) > 0:
                self.model.fit(
                    train_data, train_labels,
                    eval_set=(eval_X_valid, eval_labels_valid),
                    early_stopping_rounds=50,
                    verbose=False
                )
            else:
                self.model.fit(train_data, train_labels, verbose=False)
        else:
            self.model.fit(train_data, train_labels, verbose=False)

        # Calculate training metrics
        train_pred = self.model.predict(train_data)
        train_accuracy = (train_pred.flatten() == train_labels.values).mean()

        return {
            'train_accuracy': train_accuracy,
            'train_samples': len(train_data),
            'feature_count': len(self.feature_names),
            'label_distribution': {
                'buy': int((train_labels == 2).sum()),
                'hold': int((train_labels == 1).sum()),
                'sell': int((train_labels == 0).sum())
            }
        }

    def predict(self, df: pd.DataFrame) -> Tuple[int, float]:
        """Predict trading action for latest data.

        Returns:
            Tuple of (action, confidence)
            action: 1 (buy), 0 (hold), -1 (sell)
            confidence: probability of the predicted class
        """
        if self.model is None:
            raise ValueError("Model not trained")

        X = self._prepare_features(df)

        # Use only the latest row
        X_latest = X.iloc[[-1]]

        # Predict
        prediction = self.model.predict(X_latest)
        pred_class = int(prediction[0][0]) - 1  # Convert 0,1,2 back to -1,0,1

        # Get probability
        probabilities = self.model.predict_proba(X_latest)[0]
        confidence = float(probabilities[pred_class])

        return pred_class, confidence

    def predict_proba(self, df: pd.DataFrame) -> Dict[str, float]:
        """Get prediction probabilities for all classes."""
        if self.model is None:
            raise ValueError("Model not trained")

        X = self._prepare_features(df)
        X_latest = X.iloc[[-1]]

        probabilities = self.model.predict_proba(X_latest)[0]

        # Handle case where only 2 classes were learned
        if len(probabilities) == 2:
            return {
                'sell_probability': float(probabilities[0]),
                'hold_probability': 0.0,
                'buy_probability': float(probabilities[1])
            }
        else:
            return {
                'sell_probability': float(probabilities[0]),
                'hold_probability': float(probabilities[1]),
                'buy_probability': float(probabilities[2])
            }

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        if self.model is None:
            raise ValueError("Model not trained")

        importance = self.model.get_feature_importance()
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

    def save(self, path: str) -> None:
        """Save model to file."""
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'config': self.config
        }
        joblib.dump(model_data, path)

    def load(self, path: str) -> None:
        """Load model from file."""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.config = model_data.get('config', self.config)


# Global instance
_model: Optional[StockTradingModel] = None


def get_model() -> StockTradingModel:
    """Get global model instance."""
    global _model
    if _model is None:
        _model = StockTradingModel()
    return _model
